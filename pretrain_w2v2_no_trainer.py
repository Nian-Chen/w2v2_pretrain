#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

""" Pre-Training a 🤗 Wav2Vec2 model on unlabeled audio data """
import inspect
import time
import numpy as np
import torchaudio.sox_effects as sox_effects
import os

import argparse
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import datasets
import torch
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    AdamW,
    SchedulerType,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    get_scheduler,
    is_wandb_available,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="The dir of the dataset(including subset).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--saving_steps",
        type=int,
        default=500,
        help="Number of steps between each logging",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        help="Path to processor",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--max_gumbel_temperature",
        type=float,
        default=2.0,
        help="Maximum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--min_gumbel_temperature",
        type=float,
        default=0.5,
        help="Minimum temperature for gumbel softmax.",
    )
    parser.add_argument(
        "--gumbel_temperature_decay", type=float, default=0.999995, help="Decay of gumbel temperature during training."
    )
    parser.add_argument(
        "--pad_to_multiple_of",
        type=int,
        default=None,
        help="If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 for AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Epsilon for AdamW optimizer",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


@dataclass
class DataCollatorForWav2Vec2Pretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    model: Wav2Vec2ForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        if "speech" in features[0].keys():
            # print("proccess speech")
            for feature in features:
                feature_normalized = (feature["speech"] - np.mean(feature["speech"])) / np.sqrt(
                    np.var(feature["speech"]) + 1e-5)
                input_features.append({"input_values": feature_normalized})
                # input_features = [{"input_values": feature["input_values"]} for feature in features]
        elif "file" in features[0].keys():
            # 终版，问题出在input_values已经是normalize后的，而torchaudio处理原始信号比较精准，所以我们采用speech作为处理信号，而后再归一化
            for feature in features:
                # torchaudio需要二维数组(1,*)，输出也是二维数组，所以再调用时后输出后需要unsqueeze(0)和squeeze(0)
                # 法一
                # sr = torchaudio.backend.sox_io_backend.info(feature["file"]).sample_rate
                # start_frame = int(feature["seg_start"] * sr)
                # end_frame = int(feature["seg_end"] * sr)
                # waveform, _ = torchaudio.backend.sox_io_backend.load(
                # filepath=feature["file"],
                # num_frames=end_frame - start_frame,
                # frame_offset=start_frame
                # )
                # waveform = torchaudio.sox_effects.apply_effects_tensor(
                # waveform, sr,[['rate', str(16000)]])[0]

                # waveform = waveform[0]
                # feature_normalized = (waveform - torch.mean(waveform)) / torch.sqrt(torch.var(waveform) + 1e-5)
                # input_features.append({"input_values": feature_normalized})
                # 法二
                waveform = sox_effects.apply_effects_file(path=feature["file"], effects=[['rate', str(16000)]])[0]
                waveform = waveform[0]
                feature_normalized = (waveform - torch.mean(waveform)) / torch.sqrt(torch.var(waveform) + 1e-5)
                input_features.append({"input_values": feature_normalized})
        else:
            raise ValueError(f" ['file'] or ['speech'] is required for collect func")

        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # print(f"mask_indices_seq_length = {mask_indices_seq_length}")
        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )
        # print(f"batch['sub_attention_mask'].shape = {batch['sub_attention_mask'].shape}")

        features_shape = (batch_size, mask_indices_seq_length)
        # print(f"features_shape = {features_shape}")
        # sample randomly masked indices
        # torch.save(batch['sub_attention_mask'], "sb")

        # print(self.model.config.mask_time_prob)
        # print(self.model.config.mask_time_length)
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch['sub_attention_mask'],
        )
        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=batch["sub_attention_mask"],
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)
        # print(batch)
        return batch


def multiply_grads(params, c):
    """Multiplies grads by a constant *c*."""
    for p in params:
        if p.grad is not None:
            if torch.is_tensor(c):
                c = c.to(p.grad.device)
            p.grad.data.mul_(c)


def get_grad_norm(params, scale=1):
    """Compute grad norm given a gradient scale."""
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = (p.grad.detach().data / scale).norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    # logger.setLevel(logging.DEBUG)
    args = parse_args()
    print(f"args = {args}")
    stars = "*" * 20
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    logger.info(accelerator.state)
    # print(accelerator.state)
    print(accelerator.is_local_main_process)
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.INFO)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        # set up weights and biases if available
        if is_wandb_available():
            import wandb

            wandb.init(project=args.output_dir.split("/")[-1])
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub and not args.preprocessing_only:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset_dir = args.dataset_dir
    prepare_train_path = os.path.join(dataset_dir, 'train')
    prepare_dev_path = os.path.join(dataset_dir, 'dev')
    # prepare_train_path = "/data2_from_58175/huggingface/datasets/minnanyu/train"
    # prepare_dev_path = "/data2_from_58175/huggingface/datasets/minnanyu/test"
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    print(f"{stars}loading train_dataset{stars}")
    train_dataset = load_from_disk(prepare_train_path)

    ignored_columns = list(set(train_dataset.column_names) - set(["file","length"]))
    train_dataset = train_dataset.remove_columns(ignored_columns)
    print(train_dataset[0]["file"])
    print(len(train_dataset))

    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    val_dataset = load_from_disk(prepare_dev_path)
    val_dataset = val_dataset.remove_columns(ignored_columns).select(range(500))

    # train_dataset = train_dataset.filter(lambda batch: float(0.25)<=batch["length"]<=float(15.0))
    # train_dataset.save_to_disk("hf_datasets/train_filt")

    train_total_dur = sum(train_dataset["length"]) / 3600
    train_maxlength = sorted(train_dataset["length"], reverse=True)[0]

    val_total_dur = sum(val_dataset["length"]) / 3600
    val_maxlength = sorted(val_dataset["length"], reverse=True)[0]

    print(f"total_train_dataset:\n{train_dataset}\n")
    print(f"total duration of train_dataset:\n{train_total_dur} hours\n")
    print(f"maxlength of train_dataset:\n{train_maxlength} s\n")

    print(f"val_dataset:\n{val_dataset}\n")
    print(f"total duration of val_dataset:\n{val_total_dur} hours\n")
    print(f"maxlength of val_dataset:\n{val_maxlength} s\n")

    # 3. Load model
    config = Wav2Vec2Config.from_pretrained(args.model_name_or_path)

    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
        raise ValueError(
            "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and ``config.feat_extract_norm='layer'"
        )

    # initialize random model
    model = Wav2Vec2ForPreTraining(config)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.processor_path)

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 4. Define data collator, optimizer and scheduler
    data_collator = DataCollatorForWav2Vec2Pretraining(
        model=model, feature_extractor=processor, pad_to_multiple_of=args.pad_to_multiple_of
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        val_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 5. Train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    # print(f"completed_steps = {completed_steps}")
    # print(train_dataloader)
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # compute num of losses
            num_losses = batch["mask_time_indices"].sum()
            # print(f"num_losses = {num_losses}")
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses / sub_attention_mask.sum()

            # forward
            outputs = model(**batch)
            # print(f"outputs = {outputs}")
            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            # print("backward")
            # make sure that `num_losses` is summed for distributed training
            # and average gradients over losses of all devices
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)
            # print("multiply_grads")
            # update step
            # print(f"step = {step}")
            # print(f"args.gradient_accumulation_steps = {args.gradient_accumulation_steps}")
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                # print("computing grad norm")
                # compute grad norm for monitoring
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.parameters(), scale)
                # print("grad_norm")
                # update parameters
                optimizer.step()
                optimizer.zero_grad()
                # print("optimizer step")
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        "Gradients have overflown - skipping update step... " f"Updating gradient scale to {scale}..."
                    )

                # update gumbel temperature
                gumbel_temperature = max(
                    args.max_gumbel_temperature * args.gumbel_temperature_decay ** completed_steps,
                    args.min_gumbel_temperature,
                )
                if hasattr(model, "module"):
                    model.module.set_gumbel_temperature(gumbel_temperature)
                else:
                    model.set_gumbel_temperature(gumbel_temperature)

                progress_bar.update(1)
                completed_steps += 1
                # print("progress_bar.update")
            # 6. Log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()
                outputs.contrastive_loss.detach()
                outputs.diversity_loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather(loss).sum()
                    outputs.contrastive_loss = accelerator.gather(outputs.contrastive_loss).sum()
                    outputs.diversity_loss = accelerator.gather(outputs.diversity_loss).sum()
                    percent_masked = accelerator.gather(percent_masked).sum()

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps) / num_losses,
                    "constrast_loss": outputs.contrastive_loss / num_losses,
                    "div_loss": outputs.diversity_loss / num_losses,
                    "%_mask_idx": percent_masked / accelerator.num_processes,
                    "ppl": outputs.codevector_perplexity,
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                    "temp": torch.tensor(gumbel_temperature),
                    "grad_norm": torch.tensor(grad_norm),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    if is_wandb_available():
                        wandb.log(train_logs)

            # save model every `args.saving_steps` steps
            if (step + 1) % (args.gradient_accumulation_steps * args.saving_steps) == 0:
                if (args.push_to_hub and epoch < args.num_train_epochs - 1) or args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

                if (args.push_to_hub and epoch < args.num_train_epochs - 1) and accelerator.is_main_process:
                    repo.push_to_hub(commit_message=f"Training in progress step {completed_steps}", blocking=False)

            # if completed steps > `args.max_train_steps` stop
            if completed_steps >= args.max_train_steps:
                break

        # 7. Validate!
        model.eval()

        # init logs
        val_logs = {
            "val_loss": 0,
            "val_contrastive_loss": 0,
            "val_diversity_loss": 0,
            "val_num_losses": 0,
        }
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch.pop("sub_attention_mask", None)
                outputs = model(**batch)

            val_logs["val_loss"] += outputs.loss
            val_logs["val_contrastive_loss"] += outputs.contrastive_loss
            val_logs["val_diversity_loss"] += outputs.diversity_loss
            val_logs["val_num_losses"] += batch["mask_time_indices"].sum()

        # sum over devices in multi-processing
        if accelerator.num_processes > 1:
            val_logs = {k: accelerator.gather(v).sum() for k, v in val_logs.items()}

        val_logs = {k: v / val_logs["val_num_losses"] for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            log_str += "| {}: {:.3e}".format(k, v.item())

        if accelerator.is_local_main_process:
            progress_bar.write(log_str)
            if is_wandb_available():
                wandb.log(val_logs)

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                if args.push_to_hub:
                    repo.push_to_hub(commit_message="End of training")


if __name__ == "__main__":
    main()
