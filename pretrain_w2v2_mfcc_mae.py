#!/usr/bin/env python3
from CzcWav2vec2 import (
    Wav2Vec2ForPreTraining_mfcc,
    Wav2Vec2ForPreTraining_mfcc_auxiliary,
    get_czc_scheduler,
    get_w2v2_scheduler,
    Wav2Vec2ForPreTraining_mae,
    _compute_mask_indices_random
)
import math
import warnings
import wandb
import inspect
import time
import numpy as np
import torchaudio.sox_effects as sox_effects
import os
import datasets
from transformers.trainer_utils import (
    ShardedDDPOption,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    is_torch_tpu_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
)
from transformers.configuration_utils import PretrainedConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import collections
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.trainer_callback import (
    TrainerState,
)
from tqdm.auto import tqdm
from transformers.integrations import (  # isort: split
    hp_params,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    nested_detach,
)
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from packaging import version
from torch import nn

import librosa
from transformers import (
    __version__,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    is_apex_available,
    trainer_utils,
    is_wandb_available,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices


if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast
from transformers.optimization import get_scheduler

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    processor_path: str = field(
        metadata={"help": "Path to processor"}
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    max_gumbel_temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Maximum temperature for gumbel softmax."}
    )
    min_gumbel_temperature: Optional[float] = field(
        default=0.5, metadata={"help": "Minimum temperature for gumbel softmax."}
    )
    gumbel_temperature_decay: Optional[float] = field(
        default=0.999995, metadata={"help": "Decay of gumbel temperature during training."}
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_dir: str = field(
        default=None, metadata={"help": "The dir of the dataset(including subset)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0, metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"}
    )


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
    max_length: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
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

        batch = self.feature_extractor.pad(
            input_features,
            max_length=self.max_length,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]
        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])

        # make sure that no loss is computed on padded inputs
        if batch["attention_mask"] is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )
        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.model.config.mask_time_prob,
            self.model.config.mask_time_length,
            attention_mask=batch['sub_attention_mask'],
        )
        # mask_time_indices = _compute_mask_indices_random(
            # features_shape,
            # self.model.config.mask_time_prob,
            # attention_mask=batch['sub_attention_mask'],
        # )
        
        # sample negative indices
        # 据说从所用帧中采样负样本效果更好，原文中是从mask的帧中采样
        # sampled_negative_indices = _sample_negative_indices(
            # features_shape,
            # self.model.config.num_negatives,
            # mask_time_indices=batch["sub_attention_mask"],
        # )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        # batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)
        # print(batch)
        return batch


class Wav2Vec2PreTrainer(Trainer):
    """
    Subclassed :class:`~transformers.Trainer` for Wav2Vec2-like pretraining. Trainer can decay gumbel softmax temperature during training.
    """

    def __init__(self, *args, max_gumbel_temp=1, min_gumbel_temp=0, gumbel_temp_decay=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_update_step = 0
        self.max_gumbel_temp = max_gumbel_temp
        self.min_gumbel_temp = min_gumbel_temp
        self.gumbel_temp_decay = gumbel_temp_decay
        self.use_w2v2_lr_scheduler = False
        self.use_czc_lr_scheduler = False
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        # print(self.optimizer)
        if self.lr_scheduler is None:
            warmup_steps = (
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else math.ceil(num_training_steps * self.args.warmup_ratio)
            )
            if self.use_czc_lr_scheduler:
                print("use_czc_lr_scheduler")
                self.lr_scheduler = get_czc_scheduler(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                )
                
            elif self.use_w2v2_lr_scheduler:
                print("use_w2v2_lr_scheduler")
                self.lr_scheduler = get_w2v2_scheduler(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                )
            else:
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type,
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=num_training_steps,
                )
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    if self.use_amp:
                        with autocast():
                            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    else:
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    inputs.pop("sub_attention_mask", None)
                    if self.use_amp:
                        with autocast():
                            outputs = model(**inputs)
                    else:
                        outputs = model(**inputs)
                    ######################
                    loss = outputs.loss/inputs["mask_time_indices"].sum()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)
            if self.args.local_rank in [0, -1]:
                wandb.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            if self.args.local_rank in [0, -1]:
                wandb.log(metrics)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self.model = self.model.to(args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            num_train_epochs = int(args.num_train_epochs)
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            for step, inputs in enumerate(epoch_iterator):
                ##############################
                num_losses = inputs["mask_time_indices"].sum()
                # print(f"num_losses = {num_losses}")
                sub_attention_mask = inputs.pop("sub_attention_mask", None)
                sub_attention_mask = (
                    sub_attention_mask if sub_attention_mask is not None else torch.ones_like(
                        inputs["mask_time_indices"])
                )
                percent_masked = num_losses / sub_attention_mask.sum()

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_, outputs = self.training_step(model, inputs)
                        tr_loss += tr_loss_
                else:
                    tr_loss_, outputs = self.training_step(model, inputs)
                    tr_loss += tr_loss_
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    ###################
                    if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0 and args.local_rank in [-1, 0]:
                        outputs.loss.detach()
                        train_logs = {
                            "loss": outputs.loss / num_losses,
                            # "auxiliary": outputs.auxiliary_loss / num_losses,
                            # "div_loss": outputs.diversity_loss / num_losses,
                            "%_mask_idx": percent_masked,
                            # "ppl": outputs.codevector_perplexity,
                        }
                        wandb.log(train_logs)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            # We load the model state dict on the CPU to avoid an OOM error.
            state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME), map_location="cpu")
            # If the model is on the GPU, it still works!
            self._load_state_dict_in_model(state_dict)

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        global input_column_change
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += ["label", "label_ids"]
        columns = [k for k in self._signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))

        # 同一保留speech，可用于做变速，并重新做normalization，即为input_values
        ignored_columns.remove("speech") if "speech" in ignored_columns else None
        ignored_columns.remove("file") if "file" in ignored_columns else None
        ignored_columns.remove("text") if "text" in ignored_columns else None
        ignored_columns.remove("length") if "length" in ignored_columns else None
        # ignored_columns.remove("seg_end") if "seg_end" in ignored_columns else None
        # ignored_columns.remove("seg_start") if "seg_start" in ignored_columns else None

        # if input_column_change:
        # ignored_columns.remove("speech")
        # ignored_columns.append("input_values")
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # print(outputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        # else:
        #     loss = self.compute_loss(model, inputs)
        else:
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1 or self.deepspeed or model is not self.model:
            if model.module.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.module.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["mask_time_indices"]).sum()
                # loss = loss.sum() / inputs["input_values"].shape[0]
            else:
                raise ValueError(f"{model.module.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
        else:
            if model.config.ctc_loss_reduction == "mean":
                loss = loss.mean()
            elif model.config.ctc_loss_reduction == "sum":
                loss = loss.sum() / (inputs["mask_time_indices"]).sum()
                # loss = loss.sum() / inputs["input_values"].shape[0]
            else:
                raise ValueError(f"{model.config.ctc_loss_reduction} is not valid. Choose one of ['mean', 'sum']")
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        self.num_update_step += 1
        # make sure gumbel softmax temperature is decayed
        # if self.args.n_gpu > 1 or self.deepspeed or model is not self.model:
        #     model.module.set_gumbel_temperature(
        #         max(self.max_gumbel_temp * self.gumbel_temp_decay ** self.num_update_step, self.min_gumbel_temp)
        #     )
        # else:
        #     model.set_gumbel_temperature(
        #         max(self.max_gumbel_temp * self.gumbel_temp_decay ** self.num_update_step, self.min_gumbel_temp)
        #     )

        return loss.detach(), outputs


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    stars = "*" * 20
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)
    # logger.info(f"{stars}model_args:{stars}\n{model_args}")
    # logger.info(f"{stars}data_args:{stars}\n{data_args}")
    # logger.info(f"{stars}training_args:{stars}\n{training_args}")

    # Downloading and loading a dataset from the hub.
    dataset_dir = data_args.dataset_dir
    prepare_train_path = os.path.join(dataset_dir, 'train')
    prepare_dev_path = os.path.join(dataset_dir, 'dev')
    # prepare_train_path = "/data2_from_58175/huggingface/datasets/minnanyu/train"
    # prepare_dev_path = "/data2_from_58175/huggingface/datasets/minnanyu/test"
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"{stars}loading train_dataset{stars}")
    train_dataset = load_from_disk(prepare_train_path)

    ignored_columns = list(set(train_dataset.column_names) - set(["file","length"]))
    train_dataset = train_dataset.remove_columns(ignored_columns)
    logger.info(train_dataset[0]["file"])

    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    val_dataset = load_from_disk(prepare_dev_path)
    val_dataset = val_dataset.remove_columns(ignored_columns).select(range(500))

    # train_dataset = train_dataset.filter(lambda batch: float(0.25)<=batch["length"]<=float(15.0))
    # train_dataset.save_to_disk("hf_datasets/train_filt")

    train_total_dur = sum(train_dataset["length"]) / 3600
    train_maxlength = sorted(train_dataset["length"], reverse=True)[0]

    val_total_dur = sum(val_dataset["length"]) / 3600
    val_maxlength = sorted(val_dataset["length"], reverse=True)[0]

    logger.info(f"total_train_dataset:\n{train_dataset}\n")
    logger.info(f"total duration of train_dataset:\n{train_total_dur} hours\n")
    logger.info(f"maxlength of train_dataset:\n{train_maxlength} s\n")

    logger.info(f"val_dataset:\n{val_dataset}\n")
    logger.info(f"total duration of val_dataset:\n{val_total_dur} hours\n")
    logger.info(f"maxlength of val_dataset:\n{val_maxlength} s\n")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.processor_path)
    # pretraining is only supported for "newer" stable layer norm architecture
    # apply_spec_augment has to be True, mask_feature_prob has to be 0.0
    config = Wav2Vec2Config.from_pretrained(
        model_args.model_name_or_path,
        gradient_checkpointing=False,
    )
    # config.num_hidden_layers = 1
    # config.mask_time_prob = 0.65

    # if not config.do_stable_layer_norm or config.feat_extract_norm != "layer":
    #     raise ValueError(
    #         "PreTraining is only supported for ``config.do_stable_layer_norm=True`` and ``config.feat_extract_norm='layer'"
    #     )

    model = Wav2Vec2ForPreTraining_mae(config)

    if is_wandb_available() and training_args.local_rank in [0, -1]:
        # 只跟踪主进程，DDP下主进程local_rank为0，正常情况为-1
        import wandb
        config = {"command":sys.argv,**vars(model_args), **vars(data_args), **vars(training_args), **vars(model.config)}
        logger.info(config)
        wandb.init(project=training_args.logging_dir.split("/")[-1])
        wandb.config.update(config)
        wandb.watch(model, log="all")

    data_collator = DataCollatorForWav2Vec2Pretraining(model=model, feature_extractor=processor)

    trainer = Wav2Vec2PreTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        max_gumbel_temp=model_args.max_gumbel_temperature,
        min_gumbel_temp=model_args.min_gumbel_temperature,
        gumbel_temp_decay=model_args.gumbel_temperature_decay,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
