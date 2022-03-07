#!/bin/bash
time CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch  \
--nproc_per_node 1 pretrain_w2v2.py  \
--output_dir=training_output  \
--report_to=tensorboard  \
--logging_dir=tensorboard/ws_train_s  \
--num_train_epochs=15  \
--logging_strategy=steps  \
--logging_steps=10  \
--logging_first_step  \
--per_device_train_batch_size=4  \
--per_device_eval_batch_size=1  \
--gradient_accumulation_steps=16  \
--save_total_limit=5  \
--evaluation_strategy=steps  \
--eval_steps=2000 \
--save_strategy=steps  \
--save_steps=2000  \
--learning_rate=2e-5  \
--lr_scheduler_type=linear  \
--warmup_ratio=0.1  \
--group_by_length  \
--fp16=True  \
--dataset_dir=hf_datasets \
--model_name_or_path=/data2_from_58175/huggingface/models/wav2vec2-base/config.json \
--processor_path=processor \
--preprocessing_num_workers=8  \
--dataloader_num_workers=8  \
--freeze_feature_extractor=False  \
--prediction_loss_only=False  \
--verbose_log=False