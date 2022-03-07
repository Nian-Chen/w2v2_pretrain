#!/bin/bash
stage=0
stop_stage=2
. kaldi_utils/parse_options.sh || exit 1;
stage=$stage
stop_stage=$stop_stage

# 默认使用所有GPU，也可自定义cuda_visible_devices,n_gpus
n_gpus=`nvidia-smi -L|wc -l`
# 转数组
gpu_list=(`seq $n_gpus`)
# [1 2 3 4] --> [0 1 2 3] --> 0,1,2,3
for i in `seq $n_gpus`
do
index=$[i-1]
gpu_list[index]=$[${gpu_list[index]}-1]
# echo ${gpu_list[index]}
done
cuda_visible_devices=`echo ${gpu_list[*]}|awk '{gsub(" ",",");print $0}'`
# 0,1,2,3

# ！！！！！！！！！！！若自定义只使用前两块gpu，可解除如下注释！！！！！！！！！！！
#cuda_visible_devices=0,1
#n_gpus=2
echo "n_gpus = $n_gpus"
echo "cuda_visible_devices = $cuda_visible_devices"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Data preparation
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 0: make train_dataset"
  sh segments2dataset.sh --data_dir data_seg_train || exit 1;
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 0: make dev_dataset"
  sh segments2dataset.sh --data_dir data_seg_dev || exit 1;
fi
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # trim audio
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 1: Trim train_audio"
  sh dataset2trim.sh \
  --dataset_path data_seg_train/tmp/dataset \
  --trim_audio_outputdir ws_trim_audio || exit 1;
  sh dataset2trim.sh \
  --dataset_path data_seg_dev/tmp/dataset \
  --trim_audio_outputdir ws_trim_audio || exit 1;
  echo "log file in data_seg_train/tmp/log.*.txt"
  kaldi_utils/run.pl JOB=1:10 data_seg_train/tmp/log.JOB.txt \
  sh run_trim_ffmpeg.sh \
  --file_name data_seg_train/tmp/run_trim \
  --index JOB || exit 1;
  echo "stage 1: Trim dev_audio"
  echo "log file in data_seg_dev/tmp/log.*.txt"
  kaldi_utils/run.pl JOB=1:10 data_seg_dev/tmp/log.JOB.txt \
  sh run_trim_ffmpeg.sh \
  --file_name data_seg_dev/tmp/run_trim \
  --index JOB || exit 1;
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # make hf_datasets
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 2: Make hf_datasets"
  python make_hfdatasets.py  \
  --input_train_file=data_seg_train/tmp/dataset  \
  --input_dev_file=data_seg_dev/tmp/dataset  \
  --save_path=./hf_datasets \
  --segments_mode \
  --trim_audio_dir=ws_trim_audio \
  --min_length=0.25 \
  --max_length=15.0 \
  --is_ch || exit 1;
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # fine-tune w2v2-mtl
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 3: Fine-tune w2v2-mtl"
  # 原实验环境是4台v100
  # 若显存不足，请减小per_device_train_batch_size，同时增大gradient_accumulation_steps，使二者乘积为64
  CUDA_VISIBLE_DEVICES=$cuda_visible_devices python -m torch.distributed.launch  \
  --nproc_per_node $n_gpus finetune_w2v2.py  \
  --output_dir=training_output  \
  --report_to=tensorboard  \
  --logging_dir=tensorboard/ws_train_s  \
  --num_train_epochs=15  \
  --logging_strategy=steps  \
  --logging_steps=10  \
  --logging_first_step  \
  --per_device_train_batch_size=4  \
  --per_device_eval_batch_size=1  \
  --gradient_accumulation_steps=4  \
  --save_total_limit=5  \
  --evaluation_strategy=steps  \
  --eval_steps=2000  \
  --save_strategy=steps  \
  --save_steps=2000  \
  --learning_rate=2e-4  \
  --lr_scheduler_type=czc  \
  --warmup_ratio=0.1  \
  --group_by_length  \
  --fp16=True  \
  --encoder_decoder_mode=True \
  --encoder_or_w2v2model_path=encoder  \
  --decoder_path=decoder  \
  --dataset_dir=hf_datasets \
  --processor_path=processor \
  --preprocessing_num_workers=8  \
  --dataloader_num_workers=8  \
  --freeze_feature_extractor=True  \
  --freeze_ALN=False  \
  --freeze_all_except_lm=False  \
  --speed_perturb=True  \
  --prediction_loss_only=False  \
  --verbose_log=False || exit 1;
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 4: Decode dev set"
  echo "log file in decode_mp_dev/log/log.*.txt"
  # 解码
  # 设置解码使用的模型，默认是提交竞赛成绩的对应模型
  decode_model_path=10epochs-result-ctcgreedy-31.58
  # 或者选择上一步训练输出的模型，如下注释
  #decode_model_path=training_output/checkpoint-22000

  # decode dev
  # 分8进程解码，将子集存储在decode_mp_dev下
  kaldi_utils/run.pl JOB=1:8 decode_mp_dev/log/log.JOB.txt \
  python decode_mp_ws.py \
  --model_name_or_path=$decode_model_path \
  --datasets_name_or_path=hf_datasets/dev \
  --output_dir=decode_mp_dev \
  --total_jobs=8 \
  --job_index=JOB || exit 1;
  # 拼接子集并计算两种cer结果：beam_search和beam_search+att_rescore
  # 解码结果存在result_dev.txt中
  python concat_decoded_subset_to_finaltxt.py \
  --subset_dir=decode_mp_dev \
  --output_file=result_dev.txt || exit 1;
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "stage 4: Decode test set"
  # decode test
  # 由于测试集没有提供kaldi常规文件，故需要单独制作hf_datasets
  # ！！！！！！！！！！！请将音频放入B-Test路径下！！！！！！！！！！！
  python make_test_wavscp_hfdatasets.py || exit 1;
  # 分8进程解码，将子集存储在decode_mp_test下
  echo "log file in decode_mp_test/log/log.*.txt"
#  kaldi_utils/run.pl JOB=1:8 decode_mp_test/log/log.JOB.txt \
#  python decode_mp_ws.py \
#  --model_name_or_path=$decode_model_path \
#  --datasets_name_or_path=hf_datasets/test \
#  --output_dir=decode_mp_test \
#  --total_jobs=8 \
#  --job_index=JOB || exit 1;
  # 无文本信息，故无法计算cer
  # 解码结果存在result_dev.txt中
  python concat_decoded_subset_to_finaltxt.py \
  --subset_dir=decode_mp_test \
  --output_file=result_test.txt || exit 1;
fi