#!/usr/bin/env python3
import kenlm
from pdb import set_trace
from collections import defaultdict
from beam_search_att_rescore import (
    ctc_prefix_beam_search,
    attention_rescoring,
    attention_rescoring_lm,
    get_kenlm_decoder,
    attention_rescoring_lm,
)
from jiwer import wer
from CzcWav2vec2 import Wav2vec2_Gpt2
from pdb import set_trace
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
    AutoTokenizer,
    AutoModelForCausalLM,
    PretrainedConfig,
    BertConfig,
    EncoderDecoderConfig,
    GPT2Config,
    BertGenerationConfig
)
from transformers.modeling_outputs import Seq2SeqLMOutput
import time

import logging

import sys
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torchaudio.sox_effects as sox_effects

import numpy as np
import torch

from datasets import load_dataset, load_from_disk
import os
os.environ["HF_HOME"] = "huggingface-home"
os.environ["TRANSFORMERS_CACHE"] = "huggingface-home/transformers"
os.environ["HF_MODELS_CACHE"] = "huggingface-home/datasets"
os.environ["HF_METRICS_CACHE"] = "huggingface-home/metrics"
os.environ["HF_DATASETS_CACHE"] = "huggingface-home/datasets"
logger = logging.getLogger(__name__)
def configure_logger(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if trainer_utils.is_main_process(training_args.local_rank):
        # 设置info只能在主进程显示，避免某些信息重复显示(多进程)
        logging_level = logging.INFO
    logger.setLevel(logging_level)
@dataclass
class Arguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    datasets_name_or_path: str = field(
        metadata={"help": "Path to the datasets"}
    )
    total_jobs: Optional[int] = field(
        default=8,
        metadata={"help": "total devices for decoding, ie. num jobs"},
    )
    job_index: Optional[int] = field(
        default=1,
        metadata={"help": "total devices for decoding, ie. num jobs"},
    )
# time CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch  --nproc_per_node 4 pseudo_decode_mp.py --model_name_or_path=/data2_from_58175/wav2vec2_output/filteredbyctc_continue/task1-50-rescore-checkpoint-2454-0.08439-0.08407-0.07993 --datasets_name_or_path=/home/data/fisher_swbd_nodup_onlyspeech/swbd_pseudo_89h --output_dir=/home/data/pseudo_decode_mp --total_devices=4
def show_args(args):
    print('\n'.join(['%s:%s' % item for item in args.__dict__.items()]))
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((Arguments, TrainingArguments))
    stars = "*"*20
    args, training_args = parser.parse_args_into_dataclasses()
    # info信息只在主进程中才显示，warning则是所有进程可用
    configure_logger(training_args)
    # parser = HfArgumentParser(Arguments)
    # parser.parse_args_into_dataclasses()输出默认是元组
    # args = parser.parse_args_into_dataclasses()
    total_jobs = args.total_jobs
    total_cudas = torch.cuda.device_count()
    logger.info(f"total_cudas = {total_cudas}")
    # kaldi/utils JOB索引只从1开始，故减1，从0开始
    job_index = args.job_index - 1
    cuda_index = job_index%total_cudas
    
    # manager = Manager()
    # my_list = manager.list()
    # my_list.append(cuda_index)
    # logger.warning(f"my_list = {my_list}")
    # cuda_index 若单卡则是-1，否则0,1,2,...N
    logger.info(f"args = {args}")
    model_path = args.model_name_or_path+"/pytorch_model.bin"
    device = "cuda:"+str(cuda_index)
    # device = "cpu"
    logger.warning(f"device = {device}")
    datasets_name_or_path = args.datasets_name_or_path
    # encoder_model_path = "/data2_from_58175/wav2vec2_output/filteredbyctc_continue/checkpoint-3577"
    # decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder"
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"{stars}loading mdoel{stars}")
    encoder_model_path = "encoder"
    decoder_model_path = "decoder"
    encoder = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
    decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
    model = Wav2vec2_Gpt2(encoder=encoder,decoder=decoder)
    state_dict = torch.load(model_path,map_location="cpu")
    model.load_state_dict(state_dict)
    
        
    # encoder_state_dict = torch.load("/data2_from_58175/wav2vec2_output/fisher_joint/ws-checkpoint-9000-31.58/pytorch_model.bin",map_location="cpu")
    # model.encoder.load_state_dict(encoder_state_dict)
    
    model.to(device)
    model.eval()
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    logger.info(f"{stars}loading dataset{stars}")

    dataset = load_from_disk(datasets_name_or_path)#.remove_columns(['sampling_rate', 'seg_end', 'seg_start'])
    print(dataset)
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    processor = Wav2Vec2Processor.from_pretrained("processor")

    total_examples = len(dataset)
    print(total_examples)
    # total_examples = 200
    # 尽量等分，0到total_examples中total_devices等分，则有total_devices+1个点
    index_list = np.linspace(0,total_examples,total_jobs+1,dtype=int).tolist()
    # print(index_list)
    start_index = index_list[job_index]
    end_index = index_list[job_index+1]
    logger.warning(f"start_index,end_index = {(start_index,end_index)}")
    
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    # vocabulary = [x[1].replace("|", " ") if x[1] not in processor.tokenizer.all_special_tokens else "_" for x in sort_vocab]
    vocabulary = [x[1] for x in sort_vocab]
    # 不使用lm时必须将lm_path设为None，仅靠alpha=0.0是不行的。
    lm_path = None
    alpha = 0.00
    # lm_model = kenlm.Model(kenlm_model_path)
    # logger.info(f"{stars}loading gpt_model{stars}")
    # # gpt_path = "/data2_from_58175/wav2vec2_output/gpt/-checkpoint-2148"
    # # gpt_path = "/data2_from_58175/huggingface/models/gpt2-medium"
    # gpt_path = "/data2_from_58175/wav2vec2_output/gpt/vocab3852-minnanyu-checkpoint-5264"
    # gpt_model = AutoModelForCausalLM.from_pretrained(gpt_path)
    # # gpt_model = model.decoder
    # gpt_model.to(device)
    # logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    # gpt_model.eval()
    # # gpt_tokenizer = AutoTokenizer.from_pretrained("/data2_from_58175/huggingface/models/gpt2-medium")
    # gpt_decoder = gpt_model
    
    # tokenizer = gpt_tokenizer
    # gpt_decoder = model.decoder
    tokenizer = processor.tokenizer
    # gpt_decoder = None
    kenlm_decoder = get_kenlm_decoder(
                vocabulary=vocabulary,
                lm_path=lm_path,
                alpha=alpha,
                beta=0.0,
                rescoring_kenlm_model_path=None,
                gpt_decoder=None,
                tokenizer=tokenizer,
                )
    def map_to_pred_wenet_att_rescore(batch):
        if "speech" in batch.keys():
            speech = batch["speech"]
        elif "file" in batch.keys():
            speech = sox_effects.apply_effects_file(path=batch["file"][0],effects=[['rate', str(16000)]])[0][0]
        features = processor(speech, return_tensors="pt", padding="longest",sampling_rate=16000)
        input_values = features.input_values.to(device)
        attention_mask = features.attention_mask.to(device)

        if "text" not in batch.keys() and "labels" in batch.keys():
            batch["text"] = processor.batch_decode(batch["labels"],group_tokens=False)

        (best_score_ctc,best_score_att,best_score),predicted_ids,predicted_ids_bs = attention_rescoring_lm(input_values=input_values,
                                                             processor=processor,
                                                             beam_size=200,
                                                             attention_mask=attention_mask,
                                                             model=model,
                                                             att_weight=1.5,
                                                             output_prefix_beam_search=True,
                                                             kenlm_decoder=kenlm_decoder
                                                             )
        batch["transcription"] = processor.batch_decode(predicted_ids,group_tokens=False)
        batch["transcription_bs"] = processor.batch_decode(predicted_ids_bs,group_tokens=False)
        # logger.info(batch["transcription"])
    #     logger.info(batch["transcription_bs"])
        return batch
    result = dataset.select(range(start_index,end_index)).map(map_to_pred_wenet_att_rescore,batched=True, batch_size=1,keep_in_memory=True)
    logger.warning(f"{stars} over{stars}")
    logger.warning(f"start_index,end_index = {(start_index,end_index)}")

    subset_save_path = os.path.join(training_args.output_dir,str(job_index))
    logger.warning(f"subset is saved to path : {subset_save_path}")
    
    ignored_columns = list(set(result.column_names) - set(["id","text","transcription","transcription_bs","vote_result"]))
    result = result.remove_columns(ignored_columns)
    logger.warning(f"finished subset :\n {result}")
    
    result.save_to_disk(subset_save_path)
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S'))
    
    info_idct = defaultdict(str)
    info_idct["model_name_or_path"] = args.model_name_or_path
    info_idct["datasets_name_or_path"] = args.datasets_name_or_path
    info_idct["total_jobs"] = args.total_jobs
    info_json = os.path.join(training_args.output_dir,"info.json")
    with open(info_json,"w",encoding="utf-8") as f:
        json.dump(info_idct,f,indent=2)
    # wer_rescore = wer(truth=result["text"],hypothesis=result["transcription"])
    # wer_bs = wer(truth=result["text"],hypothesis=result["transcription_bs"])
    # logger.info(f"model_path = {model_path}")
    # logger.info(f"datasets_name_or_path = {datasets_name_or_path}")
    # logger.info(f"wer_bs = {wer_bs}")
    # logger.info(f"wer_rescore = {wer_rescore}")

# jupyter中拼接
# subset_path_list = glob.glob("xxx/*[0-9]")
# subset_path_list = sorted(subset_path_list,key=lambda x:int(x.split("/")[-1]))
# subset_list = []
# for i in subset_path_list:
    # subset = load_from_disk(i)
    # subset_list.append(subset)
# subset_list  

# /tsdata/kaldi_utils/run.pl JOB=1:10 /home/data/decode_mp/log/log.JOB.txt python decode_mp.py --model_name_or_path=/data2_from_58175/wav2vec2_output/filteredbyctc_continue/ft10h-ctcbs-0.08016-checkpoint-7320 --datasets_name_or_path=/home/data/fisher_swbd_nodup_onlyspeech/swbdtest5h --output_dir=/home/data/decode_mp --total_jobs=30 --job_index=JOB
if __name__ == "__main__":
    main()