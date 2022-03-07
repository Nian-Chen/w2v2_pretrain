#!/usr/bin/env python3
from transformers import PreTrainedTokenizerBase
import copy
import kenlm
from pyctcdecode import BeamSearchDecoderCTC
from pyctcdecode import build_ctcdecoder
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from datasets import DatasetDict, load_dataset, load_from_disk
import torch
import math
from typing import Optional, Tuple, List
from torch import nn
from pdb import set_trace
from CzcWav2vec2 import Wav2vec2_Gpt2
from transformers import (
    AutoModelForCausalLM,
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    GPT2LMHeadModel
)
def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp
def ctc_prefix_beam_search(
    logits: torch.Tensor,
    beam_size: int,
) -> Tuple[List[List[int]], torch.Tensor]:
    """
    总共就32个token，意味着topk最大为32
    """
    topk = beam_size
    batch_size = logits.shape[0]
    # For CTC prefix beam search, we only support batch_size=1
    assert batch_size == 1
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder forward and get CTC score
#     encoder_out, encoder_mask = self._forward_encoder(
#         speech, speech_lengths, decoding_chunk_size,
#         num_decoding_left_chunks,
#         simulate_streaming)  # (B, maxlen, encoder_dim)
#     maxlen = encoder_out.size(1)
    ctc_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
#     ctc_probs = self.ctc.log_softmax(
#         encoder_out)  # (1, maxlen, vocab_size)
    maxlen = ctc_probs.size(1)
    ctc_probs = ctc_probs.squeeze(0)
    # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        # set_trace()
        logp = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 初始
        # (Pdb) next_hyps.values()
        # dict_values([])
        # (Pdb) next_hyps['a']
        # (-inf, -inf)
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = logp.topk(topk)  # (beam_size,)
        # 假设topk为[0,22]
        for s in top_k_index:
            s = s.item()
            ps = logp[s].item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == 0:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    # 初始n_pb, n_pnb = -inf,-inf，而pb为0
                    # 则n_pb有值, -0.001549235312268138
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # 初始
                    # (Pdb) prefix
                    # () 仍为空
                    # (Pdb) next_hyps[prefix]
                    # (-0.001549235312268138, -inf)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s, )
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    # s=22
                    n_prefix = prefix + (s, )
                    # (Pdb) n_prefix
                    # (22,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    # (Pdb) n_pnb
                    # -7.889496326446533
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                    # (Pdb) next_hyps[n_prefix]
                    # (-inf, -7.889496326446533)
         # 第一帧遍历结束后
         # (Pdb) next_hyps
         # [((), (-0.001549235312268138, -inf)), ((22,), (-inf, -7.889496326446533)), ((6,), (-inf, -8.204001426696777)), ((10,), (-inf, -8.410624504089355)), ((12,), (-inf, -8.903583526611328)), ((18,), (-inf, -8.909826278686523)), ((24,), (-inf, -9.037549018859863)), ((14,), (-inf, -10.316286087036133)), ((13,), (-inf, -10.435552597045898)), ((8,), (-inf, -10.616910934448242)), ((7,), (-inf, -10.690228462219238)), ((17,), (-inf, -10.705060958862305)), ((16,), (-inf, -10.721846580505371)), ((15,), (-inf, -10.786240577697754)), ((21,), (-inf, -10.964200973510742)), ((23,), (-inf, -11.075313568115234)), ((5,), (-inf, -11.0880765914917)), ((20,), (-inf, -11.09679126739502)), ((26,), (-inf, -11.57800006866455)), ((19,), (-inf, -11.611882209777832)), ((9,), (-inf, -11.679497718811035)), ((29,), (-inf, -11.756319999694824)), ((27,), (-inf, -12.142388343811035)), ((11,), (-inf, -12.519916534423828)), ((25,), (-inf, -12.655060768127441)), ((28,), (-inf, -13.603233337402344)), ((4,), (-inf, -14.2069091796875)), ((30,), (-inf, -14.583219528198242)), ((31,), (-inf, -15.640634536743164)), ((3,), (-inf, -23.76201629638672)), ((1,), (-inf, -23.857267379760742)), ((2,), (-inf, -23.88033103942871))]
         # (Pdb) len(next_hyps) = 32
         # next_hyps中有一个prefix为空，即第一帧为blank/pad
        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                           key=lambda x: log_add(list(x[1])),
                           reverse=True)
        cur_hyps = next_hyps[:beam_size]
        # 虽然第一帧结束后总的路径只有topk=32个，但是取:beam_size/:64并不会报错
        # 往后每一次得到的next_hyps是包含截至当前帧的所有可能路径，但受限于beam_size
        # len(next_hyps)=32*64=2048经合并前缀后为1958，再取前64作为cur_hyps，进入下一帧搜索
        # 假设默认topk = beam_size = 20，则
        # len(next_hyps)=20*20=400经合并前缀后为385，再取前20作为cur_hyps，进入下一帧搜索
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    # 想取best的predicted_ids，需要hyps[0][0]
    # print(f"hyps of ctc_prefix_beam_search = {hyps}")
    return hyps
    
def get_kenlm_decoder(
        vocabulary: List[str],
        lm_path: Optional[str] = None,
        alpha: float = 0.3,
        beta: float = 0.0,
        rescoring_kenlm_model_path: Optional[str] = None,
        gpt_decoder: Optional[torch.nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
    # https://github.com/kensho-technologies/pyctcdecode.git
    kenlm_decoder = build_ctcdecoder(
        labels=vocabulary,
        kenlm_model_path=lm_path,
        alpha=alpha,
        beta=beta,
        rescoring_kenlm_model_path=rescoring_kenlm_model_path,
        gpt_decoder=gpt_decoder,
        tokenizer=tokenizer,
    )
    return kenlm_decoder

def ctc_prefix_beam_search_lm(
    logits: torch.Tensor,
    kenlm_decoder: BeamSearchDecoderCTC,
    beam_size: int,
    processor: Wav2Vec2Processor,
    encoder_hidden_states: Optional[torch.tensor] = None,
    encoder_attention_mask: Optional[torch.tensor] = None,
) -> Tuple[List[List[int]], torch.Tensor]:
    beam_results = kenlm_decoder.decode(logits[0].cpu().numpy(),beam_width=beam_size,encoder_hidden_states=encoder_hidden_states,encoder_attention_mask=encoder_attention_mask)
    # 借助外部包的prefix beam search结果与上边结果严格对齐，转为hyps
    # print(f"len(beam_results) = {len(beam_results)}")
    hyps = []
    for i in range(len(beam_results)):
        hyps_i = tuple(processor.tokenizer(beam_results[i][0]).input_ids)
        hyps_i_score = beam_results[i][-1]
        # beam_results[i][-1]是融合语言模型的分数，[-2]是ctc beam search分数
        hyps.append((hyps_i,hyps_i_score))
    # print(f"hyps of ctc_prefix_beam_search_lm = {hyps}")
    return hyps
def attention_rescoring(
        input_values: torch.Tensor,
        attention_mask,
        model:torch.nn.Module,
        beam_size: int,
        ctc_weight: float = 0.5,
        sos_id: int = 1,
        eos_id: int = 2,
        ignore_id: int = -100,
        output_prefix_beam_search: bool = False,
) -> List[int]:
    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask, encoder_logits = model(input_values=input_values,attention_mask=attention_mask,forward_only_encoder=True)
    hyps = ctc_prefix_beam_search(logits=encoder_logits,beam_size=beam_size)
    hyps_pad = pad_sequence([
        torch.tensor(hyp[0], device=model.device, dtype=torch.long)
        for hyp in hyps
    ], True, -100)
    # 将beam_search的结果用ignore_id=-100进行pad，做成batch
    # 制作decoder的输入id，包括首尾添加sos_id、eos_id以及-100用0取代
    dec_labels = model.add_sos_eos(hyps_pad, sos_id, eos_id, ignore_id)
    dec_input_attention_mask = dec_labels.ne(-100)
    dec_input_ids = dec_labels.masked_fill(~dec_input_attention_mask, 0)
    # dec_labels
    # dec_input_attention_mask
#     dec_input_ids[:,20]
    # decode_out (beam_size, max_hyps_len, vocab_size)
    with torch.no_grad():
        decoder_out = model.decoder(input_ids=dec_input_ids,encoder_hidden_states=encoder_hidden_states,attention_mask=dec_input_attention_mask,encoder_attention_mask=encoder_attention_mask).logits.detach()
    decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
    decoder_out = decoder_out.cpu().numpy()
    best_score = -float('inf')
    best_score_ctc = -float('inf')
    best_score_att = -float('inf')
    best_index = 0
    for i, hyp in enumerate(hyps):
        # 第i个beam，hyp[0]为id序列，hyp[1]为其ctc_score
        # 第i个beam的att_score
        ctc_score = hyp[1]
        att_score = 0.0
        for j, w in enumerate(hyp[0]):
            #
            att_score += decoder_out[i][j][w]
        att_score += decoder_out[i][len(hyp[0])][eos_id]
        # 不是各0.5
        score = ctc_score * ctc_weight + att_score * (1-ctc_weight)
        print(f"ctc_score = {ctc_score}")
        print(f"att_score = {att_score}")
        if score > best_score:
            best_score = score
            best_score_ctc = ctc_score
            best_score_att = att_score
            best_index = i
    # print(f"best_index = {best_index}")
    # print(f"best_hyps = {[list(hyps[best_index][0])]}")
    if output_prefix_beam_search == True:
        return (best_score_ctc,best_score_att,best_score),[list(hyps[best_index][0])],[list(hyps[0][0])]
    return (best_score_ctc,best_score_att,best_score),[list(hyps[best_index][0])]


def attention_rescoring_lm(
        processor: Wav2Vec2Processor,
        beam_size: int,
        kenlm_decoder: BeamSearchDecoderCTC,
        input_values: torch.Tensor,
        attention_mask,
        model:torch.nn.Module,
        att_weight: float = 1.0,
        sos_id: int = 1,
        eos_id: int = 2,
        ignore_id: int = -100,
        output_prefix_beam_search: bool = False,
) -> List[int]:
    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask, encoder_logits = model(input_values=input_values,attention_mask=attention_mask,forward_only_encoder=True)
    hyps = ctc_prefix_beam_search_lm(
                        logits=encoder_logits,
                        kenlm_decoder=kenlm_decoder,
                        beam_size=beam_size,
                        processor=processor,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        )
    # hyps = ctc_prefix_beam_search(logits=encoder_logits,beam_size=20)
    hyps_pad = pad_sequence([
        torch.tensor(hyp[0], device=model.device, dtype=torch.long)
        for hyp in hyps
    ], True, -100)
    # 将beam_search的结果用ignore_id=-100进行pad，做成batch
    # 制作decoder的输入id，包括首尾添加sos_id、eos_id以及-100用0取代
    dec_labels = model.add_sos_eos(hyps_pad, sos_id, eos_id, ignore_id)
    dec_input_attention_mask = dec_labels.ne(-100)
    dec_input_ids = dec_labels.masked_fill(~dec_input_attention_mask, 0)
    # dec_labels
    # dec_input_attention_mask
#     dec_input_ids[:,20]
    # decode_out (beam_size, max_hyps_len, vocab_size)
    with torch.no_grad():
        decoder_out = model.decoder(input_ids=dec_input_ids,encoder_hidden_states=encoder_hidden_states,attention_mask=dec_input_attention_mask,encoder_attention_mask=encoder_attention_mask).logits.detach()
    decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
    decoder_out = decoder_out.cpu().numpy()
    best_score = -float('inf')
    best_score_ctc = -float('inf')
    best_score_att = -float('inf')
    best_index = 0
    for i, hyp in enumerate(hyps):
        # 第i个beam，hyp[0]为id序列，hyp[1]为其ctc_score
        # 第i个beam的att_score
        ctc_score = hyp[1]# if hyp[1] < 0 else -1
        att_score = 0.0
        for j, w in enumerate(hyp[0]):
            #
            att_score += decoder_out[i][j][w]
        att_score += decoder_out[i][len(hyp[0])][eos_id]
        # 不是各0.5
        att_score *= att_weight
        print(f"ctc_score = {ctc_score}")
        print(f"att_score = {att_score}")
        score = ctc_score + att_score
        # score = ctc_score * ctc_weight + att_score * (1-ctc_weight)
#         print(score)
        if score > best_score:
            best_score = score
            best_score_ctc = ctc_score
            best_score_att = att_score
            best_index = i
    # print(f"best_index_lm = {best_index}")
    # print(f"best_hyps_lm = {[list(hyps[best_index][0])]}")
    if output_prefix_beam_search == True:
        return (best_score_ctc,best_score_att,best_score),[list(hyps[best_index][0])],[list(hyps[0][0])]
    return (best_score_ctc,best_score_att,best_score),[list(hyps[best_index][0])]


def main():
    # predicted_ids = [list(ctc_prefix_beam_search(torch.rand(1,200,32),5)[0][0])]
    # print(predicted_ids)
    encoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/encoder"
    decoder_model_path = "/data2_from_58175/huggingface/models/wav2vec2_gpt2/decoder"
    encoder = Wav2Vec2ForCTC.from_pretrained(encoder_model_path)
    decoder = AutoModelForCausalLM.from_pretrained(decoder_model_path)
    model = Wav2vec2_Gpt2(encoder=encoder,decoder=decoder)#.cuda("cuda:0")
    processor = Wav2Vec2Processor.from_pretrained("/data2_from_58175/huggingface/models/wav2vec2-large-960h-lv60-self")
    gpt_path = "/data2_from_58175/huggingface/models/distilgpt2"
    gpt_model = AutoModelForCausalLM.from_pretrained(gpt_path)
    gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_path)
    gpt_tokenizer.pad_token = 50256
    
    # print(model)
    # input_values = torch.randn(1,32000)
    # attention_mask = torch.ones(1,32000)
    swbdtest5h = load_from_disk("/home/data/fisher_swbd_nodup_onlyspeech/swbdtest5h")
    input_values = processor(swbdtest5h[0]["speech"], return_tensors="pt", padding="longest",sampling_rate=16000).input_values
    attention_mask = processor(swbdtest5h[0]["speech"], return_tensors="pt", padding="longest",sampling_rate=16000).attention_mask
    
    # wav_file_path = "/tsdata/diarization/voxconverse21_duke/DEV/audio/vgaez.wav"
    # speech,sampling_rate = librosa.load(wav_file_path,sr=16000)
    # speech = speech[:16000*120]
    # print(speech.shape)
    # input_values = processor(speech, return_tensors="pt", padding="longest",sampling_rate=16000).input_values.cuda("cuda:0")
    # attention_mask = processor(speech, return_tensors="pt", padding="longest",sampling_rate=16000).attention_mask.cuda("cuda:0")
    
    # (best_score_ctc,best_score_att,best_score),predicted_ids,predicted_ids_bs = attention_rescoring(input_values=input_values,
                                                         # attention_mask=attention_mask,
                                                         # model=model,
                                                         # beam_size=20,
                                                         # ctc_weight=0.5,
                                                         # output_prefix_beam_search=True)
    vocab_dict = processor.tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key,value) in vocab_dict.items())
    vocabulary = [x[1].replace("|", " ") if x[1] not in processor.tokenizer.all_special_tokens else "_" for x in sort_vocab]
    # print(f"vocabulary = {vocabulary}")
    vocabulary_ = [x[1] for x in sort_vocab]
    # lm_path = "/tsdata/xsp/w2v2/lm_4gram_fisher.arpa"
    # alpha = 0.3
    lm_path = None
    alpha = 0.0
    # kenlm_decoder = get_kenlm_decoder(
                # vocabulary=vocabulary,
                # lm_path=lm_path,
                # beam_size=20,
                # cutoff_top_n=20,
                # alpha=0.3,
                # cutoff_prob=1.0,
                # beta=0.0,
                # num_processes=1,
                # blank_id=0,
                # log_probs_input=False
                # )
    # kenlm_decoder_ = get_kenlm_decoder_(
                # vocabulary=vocabulary_,
                # lm_path=lm_path,
                # alpha=alpha,
                # beta=0.0,
                # )
    
    kenlm_decoder_ = get_kenlm_decoder(
                vocabulary=vocabulary_,
                lm_path=lm_path,
                alpha=alpha,
                beta=0.0,
                rescoring_kenlm_model_path=None,
                gpt_decoder=gpt_model,
                tokenizer=gpt_tokenizer,
                )
    # print(BeamSearchDecoderCTC.model_container[kenlm_decoder_._model_key]._kenlm_model.score("ALL RIGHT THANKS"))
    # print(isinstance(BeamSearchDecoderCTC.model_container[kenlm_decoder_._model_key]._kenlm_model,kenlm.Model))
    # (best_score_ctc,best_score_att,best_score),predicted_ids,predicted_ids_bs = attention_rescoring_lm(input_values=input_values,
                                                         # attention_mask=attention_mask,
                                                         # model=model,
                                                         # ctc_weight=0.5,
                                                         # output_prefix_beam_search=True,
                                                         # kenlm_decoder=kenlm_decoder
                                                         # )
    with torch.no_grad():
        encoder_hidden_states, encoder_attention_mask, encoder_logits = model(input_values=input_values,attention_mask=attention_mask,forward_only_encoder=True)
    hyps = ctc_prefix_beam_search(logits=encoder_logits,beam_size=20)
    hyps_lm = ctc_prefix_beam_search_lm(logits=encoder_logits,beam_size=20,kenlm_decoder=kenlm_decoder_,processor=processor)
    print(f"hyps = {hyps}")
    print(f"len(hyps) = {len(hyps)}")
    print(f"hyps_lm = {hyps_lm}")
    print(f"len(hyps_lm) = {len(hyps_lm)}")
    # hyps_lm = ctc_prefix_beam_search_lm(
                        # logits=encoder_logits,
                        # vocabulary=vocabulary,
                        # lm_path=lm_path,
                        # alpha=0.3,
                        # beta=0,
                        # cutoff_top_n=20,
                        # beam_size=20,
                        # num_processes=1,
                        # blank_id=0,
                        # log_probs_input=False
                        # )
    # (best_score_ctc,best_score_att,best_score),predicted_ids,predicted_ids_bs = attention_rescoring_lm_(input_values=input_values,
                                                         # processor=processor,
                                                         # beam_size=20,
                                                         # attention_mask=attention_mask,
                                                         # model=model,
                                                         # ctc_weight=0.5,
                                                         # output_prefix_beam_search=True,
                                                         # kenlm_decoder=kenlm_decoder_
                                                         # )
    # (best_score_ctc,best_score_att,best_score),predicted_ids,predicted_ids_bs = attention_rescoring_lm__(input_values=input_values,
                                                         # processor=processor,
                                                         # gpt_model=gpt_model,
                                                         # gpt_tokenizer=gpt_tokenizer,
                                                         # beam_size=20,
                                                         # attention_mask=attention_mask,
                                                         # model=model,
                                                         # ctc_weight=0.5,
                                                         # output_prefix_beam_search=True,
                                                         # kenlm_decoder=kenlm_decoder_,
                                                         # )
    # print(predicted_ids)
if __name__ == "__main__":
    main()