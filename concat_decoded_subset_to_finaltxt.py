import glob
import argparse
import datasets
from datasets import load_dataset,load_from_disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make dataset for w2v2_finetuning")
    parser.add_argument("-d_subset", "--subset_dir", type=str, default="decode_mp", help="the direction of subset after decode_mp")
    parser.add_argument("-f_output", "--output_file", type=str, default="result.txt", help="the name of output file(.txt)")
    parser.add_argument("-sd", "--save_result_dataset", action='store_true', default=False,help="save datasets with transcription or not")
    args = parser.parse_args()
    print(f"args = {args}")
    subset_dir = args.subset_dir
    output_file = args.output_file
    save_result_dataset = args.save_result_dataset

    subset_path_list = glob.glob(f"{subset_dir}/*[0-9]")
    subset_path_list = sorted(subset_path_list,key=lambda x:int(x.split("/")[-1]))
    print(f"subset_path_list = {subset_path_list}")
    subset_list = []
    for i in subset_path_list:
        subset = load_from_disk(i)
        subset_list.append(subset)
    result= datasets.concatenate_datasets(subset_list)
    print(f"result_hf_datasets = {result}")
    result.save_to_disk(subset_dir) if save_result_dataset else None
    if "text" in result.column_names:
        wer_metric = datasets.load_metric("wer_no_memory_error")
        wer_bs=wer_metric.compute(predictions=result["transcription_bs"],references=result["text"])
        wer=wer_metric.compute(predictions=result["transcription"],references=result["text"])
        print(f"wer_bs = {wer_bs}")
        print(f"wer = {wer}")


    line_list = []
    for i in result:
        line = i["id"] + " " + i["transcription"].replace(" ","") + "\n"
        line_list.append(line)
    with open(output_file,mode="w",encoding="utf-8") as f:
        f.writelines(line_list)
    print(f"output_file: {output_file}")
