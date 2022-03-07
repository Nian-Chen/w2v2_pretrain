import glob
import pandas as pd
from datasets import load_dataset,load_from_disk
import os
os.environ["HF_HOME"] = "huggingface-home"
os.environ["TRANSFORMERS_CACHE"] = "huggingface-home/transformers"
os.environ["HF_MODELS_CACHE"] = "huggingface-home/datasets"
os.environ["HF_METRICS_CACHE"] = "huggingface-home/metrics"
os.environ["HF_DATASETS_CACHE"] = "huggingface-home/datasets"
if __name__ == "__main__":
    btest_list = glob.glob("B-Test/*.wav")
    line_list = []
    for i in btest_list:
        file = i
        utt_id = i.split("/")[-1].split(".")[0]
        line = utt_id+"\t"+file
        line_list.append(line)
    lines2write = [line+"\n" for line in line_list]
    with open("B-Test_wavscp",mode="w",encoding="utf-8") as f:
        f.writelines(lines2write)
    file = "B-Test_wavscp"
    ws_btest_csv = pd.read_csv(file,delimiter="\t",names=['id','file'])
    ws_btest_csv.to_csv(f"{file}.csv", encoding='utf_8_sig', index=False)
    ws_btest_dataset = load_dataset('csv', data_files={"test":f"{file}.csv"})
    ws_btest_dataset["test"].save_to_disk("hf_datasets/test")
    print("output_wavscp_file: B-Test_wavscp")
    print("output_hf_datasets_file: hf_datasets/test")