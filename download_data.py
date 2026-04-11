from datasets import load_dataset
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser('Download DeepJSONEval dataset from HuggingFace')
    parser.add_argument('--saving-path', default='DeepJSONEval.xlsx', type=str, help='path to save the downloaded dataset')
    return parser.parse_args()

args = get_args()

print("Downloading DeepJSONEval dataset from HuggingFace...")
dataset = load_dataset("GTSAIInfraLabSOTAS/DeepJSON", split="train")
df = dataset.to_pandas()

writer = pd.ExcelWriter(args.saving_path)
df.to_excel(writer, sheet_name='sheet1', index=False)
writer.close()

print(f"Saved to '{args.saving_path}' ({len(df)} rows)")
