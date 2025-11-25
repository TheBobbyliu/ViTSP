from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

import json
from tqdm import tqdm
dataset = json.load(open("r001/r001.json"))

lens = []
for x in tqdm(dataset):
    text = x['conversations'][0]['value']
    tokens = tokenizer.encode(text)
    lens.append(len(tokens))

print(max(lens))
