import torch
import json
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 配置
MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-multi-axis/final_axis_rm"
DATASET_PATH = "/workspace/pj-RL/datasets/summarize_axis"
SAVE_PATH = "axis_rm_stats.json"
DEVICE = "cuda"

# 加载训练集 (用 90% 的那部分)
raw_ds = load_from_disk(DATASET_PATH)["validation"]
ds_split = raw_ds.train_test_split(test_size=0.1, seed=42)
train_dataset = ds_split["train"] 

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

all_logits = []

print(">>> 正在统计训练集分布以计算归一化参数...")
with torch.no_grad():
    for i in tqdm(range(len(train_dataset))):
        item = train_dataset[i]
        text = f"{item['info']['post']}\n\nTL;DR:{item['summary']['text']}{tokenizer.eos_token}"
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        logits = model(**inputs).logits.float().cpu().numpy()[0]
        all_logits.append(logits)

all_logits = np.array(all_logits)
axes = ["overall", "accuracy", "coverage", "coherence"]
stats = {}

for i, axis in enumerate(axes):
    stats[axis] = {
        "mean": float(all_logits[:, i].mean()),
        "std": float(all_logits[:, i].std())
    }

with open(SAVE_PATH, "w") as f:
    json.dump(stats, f, indent=4)

print(f">>> 统计量已保存至 {SAVE_PATH}。请在 GRPO 训练中使用这些值进行归一化。")
