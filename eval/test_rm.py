import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

model_path = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"

# 1. 强制对齐分词器设置
tok = AutoTokenizer.from_pretrained(model_path)
tok.padding_side = "right" 
tok.pad_token = tok.eos_token 

rm = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    dtype=torch.bfloat16, 
    trust_remote_code=True
).eval().to("cuda")

ds = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")
# 使用与训练时相同的 shuffle 种子，确保验证集一致
subset = ds["validation"].shuffle().select(range(500)) #  .shuffle(seed=42)

def get_scores(texts):
    # 2. 模仿 RewardTrainer 的处理方式
    inputs = tok(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    ).to("cuda")
    
    with torch.no_grad():
        outputs = rm(**inputs)
        # 获取 sequence classification 的 logits
        logits = outputs.logits
    return logits.float().cpu().numpy().flatten()

wins = 0
margins = []
batch_size = 4 

for i in tqdm(range(0, len(subset), batch_size)):
    indices = range(i, min(i + batch_size, len(subset)))
    batch = subset.select(indices)
    
    chosen_batch = []
    rejected_batch = []
    
    for ex in batch:
        # 3. 严格对齐训练时的拼接逻辑
        prefix = f"{ex['info']['post']}\n\nTL;DR:"
        # 注意这里的空格处理需与训练 preprocess_function 完全一致
        c_text = f" {ex['summaries'][ex['choice']]['text']}{tok.eos_token}"
        r_text = f" {ex['summaries'][1-ex['choice']]['text']}{tok.eos_token}"
        
        chosen_batch.append(prefix + c_text)
        rejected_batch.append(prefix + r_text)
    
    sc = get_scores(chosen_batch)
    sr = get_scores(rejected_batch)
    
    for s_c, s_r in zip(sc, sr):
        if s_c > s_r:
            wins += 1
        margins.append(s_c - s_r)

print(f"\n✅ 结果分析 (N={len(subset)}):")
print(f"准确率: {wins/len(subset):.2%}")
print(f"平均分差: {sum(margins)/len(margins):.4f}")