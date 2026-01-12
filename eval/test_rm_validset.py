import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# model_path = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
model_path = "/workspace/pj-RL/experiments3/qwen3-rm-normalized"

# 1. 强制对齐分词器设置
tok = AutoTokenizer.from_pretrained(model_path)
tok.padding_side = "right" 
tok.pad_token = tok.eos_token 

rm = AutoModelForSequenceClassification.from_pretrained(
    model_path, 
    dtype=torch.bfloat16, 
    trust_remote_code=True
).to("cuda")

# --- Patch: Manually load score.bias since Qwen3 defaults to bias=False ---
if rm.score.bias is None:
    print("💡 Detecting normalized RM checkpoint. Applying bias patch...")
    old_score = rm.score
    new_score = torch.nn.Linear(old_score.in_features, old_score.out_features, bias=True)
    rm.score = new_score.to(rm.device).to(rm.dtype)
    
    # Reload weights to pick up the bias from safe_tensors
    import os
    from safetensors.torch import load_file
    weight_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(weight_path):
        state_dict = load_file(weight_path)
        rm.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded score.bias: {rm.score.bias.item():.4f}")
    else:
        print("⚠️ model.safetensors not found, bias might not be loaded.")

rm.eval()

ds = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")
# 使用与训练时相同的 shuffle 种子，确保验证集一致
subset = ds["validation"].select(range(500)) #  .shuffle(seed=42)

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
chosen_scores = []
batch_size = 4 

for i in tqdm(range(0, len(subset), batch_size)):
    indices = range(i, min(i + batch_size, len(subset)))
    batch = subset.select(indices)
    
    chosen_batch = []
    rejected_batch = []
    
    for ex in batch:
        # 3. 严格对齐训练时的拼接逻辑 (与 rm.py / normalize_rm.py 一致)
        prefix = f"{ex['info']['post']}\n\nTL;DR:"
        c_text = f"{ex['summaries'][ex['choice']]['text']}{tok.eos_token}"
        r_text = f"{ex['summaries'][1-ex['choice']]['text']}{tok.eos_token}"
        
        chosen_batch.append(prefix + c_text)
        rejected_batch.append(prefix + r_text)
    
    sc = get_scores(chosen_batch)
    sr = get_scores(rejected_batch)
    
    chosen_scores.extend(sc.tolist())
    for s_c, s_r in zip(sc, sr):
        if s_c > s_r:
            wins += 1
        margins.append(s_c - s_r)

print(f"\n✅ 结果分析 (N={len(subset)}):")
print(f"准确率: {wins/len(subset):.2%}")
print(f"平均分差 (Margin): {sum(margins)/len(margins):.4f}")
print(f"Chosen 平均分 (Mean Score): {sum(chosen_scores)/len(chosen_scores):.4f}")