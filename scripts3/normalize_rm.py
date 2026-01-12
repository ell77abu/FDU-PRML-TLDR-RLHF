import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm

# ============================
# Config
# ============================
RM_PATH   = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
DATA_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"
SAVE_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-normalized"

SAMPLE_SIZE = 5000 
BATCH_SIZE  = 8
MAX_LENGTH  = 1024

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================
# Load RM
# ============================
print("Loading Reward Model...")
tokenizer = AutoTokenizer.from_pretrained(RM_PATH, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
model.eval()

# ============================
# Compute mean reward (使用随机采样)
# ============================
print("🎲 Shuffling dataset for representative sampling...")
dataset = load_from_disk(DATA_PATH)["train"].shuffle(seed=42)

n = min(SAMPLE_SIZE, len(dataset))
texts = []
for i in range(n):
    ex = dataset[i]
    prompt = f"{ex['info']['post']}\n\nTL;DR:"
    texts.append(prompt + ex["summaries"][ex["choice"]]["text"] + tokenizer.eos_token)

print(f"Computing mean RM score on {n} samples...")
all_scores = []
with torch.no_grad():
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(model.device)
        scores = model(**inputs).logits.squeeze(-1)
        all_scores.extend(scores.cpu().float().tolist())

mean_reward = sum(all_scores) / len(all_scores)
print(f"  → Human reference mean = {mean_reward:.6f}")

# ============================
# Apply calibration & Fix Architecture
# ============================
print("🔧 Applying zero-point calibration...")

# 强制替换为带 bias 的层
old_score = model.score
new_score = torch.nn.Linear(old_score.in_features, old_score.out_features, bias=True)
new_score.weight.data = old_score.weight.data
new_score.bias.data.fill_(-mean_reward) # 核心：将均值的负值填入 bias
model.score = new_score.to(model.device).to(model.dtype)

# ============================
# 验证：再次运行同样的 n 条数据
# ============================
print("🔍 Verifying calibration on all samples...")
with torch.no_grad():
    verify_scores = []
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(model.device)
        scores = model(**inputs).logits.squeeze(-1)
        verify_scores.extend(scores.cpu().float().tolist())

new_mean = sum(verify_scores) / n
print(f"  → New human reference mean = {new_mean:.6f} (Should be near 0)")

# ============================
# 保存模型（使用 PyTorch 格式确保 bias 被保存）
# ============================
print("Saving calibrated RM...")

# 使用 PyTorch 格式保存（.bin）而不是 safetensors
# 这样可以确保 bias 参数被正确保存
model.save_pretrained(SAVE_PATH, safe_serialization=False)
tokenizer.save_pretrained(SAVE_PATH)

# 修改 config.json 添加自定义字段标记这是归一化模型
config_file = os.path.join(SAVE_PATH, "config.json")
with open(config_file, "r") as f:
    config = json.load(f)

# 添加自定义字段
config["_normalized_rm"] = True
config["_normalization_bias"] = float(-mean_reward)
config["_score_head_has_bias"] = True

with open(config_file, "w") as f:
    json.dump(config, f, indent=2)

print(f"Done. Calibrated RM saved to: {SAVE_PATH}")
print(f"Config updated with normalization metadata")
print(f"   - Bias value: {-mean_reward:.6f}")
print(f"   - Saved in PyTorch format (.bin) to preserve bias layer")