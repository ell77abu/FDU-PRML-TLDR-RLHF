import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm
import os

# ============================
# Config
# ============================
RM_PATH   = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
DATA_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"
SAVE_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-normalized"

SAMPLE_SIZE = 2000     # how many chosen summaries to estimate mean
BATCH_SIZE  = 8
MAX_LENGTH  = 1024

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================
# Load RM + tokenizer
# ============================
print("🔹 Loading Reward Model...")
tokenizer = AutoTokenizer.from_pretrained(RM_PATH, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    RM_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
model.eval()

# ============================
# Load RM dataset
# ============================
print("🔹 Loading RM dataset...")
dataset = load_from_disk(DATA_PATH)["train"]

# ============================
# Build chosen (human reference) texts
# ============================
print("🔹 Building human reference summaries...")
texts = []
n = min(SAMPLE_SIZE, len(dataset))

for i in range(n):
    ex = dataset[i]
    prompt = f"{ex['info']['post']}\n\nTL;DR:"
    chosen = ex["summaries"][ex["choice"]]["text"]
    texts.append(prompt + chosen + tokenizer.eos_token)

# ============================
# Compute mean reward
# ============================
print("⚖️ Computing mean RM score on human references...")

all_scores = []

with torch.no_grad():
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(model.device)

        scores = model(**inputs).logits.squeeze(-1)
        all_scores.extend(scores.cpu().float().tolist())

mean_reward = sum(all_scores) / len(all_scores)
print(f"  → Human reference mean = {mean_reward:.6f}")

# ============================
# Apply calibration (shift RM head bias)
# ============================
print("🔧 Applying zero-point calibration...")

# Qwen-style RM head is called "score"
if not hasattr(model, "score"):
    raise RuntimeError("❌ This RM does not have a `.score` head — cannot calibrate.")

if model.score.bias is None:
    print("💡 RM score head has no bias. Creating a new one with bias=True...")
    old_score = model.score
    new_score = torch.nn.Linear(old_score.in_features, old_score.out_features, bias=True)
    new_score.weight.data = old_score.weight.data
    # Initialize bias to exactly 0 on the correct device
    new_score.bias.data = torch.zeros(old_score.out_features, device=old_score.weight.device, dtype=old_score.weight.dtype)
    model.score = new_score.to(model.device).to(model.dtype)

with torch.no_grad():
    # Directly set the bias to -mean_reward to shift the distribution to 0
    # mean_reward is the average score, so subtracting it from every output makes the new average 0
    model.score.bias.data.fill_(-mean_reward)
    print(f"  → Score head bias set to: {model.score.bias.item():.6f}")

# ============================
# Verify
# ============================
print("🔍 Verifying new mean on full sample...")

with torch.no_grad():
    verify_scores = []
    # Verify on the same 'n' samples used for mean calculation
    for i in tqdm(range(0, n, BATCH_SIZE)):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(model.device)
        scores = model(**inputs).logits.squeeze(-1)
        verify_scores.extend(scores.cpu().float().tolist())

new_mean = sum(verify_scores) / len(verify_scores)
print(f"  → New human reference mean = {new_mean:.6f} (Target ≈ 0)")

# ============================
# Save calibrated RM
# ============================
print("💾 Saving calibrated RM...")

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("✅ Done. Calibrated RM saved to:", SAVE_PATH)
