import os
import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# ===============================
# 1. 环境与路径配置
# ===============================
MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-multi-axis/final_axis_rm"
DATASET_PATH = "/workspace/pj-RL/datasets/summarize_axis"
OUTPUT_DIR = "/workspace/pj-RL/experiments3/qwen3-rm-multi-axis/analysis"
TARGET_AXES = ["overall", "accuracy", "coverage", "coherence"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# 2. 严格对齐数据划分 (Seed 42)
# ===============================
print(">>> 正在加载测试集 (严格隔离部分)...")
raw_ds = load_from_disk(DATASET_PATH)["validation"]
ds_split = raw_ds.train_test_split(test_size=0.1, seed=42)
test_dataset = ds_split["test"] 

# ===============================
# 3. 加载模型
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE)
model.eval()

# ===============================
# 4. 推理与真实标签提取
# ===============================
preds_list = []
gts_list = []

print(f">>> 开始对 {len(test_dataset)} 条测试样本进行对比分析...")

for i in tqdm(range(len(test_dataset))):
    item = test_dataset[i]
    info = item["info"]
    summary = item["summary"]
    
    # 构建输入文本
    text = f"{info['post']}\n\nTL;DR:{summary['text']}{tokenizer.eos_token}"
    
    # 提取真实标签并归一化 (1-7 -> 0-1)
    axes_dict = summary.get("axes", {})
    gt = [(float(axes_dict.get(a, 4) or 4) - 1.0) / 6.0 for a in TARGET_AXES]
    gts_list.append(gt)
    
    # 模型推理
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits.float().cpu().numpy()[0]
        preds_list.append(logits)

preds = np.array(preds_list)
gts = np.array(gts_list)

# ===============================
# 5. 综合对比分析
# ===============================
print("\n" + "="*70)
print(f"{'评估维度':<12} | {'Pearson':<10} | {'Spearman':<10} | {'MAE':<10} | {'预测Std':<10}")
print("-" * 70)

for i, axis in enumerate(TARGET_AXES):
    p_corr, _ = stats.pearsonr(preds[:, i], gts[:, i])
    s_corr, _ = stats.spearmanr(preds[:, i], gts[:, i])
    mae = np.mean(np.abs(preds[:, i] - gts[:, i]))
    pred_std = preds[:, i].std()
    
    print(f"{axis:<12} | {p_corr:>10.4f} | {s_corr:>10.4f} | {mae:>10.4f} | {pred_std:>10.4f}")

# ===============================
# 6. 可视化：预测值 vs 真实值 散点图
# ===============================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, axis in enumerate(TARGET_AXES):
    axes[i].scatter(gts[:, i], preds[:, i], alpha=0.4, s=20, color='darkblue')
    axes[i].plot([0, 1], [0, 1], transform=axes[i].transAxes, color='red', linestyle='--', label='Perfect Match')
    axes[i].set_title(f"Correlation: {axis.capitalize()}")
    axes[i].set_xlabel("Ground Truth (Normalized)")
    axes[i].set_ylabel("Model Prediction")
    axes[i].grid(True, linestyle=':', alpha=0.6)
    axes[i].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/prediction_vs_gt_comparison.pdf")
print(f"\n>>> 对比分析图表已保存至: {OUTPUT_DIR}/prediction_vs_gt_comparison.pdf")

# ===============================
# 7. 案例抽查
# ===============================
print("\n>>> 随机抽查（此处展示第一条):")
print(f"文本截断: {test_dataset[0]['summary']['text'][:100]}...")
print(f"真实分数 (1-7): {gts[0]*6+1}")
print(f"预测分数 (1-7): {preds[0]*6+1}")