#!/usr/bin/env python3
"""
测试 RM 归一化效果
对比原始 RM 和归一化 RM 的输出分布统计指标
"""

import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm

# ============================
# Config
# ============================
ORIGINAL_RM_PATH = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
NORMALIZED_RM_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-normalized"
DATA_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"

TEST_SIZE = 2000  # 测试样本数量
BATCH_SIZE = 8
MAX_LENGTH = 1024

def load_model(path, name):
    """加载 RM 模型"""
    print(f"🔹 Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 检查是否是归一化模型，如果 score head 没有 bias，需要重新创建
    if hasattr(model, "score") and model.score.bias is None:
        print(f"  ⚠️  Score head has no bias, attempting to load from state dict...")
        # 尝试从 state dict 中加载 bias
        state_dict_path = path + "/pytorch_model.bin"
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=model.device)
            if "score.bias" in state_dict:
                # 重新创建带 bias 的层
                old_score = model.score
                new_score = torch.nn.Linear(old_score.in_features, old_score.out_features, bias=True)
                new_score.weight.data = old_score.weight.data
                new_score.bias.data = state_dict["score.bias"].to(model.device, dtype=model.dtype)
                model.score = new_score
                print(f"  ✅ Successfully loaded score.bias = {new_score.bias.item():.6f}")
            else:
                print(f"  ℹ️  No score.bias found in state dict")
        else:
            print(f"  ℹ️  No pytorch_model.bin found, using safetensors (may not have bias)")
    elif hasattr(model, "score") and model.score.bias is not None:
        print(f"  ✅ Score head has bias = {model.score.bias.item():.6f}")
    
    model.eval()
    return tokenizer, model

def compute_scores(model, tokenizer, texts):
    """计算一批文本的 RM 分数"""
    all_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Computing scores"):
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
    return np.array(all_scores)

def print_statistics(scores, model_name):
    """打印统计指标"""
    print(f"\n{'='*60}")
    print(f"📊 {model_name} Statistics")
    print(f"{'='*60}")
    print(f"  Sample Size:     {len(scores)}")
    print(f"  Mean:            {np.mean(scores):.6f}")
    print(f"  Std Dev:         {np.std(scores):.6f}")
    print(f"  Min:             {np.min(scores):.6f}")
    print(f"  Max:             {np.max(scores):.6f}")
    print(f"  Range:           {np.max(scores) - np.min(scores):.6f}")
    print(f"  Median:          {np.median(scores):.6f}")
    print(f"  25th Percentile: {np.percentile(scores, 25):.6f}")
    print(f"  75th Percentile: {np.percentile(scores, 75):.6f}")
    print(f"  5th Percentile:  {np.percentile(scores, 5):.6f}")
    print(f"  95th Percentile: {np.percentile(scores, 95):.6f}")

def main():
    print("\n" + "="*60)
    print("🎯 RM Normalization Test")
    print("="*60)
    
    # 加载数据集
    print("\n🔹 Loading dataset...")
    dataset = load_from_disk(DATA_PATH)["train"].shuffle(seed=42)
    n = min(TEST_SIZE, len(dataset))
    
    # 构建测试文本（使用 chosen summaries）
    print(f"🔹 Building {n} test samples...")
    texts = []
    for i in range(n):
        ex = dataset[i]
        prompt = f"{ex['info']['post']}\n\nTL;DR:"
        chosen = ex["summaries"][ex["choice"]]["text"]
        texts.append(prompt + chosen)
    
    # 加载两个模型
    original_tokenizer, original_model = load_model(ORIGINAL_RM_PATH, "Original RM")
    normalized_tokenizer, normalized_model = load_model(NORMALIZED_RM_PATH, "Normalized RM")
    
    # 为文本添加 EOS token
    texts_original = [t + original_tokenizer.eos_token for t in texts]
    texts_normalized = [t + normalized_tokenizer.eos_token for t in texts]
    
    # 计算分数
    print("\n⚖️ Computing scores for Original RM...")
    original_scores = compute_scores(original_model, original_tokenizer, texts_original)
    
    print("\n⚖️ Computing scores for Normalized RM...")
    normalized_scores = compute_scores(normalized_model, normalized_tokenizer, texts_normalized)
    
    # 打印统计信息
    print_statistics(original_scores, "Original RM")
    print_statistics(normalized_scores, "Normalized RM")
    
    # 对比差异
    print(f"\n{'='*60}")
    print(f"📈 Comparison")
    print(f"{'='*60}")
    mean_diff = np.mean(normalized_scores) - np.mean(original_scores)
    print(f"  Mean Shift:      {mean_diff:.6f}")
    print(f"  Expected Shift:  {-np.mean(original_scores):.6f}")
    print(f"  Shift Accuracy:  {abs(mean_diff + np.mean(original_scores)):.6f} (should be ~0)")
    
    # 检查相对顺序是否保持
    rank_correlation = np.corrcoef(original_scores, normalized_scores)[0, 1]
    print(f"  Rank Correlation: {rank_correlation:.6f} (should be ~1.0)")
    
    # 最终总结
    print(f"\n{'='*60}")
    print(f"✅ Test Summary")
    print(f"{'='*60}")
    
    if abs(np.mean(normalized_scores)) < 0.01:
        print(f"  ✅ Normalization successful: mean ≈ 0")
    else:
        print(f"  ⚠️  Mean is {np.mean(normalized_scores):.6f}, expected ~0")
    
    if rank_correlation > 0.999:
        print(f"  ✅ Ranking preserved: correlation = {rank_correlation:.6f}")
    else:
        print(f"  ⚠️  Ranking slightly changed: correlation = {rank_correlation:.6f}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
