import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd
import os

# --- 1. 配置路径 ---
base_model_path = "/workspace/pj-RL/models/Qwen3-1.7B"
sft_model_path = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
rm_model_path = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
ppo_model_path = "/workspace/pj-RL/experiments3/qwen3-ppo-5000merged"
# ppo_model_path = "/workspace/pj-RL/experiments3/qwen3-ppo-5000/final_ppo_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_post_only(prompt: str) -> str:
    """
    仿照sft.py中的prompt处理格式：
    输入：
        SUBREDDIT: ...
        TITLE: ...
        POST: xxx
        TL;DR:

    输出：
        xxx
        TL;DR:
    """
    # 去掉 SUBREDDIT / TITLE
    if "POST:" in prompt:
        prompt = prompt.split("POST:", 1)[1]

    # 保留 TL;DR:
    if "TL;DR:" in prompt:
        post, _ = prompt.split("TL;DR:", 1)
        prompt = post.strip() + "\n\nTL;DR:"
    else:
        prompt = prompt.strip() + "\n\nTL;DR:"

    return prompt

# --- 2. 加载模型与 Tokenizer ---
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

# Base 模型用于生成摘要
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16
).to(device).eval()

# SFT 模型用于生成摘要
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path, torch_dtype=torch.bfloat16
).to(device).eval()

# RM 模型用于评分
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path, torch_dtype=torch.bfloat16, load_in_8bit=True, device_map={"": device}
).eval()

# PPO 模型用于生成摘要
ppo_model = AutoModelForCausalLM.from_pretrained(
    ppo_model_path, torch_dtype=torch.bfloat16
).to(device).eval()

# --- 3. 加载测试数据集 ---
dataset = load_from_disk("/workspace/pj-RL/datasets/openai_summarize_tldr")
# 选取 100 条数据进行对比测试
test_samples = dataset["test"].shuffle(seed=42).select(range(100))

results = []

print("开始对比评估...")
for i, sample in enumerate(tqdm(test_samples)):
    raw_prompt = sample['prompt']
    human_summary = sample['label']

    # 处理prompt格式（仿照sft.py）
    prompt = extract_post_only(raw_prompt)

    # # --- Step A: Base 模型生成摘要 ---
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     outputs = base_model.generate(
    #         **inputs,
    #         max_new_tokens=80,
    #         do_sample=False,  # 使用贪婪解码获得确定性结果
    #         temperature=0.8,
    #         repetition_penalty=1.1,
    #         pad_token_id=tokenizer.pad_token_id
    #     )
    # # 提取生成的摘要部分
    # full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # base_summary = full_text.split("TL;DR:")[-1].strip()

    # --- Step B: SFT 模型生成摘要 ---
    with torch.no_grad():
        outputs = sft_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,  # 使用贪婪解码获得确定性结果
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    # 提取生成的摘要部分
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sft_summary = full_text.split("TL;DR:")[-1].strip()

    # --- Step C: PPO 模型生成摘要 ---
    with torch.no_grad():
        outputs = ppo_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,  # 使用贪婪解码获得确定性结果
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    # 提取生成的摘要部分
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ppo_summary = full_text.split("TL;DR:")[-1].strip()

    # --- Step D: RM 评分三组对比 ---

    # 对比1: 人类摘要 vs SFT摘要
    text_human = prompt + human_summary + tokenizer.eos_token
    text_sft = prompt + sft_summary + tokenizer.eos_token

    rm_inputs_1 = tokenizer([text_human, text_sft], return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits_1 = reward_model(**rm_inputs_1).logits.squeeze(-1)
        score_human_vs_sft = logits_1[0].item()
        score_sft_vs_human = logits_1[1].item()

    # 对比2: 人类摘要 vs PPO摘要
    text_human_2 = prompt + human_summary + tokenizer.eos_token
    text_ppo = prompt + ppo_summary + tokenizer.eos_token

    rm_inputs_2 = tokenizer([text_human_2, text_ppo], return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits_2 = reward_model(**rm_inputs_2).logits.squeeze(-1)
        score_human_vs_ppo = logits_2[0].item()
        score_ppo_vs_human = logits_2[1].item()

    # 对比3: SFT摘要 vs PPO摘要
    text_sft_3 = prompt + sft_summary + tokenizer.eos_token
    text_ppo_3 = prompt + ppo_summary + tokenizer.eos_token

    rm_inputs_3 = tokenizer([text_sft_3, text_ppo_3], return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits_3 = reward_model(**rm_inputs_3).logits.squeeze(-1)
        score_sft_vs_ppo = logits_3[0].item()
        score_ppo_vs_sft = logits_3[1].item()

    results.append({
        # 对比1: 人类 vs SFT
        "human_vs_sft_human_score": score_human_vs_sft,
        "human_vs_sft_sft_score": score_sft_vs_human,
        "human_vs_sft_human_win": score_human_vs_sft > score_sft_vs_human,

        # 对比2: 人类 vs PPO
        "human_vs_ppo_human_score": score_human_vs_ppo,
        "human_vs_ppo_ppo_score": score_ppo_vs_human,
        "human_vs_ppo_human_win": score_human_vs_ppo > score_ppo_vs_human,

        # 对比3: SFT vs PPO
        "sft_vs_ppo_sft_score": score_sft_vs_ppo,
        "sft_vs_ppo_ppo_score": score_ppo_vs_sft,
        "sft_vs_ppo_sft_win": score_sft_vs_ppo > score_ppo_vs_sft,

        # 生成的文本
        "human_text": human_summary,
        "sft_text": sft_summary,
        "ppo_text": ppo_summary,
        "prompt": prompt
    })

# --- 4. 结果分析 ---
df = pd.DataFrame(results)

# 对比1: 人类摘要 vs SFT摘要
human_vs_sft_win_rate = df['human_vs_sft_human_win'].mean()

# 对比2: 人类摘要 vs PPO摘要
human_vs_ppo_win_rate = df['human_vs_ppo_human_win'].mean()

# 对比3: SFT摘要 vs PPO摘要
sft_vs_ppo_win_rate = df['sft_vs_ppo_sft_win'].mean()

print("\n" + "="*50)
print("RM对比评估结果汇总：")
print("="*50)
print(f"测试样本数量: {len(df)}")

print("\n对比1: 人类摘要 vs SFT摘要")
print(f"人类摘要平均得分: {df['human_vs_sft_human_score'].mean():.4f}")
print(f"SFT摘要平均得分: {df['human_vs_sft_sft_score'].mean():.4f}")
print(f"人类摘要胜率 (Human Win Rate): {human_vs_sft_win_rate*100:.2f}%")

print("\n对比2: 人类摘要 vs PPO摘要")
print(f"人类摘要平均得分: {df['human_vs_ppo_human_score'].mean():.4f}")
print(f"PPO摘要平均得分: {df['human_vs_ppo_ppo_score'].mean():.4f}")
print(f"人类摘要胜率 (Human Win Rate): {human_vs_ppo_win_rate*100:.2f}%")

print("\n对比3: SFT摘要 vs PPO摘要")
print(f"SFT摘要平均得分: {df['sft_vs_ppo_sft_score'].mean():.4f}")
print(f"PPO摘要平均得分: {df['sft_vs_ppo_ppo_score'].mean():.4f}")
print(f"SFT摘要胜率 (SFT Win Rate): {sft_vs_ppo_win_rate*100:.2f}%")

print("="*50)

# 保存详细对比到 CSV 方便查阅
output_file = "rm_comparison_evaluation.csv"
df.to_csv(output_file, index=False)
print(f"详细对比已保存至 {output_file}")

# 打印几个示例结果
print("\n示例结果 (前3个样本):")
for i in range(min(3, len(df))):
    row = df.iloc[i]
    print(f"\n样本 {i+1}:")
    print(f"人类摘要: {row['human_text'][:80]}...")
    print(f"SFT摘要: {row['sft_text'][:80]}...")
    print(f"PPO摘要: {row['ppo_text'][:80]}...")
    print(f"人类vs SFT - 人类胜: {row['human_vs_sft_human_win']}")
    print(f"人类vs PPO - 人类胜: {row['human_vs_ppo_human_win']}")
    print(f"SFT vs PPO - SFT胜: {row['sft_vs_ppo_sft_win']}")
