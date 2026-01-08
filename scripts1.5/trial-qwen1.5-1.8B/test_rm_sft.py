import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# --- 1. 配置路径 ---
sft_model_path = "./models/sft-tldr/final_checkpoint"
rm_model_path = "./models/rm-tldr/final_rm"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 加载模型与 Tokenizer ---
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
# SFT 模型用于生成摘要
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path, torch_dtype=torch.bfloat16
).to(device).eval()

# RM 模型用于评分
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path, torch_dtype=torch.bfloat16, load_in_8bit=True, device_map={"": device}
).eval()

# --- 3. 加载测试数据集 ---
dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test")
# 选取 50 条数据进行对比测试
test_samples = dataset.shuffle(seed=42).select(range(50))

results = []

print("开始对比评估...")
for i, sample in enumerate(tqdm(test_samples)):
    prompt = sample['prompt']
    human_summary = sample['label']
    
    # --- Step A: SFT 模型生成摘要 ---
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sft_model.generate(
            **inputs, 
            max_new_tokens=80, 
            # do_sample=True, 
            do_sample=False, 
            temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    # 提取生成的摘要部分
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sft_summary = full_text.split("TL;DR:")[-1].strip()

    # --- Step B: RM 评分 ---
    # 构造 RM 的输入格式：Prompt + Summary
    text_human = prompt + human_summary + tokenizer.eos_token
    text_sft = prompt + sft_summary + tokenizer.eos_token
    
    rm_inputs = tokenizer([text_human, text_sft], return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        logits = reward_model(**rm_inputs).logits.squeeze(-1)
        score_human = logits[0].item()
        score_sft = logits[1].item()

    results.append({
        "id": i,
        "human_score": score_human,
        "sft_score": score_sft,
        "win": score_human > score_sft,
        "sft_text": sft_summary,
        "human_text": human_summary
    })

# --- 4. 结果分析 ---
df = pd.DataFrame(results)
win_rate = df['win'].mean()

print("\n" + "="*30)
print(f"评估完成！对比结果如下：")
print(f"人类摘要平均分: {df['human_score'].mean():.4f}")
print(f"SFT 摘要平均分: {df['sft_score'].mean():.4f}")
print(f"RM 对人类的胜率 (Win Rate): {win_rate*100:.2f}%")
print("="*30)

# 保存详细对比到 CSV 方便查阅
df.to_csv("rm_vs_human_test.csv", index=False)
print("详细对比已保存至 rm_vs_human_test.csv")