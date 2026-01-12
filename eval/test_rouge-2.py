"""
ROUGE 评估脚本 v2.0
评估模型在 TL;DR 数据集上的摘要质量
新增：平均长度统计、大规模样本支持、显存优化
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
import csv
import numpy as np
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import evaluate

# =========================================================
# 配置
# =========================================================

MODELS_TO_EVALUATE = [
    {"name": "Base-Qwen3-1.7B", "path": "../models/Qwen3-1.7B"},
    {"name": "SFT-Full", "path": "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"},
    {"name": "grpo-baseline", "path": "/workspace/pj-RL/experiments3/qwen3-grpo-final-10k/checkpoint-1500"},
    {"name": "grpo-axis", "path": "/workspace/pj-RL/experiments3/qwen3-grpo-axis-merged/checkpoint-1200-merged"},
    {"name": "grpo-hybrid", "path": "/workspace/pj-RL/qwen3-grpo-hybrid-rm/checkpoint-1950"},    
]

DATASET_PATH = "/workspace/pj-RL/datasets/openai_summarize_tldr"
TEST_SAMPLES = 500  # 已修改为 500 条

GENERATION_CONFIG = {
    "max_new_tokens": 80, # 稍微增加上限，给长摘要留出空间
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "repetition_penalty": 1.1,
}

RESULTS_DIR = "./rouge_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================================
# 函数定义
# =========================================================

def extract_post_only(prompt: str) -> str:
    """提取POST内容，确保推理引导词格式标准"""
    if "POST:" in prompt:
        prompt = prompt.split("POST:", 1)[1]
    
    if "TL;DR:" in prompt:
        post, _ = prompt.split("TL;DR:", 1)
        prompt = post.strip() + "\n\nTL;DR: " # 增加空格引导生成
    else:
        prompt = prompt.strip() + "\n\nTL;DR: "
    
    return prompt

def evaluate_model_rouge(model, tokenizer, test_dataset, device):
    print("📋 正在初始化 ROUGE 评估器...")
    try:
        rouge = evaluate.load("rouge")
    except Exception as e:
        print(f"❌ ROUGE 加载失败: {e}")
        raise e

    model.eval()
    predictions = []
    references = []
    gen_lengths = [] # 用于统计 token 长度
    
    print(f"🚀 Generating summaries for {len(test_dataset)} samples...")
    
    for example in tqdm(test_dataset):
        raw_prompt = example["prompt"]
        reference_summary = example["label"]
        clean_prompt = extract_post_only(raw_prompt)
        
        inputs = tokenizer(
            clean_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 统计生成的长度 (不含 Prompt)
        prompt_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][prompt_len:]
        gen_lengths.append(len(generated_tokens))
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 TL;DR: 之后的部分
        if "TL;DR:" in full_text:
            predicted_summary = full_text.split("TL;DR:")[-1].strip()
        else:
            predicted_summary = full_text.strip()
        
        predictions.append(predicted_summary)
        references.append(reference_summary)
    
    print("📊 Computing ROUGE scores...")
    rouge_results = rouge.compute(
        predictions=predictions, 
        references=references,
        use_stemmer=True
    )
    
    avg_len = np.mean(gen_lengths)
    return rouge_results, avg_len, predictions, references

# =========================================================
# 主函数
# =========================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    # 增加安全性检查，防止样本数溢出
    max_test = len(dataset["test"])
    num_samples = min(TEST_SAMPLES, max_test)
    test_dataset = dataset["test"].select(range(num_samples))
    print(f"Loaded {num_samples} test samples")
    
    all_summary_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_config in MODELS_TO_EVALUATE:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\n{'='*20} Evaluating: {model_name} {'='*20}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        rouge_scores, avg_gen_len, _, _ = evaluate_model_rouge(
            model, tokenizer, test_dataset, device
        )
        
        print(f"\n✨ {model_name} Results:")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"  Avg Gen Length: {avg_gen_len:.2f} tokens")
        
        # 记录结果
        res = {
            "model_name": model_name,
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL'],
            "avg_len": avg_gen_len,
            "timestamp": timestamp
        }
        all_summary_results.append(res)
        
        # 保存单个模型的详细 JSON
        json_path = f"{RESULTS_DIR}/{model_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)

        # 彻底释放显存
        del model
        del tokenizer
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    # 保存对比汇总 CSV
    csv_path = f"{RESULTS_DIR}/final_comparison_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "rouge1", "rouge2", "rougeL", "avg_len", "timestamp"])
        writer.writeheader()
        writer.writerows(all_summary_results)
    
    print(f"\n✅ All tests done. Comparison saved to: {csv_path}")
    
    # 打印最终控制台表格
    print("\n" + "综 合 对 比 表".center(60, "-"))
    print(f"{'Model':<25} {'R-1':<8} {'R-2':<8} {'R-L':<8} {'AvgLen':<8}")
    for r in all_summary_results:
        print(f"{r['model_name']:<25} {r['rouge1']:<8.4f} {r['rouge2']:<8.4f} {r['rougeL']:<8.4f} {r['avg_len']:<8.1f}")

if __name__ == "__main__":
    main()