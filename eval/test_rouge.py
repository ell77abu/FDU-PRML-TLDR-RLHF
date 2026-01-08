"""
ROUGE评估脚本
评估模型在TL;DR数据集上的摘要质量
"""

import os
import torch
import json
import csv
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import evaluate

# =========================================================
# 配置
# =========================================================

MODELS_TO_EVALUATE = [
    {
        "name": "Base-Qwen3-1.7B",
        "path": "../models/Qwen3-1.7B",
    },
    {
        "name": "SFT-Full-Baseline",
        "path": "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint",
    },
    # {
    #     "name": "SFT-LoRA-Comparison",
    #     "path": "/workspace/pj-RL/experiments3/qwen3-sft-lora/final_checkpoint", 
    #     # 注意：如果是LoRA，请确保路径下是merge后的完整权重，或者脚本能自动处理adapter
    # },
    # {
    #     "name": "PPO-Baseline",
    #     "path": "../experiments3/qwen3-ppo-baseline/final_checkpoint",
    # },
    # {
    #     "name": "PPO-Improved",
    #     "path": "../experiments3/qwen3-ppo-improved/final_checkpoint",
    # },
]

DATASET_PATH = "/workspace/pj-RL/datasets/openai_summarize_tldr"
TEST_SAMPLES = 700

GENERATION_CONFIG = {
    "max_new_tokens": 60,
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

# 环境对齐
def extract_post_only(prompt: str) -> str:
    """提取POST内容，与SFT训练时保持一致"""
    if "POST:" in prompt:
        prompt = prompt.split("POST:", 1)[1]
    
    if "TL;DR:" in prompt:
        post, _ = prompt.split("TL;DR:", 1)
        prompt = post.strip() + "\n\nTL;DR:"
    else:
        prompt = prompt.strip() + "\n\nTL;DR:"
    
    return prompt

def evaluate_model_rouge(model, tokenizer, test_dataset, device):
    """评估模型的ROUGE分数"""
    model.eval()
    predictions = []
    references = []
    
    print(f"Generating summaries for {len(test_dataset)} samples...")
    
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
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "TL;DR:" in full_text:
            predicted_summary = full_text.split("TL;DR:")[-1].strip()
        else:
            predicted_summary = full_text.strip()
        
        predictions.append(predicted_summary)
        references.append(reference_summary)
    
    print("Computing ROUGE scores...")
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=predictions, 
        references=references,
        use_stemmer=True
    )
    
    return rouge_scores, predictions, references

# =========================================================
# 主函数
# =========================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    test_dataset = dataset["test"].select(range(TEST_SAMPLES))
    print(f"Loaded {len(test_dataset)} test samples")
    
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for model_config in MODELS_TO_EVALUATE:
        model_name = model_config["name"]
        model_path = model_config["path"]
        
        print(f"\nEvaluating: {model_name}")
        print(f"Path: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
        rouge_scores, predictions, references = evaluate_model_rouge(
            model, tokenizer, test_dataset, device
        )
        
        print(f"\n{model_name} ROUGE Scores:")
        print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
        print(f"  ROUGE-Lsum: {rouge_scores['rougeLsum']:.4f}")
        
        result_detail = {
            "model_name": model_name,
            "model_path": model_path,
            "timestamp": timestamp,
            "test_samples": TEST_SAMPLES,
            "rouge_scores": {
                "rouge1": float(rouge_scores['rouge1']),
                "rouge2": float(rouge_scores['rouge2']),
                "rougeL": float(rouge_scores['rougeL']),
                "rougeLsum": float(rouge_scores['rougeLsum']),
            },
            "generation_config": GENERATION_CONFIG,
        }
        
        json_path = f"{RESULTS_DIR}/{model_name}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result_detail, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {json_path}")
        
        all_results.append({
            "model_name": model_name,
            "timestamp": timestamp,
            "rouge1": float(rouge_scores['rouge1']),
            "rouge2": float(rouge_scores['rouge2']),
            "rougeL": float(rouge_scores['rougeL']),
            "rougeLsum": float(rouge_scores['rougeLsum']),
        })
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    csv_path = f"{RESULTS_DIR}/rouge_comparison_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["model_name", "timestamp", "rouge1", "rouge2", "rougeL", "rougeLsum"]
        )
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nComparison saved to: {csv_path}")
    print("\nROUGE Comparison:")
    print(f"{'Model':<30} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10}")
    print("-" * 60)
    for result in all_results:
        print(f"{result['model_name']:<30} {result['rouge1']:<10.4f} {result['rouge2']:<10.4f} {result['rougeL']:<10.4f}")

if __name__ == "__main__":
    main()
