import os
import torch
import wandb
import logging
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ================= 0. 环境与路径配置 =================
os.environ["WANDB_CACHE_DIR"] = "/workspace/wandb_cache"
os.environ["WANDB_DIR"] = "/workspace/wandb_logs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 基础模型与两个 RM 的路径
SFT_MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
BASE_RM_PATH = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
AXIS_RM_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-multi-axis/final_axis_rm"
DATASET_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"

# ================= 1. 归一化统计与权重配置 =================
# 注意：你需要根据 axis_rm_states 和基础 RM 的训练记录填写以下值
STATS = {
    "base": {"mean": 0.19737262239456177, "std": 1.103783268960716, "weight": 0.5}, # 基础 RM 权重
    "overall": {"mean": 0.6258134245872498, "std": 0.14399273693561554, "weight": 0.2},
    "accuracy": {"mean": 0.8554375171661377, "std": 0.06737399101257324, "weight": 0.05}, 
    "coverage": {"mean": 0.6518884301185608, "std": 0.15281224250793457, "weight": 0.2},
    "coherence": {"mean": 0.9134042859077454, "std": 0.05094359815120697, "weight": 0.05},
    
}

# 组合权重列表 (1个基础 + 4个Axis)
COMBINED_WEIGHTS = np.array([
    STATS["base"]["weight"],
    STATS["accuracy"]["weight"],
    STATS["coverage"]["weight"],
    STATS["coherence"]["weight"],
    STATS["overall"]["weight"]
])

# ================= 2. 加载 Tokenizer 与模型 =================
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, fix_mistral_regex=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 加载 基础 RM (1维度)
base_rm = AutoModelForSequenceClassification.from_pretrained(
    BASE_RM_PATH, num_labels=1, torch_dtype=torch.bfloat16, 
    device_map={"": 0}, local_files_only=True
).eval()

# 加载 Axis RM (4维度)
axis_rm = AutoModelForSequenceClassification.from_pretrained(
    AXIS_RM_PATH, num_labels=4, torch_dtype=torch.bfloat16, 
    device_map={"": 0}, local_files_only=True
).eval()

# ================= 3. 混合奖励函数 =================
def hybrid_reward_func(prompts, completions, **kwargs):
    """
    计算混合奖励：基础 RM (1D) + Axis RM (4D)
    """
    texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
    
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, 
        truncation=True, max_length=1024
    ).to("cuda")

    with torch.no_grad():
        # 1. 基础 RM 推理
        base_logits = base_rm(**inputs).logits.cpu().float().numpy() # [batch, 1]
        
        # 2. Axis RM 推理
        axis_logits = axis_rm(**inputs).logits.cpu().float().numpy() # [batch, 4]
    
    # 合并所有维度 [batch, 5]
    all_logits = np.concatenate([base_logits, axis_logits], axis=1)
    
    # 提取所有均值和标准差进行向量化归一化
    means = np.array([STATS[k]["mean"] for k in ["base", "accuracy", "coverage", "coherence", "overall"]])
    stds = np.array([STATS[k]["std"] for k in ["base", "accuracy", "coverage", "coherence", "overall"]])
    
    # 执行 Z-Score: (x - mu) / sigma
    norm_logits = (all_logits - means) / (stds + 1e-8)
    
    # 加权求和
    final_scores = np.dot(norm_logits, COMBINED_WEIGHTS)
    
    return final_scores.tolist()

# ================= 4. 训练准备与配置 =================
raw_dataset = load_from_disk(DATASET_PATH)["train"]
train_dataset = raw_dataset.shuffle(seed=42).select(range(5000)).map(
    lambda x: {"prompt": f"{x['info']['post']}\n\nTL;DR:"}
)

training_args = GRPOConfig(
    output_dir="./qwen3-grpo-hybrid-rm",
    learning_rate=1e-6,             # 多个 RM 信号更复杂，建议降低学习率
    lr_scheduler_type="cosine",
    warmup_steps=40,                
    weight_decay=0.01,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16, 
    num_generations=8,              
    max_prompt_length=512,
    max_completion_length=60,      
    beta=0.1,                       
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",    
    num_train_epochs=1,
    logging_steps=10,
    save_steps=150,                 
    save_total_limit=2,       
    report_to="wandb",
    log_completions=True,
)

# ================= 5. 启动训练 =================
wandb.init(project="qwen3-grpo-tldr", name="hybrid-rm-weighted-norm")

trainer = GRPOTrainer(
    model=SFT_MODEL_PATH,
    reward_funcs=[hybrid_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)

trainer.train()
trainer.save_model("./qwen3-grpo-hybrid-impr-2")