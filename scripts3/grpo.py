import os
import torch
import wandb
import logging
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from transformers import pipeline

os.environ["WANDB_CACHE_DIR"] = "/workspace/wandb_cache"
os.environ["WANDB_DIR"] = "/workspace/wandb_logs"

# ================= 0. 日志级别配置 =================
# 静默控制台的冗长输出，但保留 WandB 的数据收集
logging.getLogger("trl").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ================= 1. 环境与路径配置 =================
SFT_MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
RM_MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
DATASET_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================= 2. 数据准备 =================
raw_dataset = load_from_disk(DATASET_PATH)["train"]

def format_reward_dataset(example):
    # 严格匹配 RM 训练时的 prompt 格式
    return {"prompt": f"{example['info']['post']}\n\nTL;DR:"}

# 建议直接开启 2000 条测试，以观察 Reward 稳定上升趋势
train_dataset = raw_dataset.shuffle(seed=42).select(range(5000)).map(format_reward_dataset)

# ================= 3. 奖励函数定义 (对齐 RM 训练分布) =================
reward_pipe = pipeline(
    "sentiment-analysis",
    model=RM_MODEL_PATH,
    device=0, 
    model_kwargs={"torch_dtype": torch.bfloat16}
)

def get_reward_score(prompts, completions, **kwargs):
    """
    严格对齐 RM 训练格式: prompt + completion + eos_token
    """
    eos_token = tokenizer.eos_token
    
    # 这里的关键是加上 eos_token
    texts = [p + c + eos_token for p, c in zip(prompts, completions)]
    
    pipe_outputs = reward_pipe(
        texts, 
        batch_size=len(texts), 
        truncation=True, 
        max_length=1024
    )
    return [output["score"] for output in pipe_outputs]

# ================= 4. GRPO 训练参数 (正式 10k 冲刺版) =================
training_args = GRPOConfig(
    output_dir="./qwen3-grpo-final-10k",
    
    # --- 学习率策略 ---
    learning_rate=2e-6,           
    lr_scheduler_type="cosine",
    # 10000 / (1 * 16) = 625 steps
    # 建议 warmup 设为总步数的 5-8%，即 30-50 步
    warmup_steps=40,                
    weight_decay=0.01,
    
    # --- 显存与计算效率 (4090 优化) ---
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=16, 
    num_generations=8,              # 保持 8 能够显著平滑 Reward 曲线，利于观察趋势
    max_prompt_length=512,
    max_completion_length=64,      # 必须保持 128，给模型留出完整收尾空间
    
    # --- KL 监控与惩罚 (核心指标) ---
    beta=0.1,                       # 显式开启 beta 确保 WandB 记录 objective/kl
    
    # --- 精度与加速 ---
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",           
    
    # --- 日志与保存 (针对 2 天限时优化) ---
    num_train_epochs=1,
    logging_steps=1,
    save_steps=150,                 # 略微调大，减少频繁保存模型导致的 IO 阻塞
    save_total_limit=2,             # 只保留最新的两个，节省磁盘空间
    report_to="wandb",
    log_completions=True,           
)

# ================= 5. 初始化与训练 =================

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

wandb.init(
    project="qwen3-grpo-tldr",
    name="grpo-2k-g8-beta0.1",
    config=training_args
)

trainer = GRPOTrainer(
    model=SFT_MODEL_PATH,
    reward_funcs=[get_reward_score],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ),
)

trainer.train()
trainer.save_model("./qwen3-grpo-10k")
wandb.finish()