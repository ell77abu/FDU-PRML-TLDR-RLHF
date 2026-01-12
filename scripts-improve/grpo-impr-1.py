import os
import torch
import wandb
import logging
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# ================= 0. 环境与路径配置 =================
os.environ["WANDB_CACHE_DIR"] = "/workspace/wandb_cache"
os.environ["WANDB_DIR"] = "/workspace/wandb_logs"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("trl").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

SFT_MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
# 指向你的多维度 Axis-RM 路径
AXIS_RM_MODEL_PATH = "/workspace/pj-RL/experiments3/qwen3-rm-multi-axis/final_axis_rm" 
DATASET_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"

# ================= 1. Axis-RM 配置 (归一化参数) =================
# 根据 axis_rm_states 文件中记录填写
AXIS_STATS = {
    "means": np.array([0.6258134245872498, 0.8554375171661377, 0.6518884301185608, 0.9134042859077454]),   
    "stds": np.array([0.14399273693561554, 0.06737399101257324, 0.15281224250793457, 0.05094359815120697]),   
    "weights": np.array([0.4, 0.1, 0.4, 0.1]) 
}

# ================= 2. 加载模型与 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# 加载多维度 Reward Model
# 注意：num_labels 必须与你训练 Axis-RM 时一致（4）
axis_rm = AutoModelForSequenceClassification.from_pretrained(
    AXIS_RM_MODEL_PATH,
    num_labels=4,
    torch_dtype=torch.bfloat16,
    device_map={"": 0} # 放在 0 号显卡
).eval()

# ================= 3. 定义多维度奖励函数 =================
def axis_reward_func(prompts, completions, **kwargs):
    """
    1. 拼接文本
    2. 获取 4 维 Logits
    3. Z-Score 归一化
    4. 加权求和
    """
    texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
    
    # 对输入进行编码
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    ).to(axis_rm.device)

    with torch.no_grad():
        # 获取 raw logits (shape: [batch_size, 4])
        logits = axis_rm(**inputs).logits.cpu().float().numpy()

    # 归一化处理: (x - mu) / sigma
    norm_logits = (logits - AXIS_STATS["means"]) / (AXIS_STATS["stds"] + 1e-8)
    
    # 加权求和: dot product
    final_scores = np.dot(norm_logits, AXIS_STATS["weights"])
    
    return final_scores.tolist()

# ================= 4. 数据准备 =================
raw_dataset = load_from_disk(DATASET_PATH)["train"]

def format_reward_dataset(example):
    return {"prompt": f"{example['info']['post']}\n\nTL;DR:"}

train_dataset = raw_dataset.shuffle(seed=42).select(range(5000)).map(format_reward_dataset)

# ================= 5. GRPO 训练参数 =================
training_args = GRPOConfig(
    output_dir="./qwen3-grpo-axis-rm",
    learning_rate=2e-6,           
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
    logging_steps=1,
    save_steps=150,                 
    save_total_limit=2,             
    report_to="wandb",
    log_completions=True,           
)

# ================= 6. 初始化与训练 =================
wandb.init(
    project="qwen3-grpo-tldr",
    name="grpo-axis-rm-impr-1",
    config={
        **training_args.to_dict(),
        "axis_weights": AXIS_STATS["weights"].tolist()
    }
)

trainer = GRPOTrainer(
    model=SFT_MODEL_PATH,
    reward_funcs=[axis_reward_func], # 使用新的多维度奖励函数
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
trainer.save_model("./qwen3-grpo-axis-impr-1")
wandb.finish()