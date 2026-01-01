import os
import time
import torch
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
)
from datasets import load_dataset
from trl import RewardTrainer

# --- 1. 配置 ---
model_id = "./models/sft-tldr/final_checkpoint"  # SFT 模型路径
output_dir = "./models/rm-tldr"
DEBUG = False 

TRAIN_SAMPLE = 20000 # 原为8000
EVAL_SAMPLE = 1000

# --- 0. 初始化 Weights & Biases ---
run = wandb.init(
    project="qwen-rm-optimized",
    name=f"qwen-1.8b-rm-{int(time.time())}",
    config={
        "model_id": model_id,
        "train_sample": TRAIN_SAMPLE,
        "eval_sample": EVAL_SAMPLE,
        "learning_rate": 2e-5,
        "weight_decay": 0.1,
        "global_batch": 64,
        "max_seq_length": 1024,
    },
)

# --- 2. 加载模型与分词器 ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right" # 必须右填充
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False 

# --- 3. 数据处理 ---
dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

if DEBUG:
    train_dataset_raw = dataset["train"].select(range(256))
    eval_dataset_raw = dataset["validation"].select(range(128))
else:
    # 训练集和验证集都进行 shuffle，确保分布均匀
    train_dataset_raw = dataset["train"].shuffle(seed=42).select(range(min(TRAIN_SAMPLE, len(dataset["train"]))))
    eval_dataset_raw = dataset["validation"].shuffle(seed=42).select(range(min(EVAL_SAMPLE, len(dataset["validation"]))))

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, summaries, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
        # 更加清晰的 Prompt 构造
        # p = f"Post: {prompt['post']}\n\nTL;DR:"
        p = f"Post: {prompt['post']}\nTL;DR:"
        c = f" {summaries[choice]['text']}{tokenizer.eos_token}"
        r = f" {summaries[1 - choice]['text']}{tokenizer.eos_token}"

        # 适当调整 max_length，确保摘要不被截断过多
        tokenized_chosen = tokenizer(p + c, max_length=1024, truncation=True)
        tokenized_rejected = tokenizer(p + r, max_length=1024, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

train_dataset = train_dataset_raw.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
eval_dataset = eval_dataset_raw.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)

# --- 4. Data Collator ---
class RewardDataCollator:
    def __call__(self, features):
        batch = {}
        for k in ["chosen", "rejected"]:
            inputs = [{"input_ids": f[f"input_ids_{k}"], "attention_mask": f[f"attention_mask_{k}"]} for f in features]
            padded = tokenizer.pad(inputs, padding=True, return_tensors="pt")
            batch[f"input_ids_{k}"] = padded["input_ids"]
            batch[f"attention_mask_{k}"] = padded["attention_mask"]
        return batch

# --- 5. 训练参数设置 (针对 4090 深度优化) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,   
    gradient_accumulation_steps=32,  # 总 Batch Size = 64
    
    # 策略调整：针对 1.8B 模型提高学习率和正则化
    learning_rate=2e-5,              
    lr_scheduler_type="linear",      # 线性衰减在小模型上更稳定
    warmup_steps=100,                # 固定的热身步数
    weight_decay=0.1,                # 强化正则化，防止 1.8B 坍缩
    max_grad_norm=1.0,               
    
    num_train_epochs=1,              
    bf16=True,                      
    gradient_checkpointing=True,     
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # 强化监控：每 20 步验证一次
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=40,                   
    save_strategy="steps",
    save_steps=40,
    
    # 自动保存并加载最佳模型
    load_best_model_at_end=True,     
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,              # 节省磁盘空间
    
    remove_unused_columns=False,
    report_to=["wandb"],            
)

# --- 6. 启动训练 ---
trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=RewardDataCollator(),
)



print("🚀 开始优化后的 Reward Model 训练...")
trainer.train()

# --- 7. 保存结果 ---
trainer.save_model(f"{output_dir}/final_rm")
tokenizer.save_pretrained(f"{output_dir}/final_rm")
print(f"✅ 最佳模型已保存至 {output_dir}/final_rm")