import os
import time
import torch
import torch.nn as nn
import wandb
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
)
from datasets import load_dataset
from trl import RewardTrainer

# --- 1. 配置 ---
model_id = "./models/sft-tldr/final_checkpoint"
output_dir = "./models/rm-tldr-margin-final"
DEBUG = False 

TRAIN_SAMPLE = 20000 
EVAL_SAMPLE = 1000

# --- 0. 初始化 Weights & Biases ---
wandb.init(
    project="qwen-rm-optimized",
    name=f"qwen-1.8b-margin-rm-final",
    config={
        "learning_rate": 1e-5,
        "margin": 1.0,
        "train_sample": TRAIN_SAMPLE,
    },
)

# --- 2. 加载模型与分词器 ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right" 
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False 

# --- 3. 自定义 Trainer：修复 Eval IndexError 和 Accuracy 问题 ---
class MarginRewardTrainer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        margin = 1.0
        # 计算奖励
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )[0]
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )[0]

        # Margin Loss
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - margin).mean()

        if return_outputs:
            # 供训练监控使用
            stacked_logits = torch.cat([rewards_chosen, rewards_rejected], dim=1).detach()
            return loss, {"logits": stacked_logits}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        核心修正：重写预测步。确保在评估循环中，返回的 logits 形状为 (batch_size, 2)。
        """
        device = model.device
        with torch.no_grad():
            # 准备输入
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            
            # 计算 Loss
            loss = self.compute_loss(model, inputs, return_outputs=False)
            if prediction_loss_only:
                return (loss, None, None)

            # 显式计算两个分数并拼接
            r_chosen = model(input_ids=inputs["input_ids_chosen"], attention_mask=inputs["attention_mask_chosen"])[0]
            r_rejected = model(input_ids=inputs["input_ids_rejected"], attention_mask=inputs["attention_mask_rejected"])[0]
            
            # 形状必须为 (batch_size, 2)
            logits = torch.cat([r_chosen, r_rejected], dim=1).detach()
            # 伪造 labels，TRL 内部计算 accuracy 时并不使用 label_ids
            labels = torch.zeros(logits.shape[0], device=device)

        return (loss, logits, labels)

# --- 4. 数据处理 ---
dataset = load_dataset("openai/summarize_from_feedback", "comparisons")

if DEBUG:
    train_dataset_raw = dataset["train"].select(range(256))
    eval_dataset_raw = dataset["validation"].select(range(128))
else:
    train_dataset_raw = dataset["train"].shuffle(seed=42).select(range(min(TRAIN_SAMPLE, len(dataset["train"]))))
    eval_dataset_raw = dataset["validation"].shuffle(seed=42).select(range(min(EVAL_SAMPLE, len(dataset["validation"]))))

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [], "attention_mask_chosen": [],
        "input_ids_rejected": [], "attention_mask_rejected": [],
    }
    for prompt, summaries, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
        if summaries[0]['text'] == summaries[1]['text']:
            continue
        p = f"Post: {prompt['post']}\nTL;DR:"
        c = f" {summaries[choice]['text']}{tokenizer.eos_token}"
        r = f" {summaries[1 - choice]['text']}{tokenizer.eos_token}"

        t_chosen = tokenizer(p + c, max_length=1024, truncation=True)
        t_rejected = tokenizer(p + r, max_length=1024, truncation=True)

        new_examples["input_ids_chosen"].append(t_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(t_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(t_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(t_rejected["attention_mask"])
    return new_examples

train_dataset = train_dataset_raw.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
eval_dataset = eval_dataset_raw.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)

# --- 5. Data Collator ---
class RewardDataCollator:
    def __call__(self, features):
        batch = {}
        for k in ["chosen", "rejected"]:
            inputs = [{"input_ids": f[f"input_ids_{k}"], "attention_mask": f[f"attention_mask_{k}"]} for f in features]
            padded = tokenizer.pad(inputs, padding=True, return_tensors="pt")
            batch[f"input_ids_{k}"] = padded["input_ids"]
            batch[f"attention_mask_{k}"] = padded["attention_mask"]
        return batch

# --- 6. 训练参数设置 ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,   
    gradient_accumulation_steps=32,
    per_device_eval_batch_size=4,    # 调小以防止 Eval 时 OOM
    
    learning_rate=1e-5,              
    lr_scheduler_type="cosine",      
    warmup_ratio=0.1,                
    weight_decay=0.1,                
    max_grad_norm=1.0,               
    
    num_train_epochs=1,              
    bf16=True,                      
    gradient_checkpointing=True,     
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    logging_steps=15,
    evaluation_strategy="steps",
    eval_steps=50,                   
    save_strategy="steps",
    save_steps=50,
    
    load_best_model_at_end=True,     
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=1,              
    remove_unused_columns=False,
    report_to=["wandb"],            
)

# --- 7. 启动训练 ---
trainer = MarginRewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=RewardDataCollator(),
)

print(f"🚀 任务启动。训练集大小: {len(train_dataset)}。修复了 Eval 索引问题。")
trainer.train()

# --- 8. 保存 ---
trainer.save_model(f"{output_dir}/final_rm")
tokenizer.save_pretrained(f"{output_dir}/final_rm")
print(f"✅ 完成！模型保存在 {output_dir}/final_rm")