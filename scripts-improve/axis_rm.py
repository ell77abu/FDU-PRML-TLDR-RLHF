import os
import torch
import torch.nn as nn
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import wandb

# ===============================
# 0. 配置
# ===============================
MODEL_ID = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
OUTPUT_DIR = "/workspace/pj-RL/experiments3/qwen3-rm-multi-axis"
TARGET_AXES = ["overall", "accuracy", "coverage", "coherence"]

# 初始化 wandb
wandb.init(
    project="axis-rm-training",
    name="qwen3-rm-multi-axis-full",
    config={
        "model_id": MODEL_ID,
        "output_dir": OUTPUT_DIR,
        "target_axes": TARGET_AXES,
        "warmup_steps": 50,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 16,
    }
)

# ===============================
# 1. 加载与预处理
# ===============================
raw_ds = load_from_disk("/workspace/pj-RL/datasets/summarize_axis")["validation"]
ds_split = raw_ds.train_test_split(test_size=0.1, seed=42)
split_dataset = DatasetDict({"train": ds_split["train"], "validation": ds_split["test"]})

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def preprocess_fn(examples):
    texts, labels = [], []
    for info, summary in zip(examples["info"], examples["summary"]):
        prompt = f"{info['post']}\n\nTL;DR:"
        texts.append(prompt + summary["text"] + tokenizer.eos_token)
        
        axes_dict = summary.get("axes", {})
        sample_labels = [(float(axes_dict.get(a, 4) or 4) - 1.0) / 6.0 for a in TARGET_AXES]
        labels.append(sample_labels)

    tokenized = tokenizer(texts, truncation=True, max_length=1024)
    tokenized["labels"] = labels
    return tokenized

tokenized_ds = split_dataset.map(preprocess_fn, batched=True, remove_columns=raw_ds.column_names)

# 强制检查标签
print(f"DEBUG - 样本标签: {tokenized_ds['train'][0]['labels']}")

# ===============================
# 2. 模型定义与重锤初始化
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(TARGET_AXES), problem_type="regression",
    torch_dtype=torch.bfloat16, trust_remote_code=True
)

# 【核心改进 1】全零初始化。让模型起步时预测值全为 0，从而使初始 MSE 保持在 0.5 以下。
if hasattr(model, "score"):
    nn.init.zeros_(model.score.weight)
    print(">>> 已执行全零初始化回归头")

# 【核心改进 2】第一阶段：冻结主干。先让回归头学会处理巨大的 Hidden States。
for name, param in model.named_parameters():
    if "score" not in name:
        param.requires_grad = False

class SafeRegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # 【核心改进 3】使用 HuberLoss。当误差极大时，它表现为线性（不像 MSE 是平方），防止梯度爆炸。
        loss_fct = nn.HuberLoss(delta=1.0)
        loss = loss_fct(outputs.logits, labels.to(outputs.logits.dtype))
        return (loss, outputs) if return_outputs else loss

# ===============================
# 3. 训练阶段 1：热身回归头 (少量 Steps)
# ===============================
warmup_args = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/warmup",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-4, # 仅训练 Head，可以用大点的学习率
    max_steps=50,       # 只跑 50 步快速对齐数值
    bf16=True,
    max_grad_norm=0.5,
    report_to="none"
)

trainer = SafeRegressionTrainer(
    model=model, args=warmup_args,
    train_dataset=tokenized_ds["train"],
    data_collator=DataCollatorWithPadding(tokenizer)
)

print(">>> 阶段 1：正在稳定回归头数值...")
trainer.train()

# ===============================
# 4. 训练阶段 2：全量微调 (正式训练)
# ===============================
print(">>> 阶段 2：解锁主干，开始全量微调...")
for param in model.parameters():
    param.requires_grad = True

main_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1e-5, # 全量微调，学习率一定要小
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    max_grad_norm=0.3,
    eval_strategy="steps", eval_steps=40,  # 每40步进行eval
    save_strategy="steps", save_steps=40,  # 每40步保存模型
    load_best_model_at_end=True,
    report_to="wandb",  # 启用wandb记录
    logging_steps=20,  # 每20步记录训练loss
)

trainer = SafeRegressionTrainer(
    model=model, args=main_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    data_collator=DataCollatorWithPadding(tokenizer)
)

trainer.train()

# 保存
final_path = f"{OUTPUT_DIR}/final_axis_rm"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)