import os
import time
import torch
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from datasets import load_from_disk
from trl import RewardTrainer, RewardConfig

# =========================================================
# 0. 基础配置
# =========================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 模型路径（本地）----
# 当前工作目录是 /workspace/pj-RL/trial2-qwen3-1.7B，模型目录在上一级 pj-RL/trial2-sft-qwen3-1.7b
MODEL_ID = "../trial2-sft-qwen3-1.7b/final_checkpoint"   # ← 你的 SFT checkpoint
OUTPUT_DIR = "./trial2-rm-qwen3-1.7b"

# ---- 数据路径（本地）----
DATASET_PATH = "/workspace/pj-RL/datasets/summarize_from_feedback"

# ---- 训练规模 ----
TRAIN_SAMPLE = 20000
EVAL_SAMPLE = 1000
TEST_SAMPLE = 1000
NUM_EPOCHS = 1

DEBUG = False

# =========================================================
# 1. W&B
# =========================================================
run = wandb.init(
    project="qwen3-rm-tldr",
    name=f"qwen3-1.7b-rm-{int(time.time())}",
    config={
        "model": "qwen3-1.7b",
        "train_samples": TRAIN_SAMPLE,
        "eval_samples": EVAL_SAMPLE,
        "epochs": NUM_EPOCHS,
        "lr": 2e-5,
        "weight_decay": 0.1,
        "global_batch_size": 64,
        "max_length": 1024,
    },
)

# =========================================================
# 2. Tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    fix_mistral_regex=True,
)

tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# 3. Reward Model
# =========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# =========================================================
# 4. 加载本地数据集
# =========================================================
dataset = load_from_disk(DATASET_PATH)

if DEBUG:
    train_raw = dataset["train"].select(range(256))
    eval_raw = dataset["validation"].select(range(128))
else:
    train_raw = (
        dataset["train"]
        .shuffle(seed=42)
        .select(range(min(TRAIN_SAMPLE, len(dataset["train"]))))
    )
    eval_raw = (
        dataset["validation"]
        .shuffle(seed=42)
        .select(range(min(EVAL_SAMPLE, len(dataset["validation"]))))
    )
    # # 如果有测试集，用测试集；否则复用验证集
    # if "test" in dataset:
    #     test_raw = (
    #         dataset["test"]
    #         .shuffle(seed=42)
    #         .select(range(min(TEST_SAMPLE, len(dataset["test"]))))
    #     )
    # else:
    #     test_raw = eval_raw

# =========================================================
# 5. Prompt + Tokenize（关键：统一格式）
# =========================================================
def preprocess_function(examples):
    chosen = []
    rejected = []

    for info, summaries, choice in zip(
        examples["info"],
        examples["summaries"],
        examples["choice"],
    ):
        # -------- Prompt（与你 SFT / PPO 完全一致）--------
        prompt = f"{info['post']}\n\nTL;DR:"

        chosen_text = prompt + summaries[choice]["text"] + tokenizer.eos_token
        rejected_text = prompt + summaries[1 - choice]["text"] + tokenizer.eos_token

        chosen.append(chosen_text)
        rejected.append(rejected_text)

    return {
        "chosen": chosen,
        "rejected": rejected,
    }

train_dataset = train_raw.map(
    preprocess_function,
    batched=True,
    remove_columns=train_raw.column_names,
)
eval_dataset = eval_raw.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_raw.column_names,
)
# test_dataset = test_raw.map(
#     preprocess_function,
#     batched=True,
#     remove_columns=test_raw.column_names,
# )

# =========================================================
# 5.1 打印一条喂给模型的拼接文本格式（不训练时查看）
# =========================================================
sample_info = train_raw[0]["info"]
sample_summaries = train_raw[0]["summaries"]
sample_choice = train_raw[0]["choice"]

sample_prompt = f"{sample_info['post']}\n\nTL;DR:"
sample_chosen_text = sample_prompt + sample_summaries[sample_choice]["text"] + tokenizer.eos_token
sample_rejected_text = sample_prompt + sample_summaries[1 - sample_choice]["text"] + tokenizer.eos_token

print("--- Sample Preview (RM input text) ---")
print("Prompt:\n", sample_prompt)
print("Chosen (label=preferred):\n", sample_chosen_text)
print("Rejected:\n", sample_rejected_text)
print("-------------------------------------")

# =========================================================
# 6. Data Collator
# =========================================================
class RewardDataCollator:
    def __call__(self, features):
        raise NotImplementedError("Use TRL's default DataCollatorForPreference.")

# =========================================================
# 7. TrainingArguments
# =========================================================
training_args = RewardConfig(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,  # 2 * 32 = 64

    num_train_epochs=NUM_EPOCHS,

    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_steps=100,

    weight_decay=0.1,
    max_grad_norm=1.0,

    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    save_total_limit=2,
    remove_unused_columns=False,
    report_to=["wandb"],
)

# =========================================================
# 8. RewardTrainer
# =========================================================
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# =========================================================
# 9. Train
# =========================================================
print("🚀 Starting Reward Model training (Qwen3-1.7B)...")
trainer.train()

# # =========================================================
# # 9.1 Evaluate on validation and test
# # =========================================================
# print("\n📊 Evaluating on validation set...")
# val_metrics = trainer.evaluate()
# print("Validation metrics:", val_metrics)
# try:
#     wandb.log({f"val_{k}": v for k, v in val_metrics.items()})
# except Exception:
#     pass

# print("\n🧪 Testing on test set...")
# test_metrics = trainer.evaluate(eval_dataset=test_dataset)
# print("Test metrics:", test_metrics)
# try:
#     wandb.log({f"test_{k}": v for k, v in test_metrics.items()})
# except Exception:
#     pass

# =========================================================
# 10. Save
# =========================================================
final_path = f"{OUTPUT_DIR}/final_rm"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

print(f"✅ Reward Model saved to: {final_path}")
