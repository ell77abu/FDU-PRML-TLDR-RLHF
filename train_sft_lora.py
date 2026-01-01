import os
import time
import torch
import wandb

# =========================
# 1. 环境与基本配置
# =========================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = './models/Qwen1.5-1.8B'
output_dir = "./sft-tldr-lora"

run = wandb.init(
    project="prml-sft-lora",
    name=f"qwen1.5-1.8b-lora-{int(time.time())}",
    config={
        "train_sample": 10000,
        "eval_sample": 750,
        "learning_rate": 1e-4,
        "num_train_epochs": 1,
        "batch_size": 1,
        "grad_accum": 8,
        "max_seq_length": 768,
        "lora_r": 8,
    },
)

# =========================
# 2. Tokenizer
# =========================
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =========================
# 3. 数据集
# =========================
from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM

dataset = load_dataset("CarperAI/openai_summarize_tldr")

train_sample = 10000
eval_sample = 750
test_sample = 750

dataset_small = {
    "train": dataset["train"].select(range(train_sample)),
    "valid": dataset["valid"].select(range(eval_sample)),
    "test": dataset["test"].select(range(test_sample)),
}

def formatting_prompts_func(example):
    texts = []
    for i in range(len(example["prompt"])):
        text = f"{example['prompt'][i]} {example['label'][i]}{tokenizer.eos_token}"
        texts.append(text)
    return texts

response_template = "TL;DR:"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# =========================
# 4. 加载模型 + 注入 LoRA
# =========================
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # LoRA rank（4090 推荐 8 或 16）
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.to(device)

# =========================
# 5. 训练参数
# =========================
from transformers import TrainingArguments
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,          # LoRA 推荐较大 LR
    num_train_epochs=1,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    report_to=["wandb"],
    run_name=run.name,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_small["train"],
    eval_dataset=dataset_small["valid"],
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=768,
)

# =========================
# 6. 开始训练
# =========================
print("🚀 Starting LoRA SFT training...")
trainer.train()

# =========================
# 7. 推理测试
# =========================
print("\n--- 推理测试 ---")

def test_model(idx=0):
    prompt = dataset_small["test"][idx]["prompt"]
    gt = dataset_small["test"][idx]["label"]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = decoded.split("TL;DR:")[-1].strip()

    print("\n[Prompt]")
    print(prompt[-200:])
    print("\n[Human]")
    print(gt)
    print("\n[Model]")
    print(summary)

test_model(7)
test_model(5)

# =========================
# 8. 保存 LoRA Adapter
# =========================
trainer.save_model(f"{output_dir}/lora_adapter")
wandb.finish()
