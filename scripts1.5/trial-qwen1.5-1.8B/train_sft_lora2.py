import os
import time
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 设置随机种子保证可复现性
set_seed(42)

# =========================
# 1. 环境与基本配置
# =========================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = './models/Qwen1.5-1.8B'
output_dir = "./sft-tldr-lora-improved"

# 更加科学的超参数配置
config = {
    "train_sample": 50000,
    "eval_sample": 2000,
    "learning_rate": 1e-4,       # 对于 LoRA，1e-4 通常比 5e-5 收敛更快更稳
    "num_train_epochs": 3,
    "per_device_batch_size": 4,  # 1.8B 模型较小，可适当增加 BS
    "grad_accum": 4,             # 全局 Batch Size = 4 * 4 = 16
    "max_seq_length": 512,
    "lora_r": 32,                # 增加到 32 提升 1.8B 小模型的拟合能力
    "lora_alpha": 64,            # 通常为 r 的 2 倍
}

run = wandb.init(
    project="prml-sft-lora-improved",
    name=f"qwen1.5-1.8b-sft-{int(time.time())}",
    config=config,
)

# =========================
# 2. Tokenizer (修正 Pad Token 问题)
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True,
)

# Qwen1.5 默认没有 pad_token，使用 eos_token 填充
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# =========================
# 3. 数据集与过滤 (逻辑优化)
# =========================
dataset = load_dataset("CarperAI/openai_summarize_tldr")

def preprocess_and_filter(example):
    # 提取内容
    if "POST:" in example['prompt'] and "TL;DR:" in example['prompt']:
        post_start = example['prompt'].find("POST:") + 5
        tldr_start = example['prompt'].find("TL;DR:")
        post_content = example['prompt'][post_start:tldr_start].strip()
        # 使用TL;DR格式，与RM训练和PPO训练保持一致
        example['processed_prompt'] = f"{post_content}\nTL;DR:"
    else:
        example['processed_prompt'] = example['prompt']

    # 质量过滤条件
    prompt_len = len(example['processed_prompt'].split())
    label_len = len(example['label'].split())
    
    keep = (30 < prompt_len < 400 and 
            5 < label_len < 60 and 
            label_len / prompt_len < 0.6)
    return keep

print(f"原始训练集大小: {len(dataset['train'])}")
dataset = dataset.filter(preprocess_and_filter)
print(f"过滤后训练集大小: {len(dataset['train'])}")

# 抽样
dataset_small = {
    "train": dataset["train"].select(range(min(config["train_sample"], len(dataset["train"])))),
    "valid": dataset["valid"].select(range(min(config["eval_sample"], len(dataset["valid"])))),
    "test": dataset["test"].select(range(min(100, len(dataset["test"])))),
}

def formatting_prompts_func(example):
    texts = []
    for i in range(len(example["processed_prompt"])):
        # 确保 prompt 和 label 之间有清晰的界限
        text = f"{example['processed_prompt'][i]} {example['label'][i]}{tokenizer.eos_token}"
        texts.append(text)
    return texts

# 使用TL;DR作为response template，与数据格式保持一致
response_template = "TL;DR:"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

# =========================
# 4. 模型与 LoRA 配置 (扩展目标模块)
# =========================
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # Qwen1.5 推荐 bf16
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", # 包含 MLP 效果更好
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 5. 训练参数 (性能优化)
# =========================
training_args = TrainingArguments(
    output_dir=f"{output_dir}-final",
    per_device_train_batch_size=config["per_device_batch_size"],
    gradient_accumulation_steps=config["grad_accum"],
    gradient_checkpointing=False,         # 1.8B 模型显存足够，关闭可提速约 20-30%
    
    learning_rate=config["learning_rate"],
    num_train_epochs=config["num_train_epochs"],
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=400,
    save_total_limit=2,

    bf16=True,
    tf32=True,                            # 如果是 Ampere 架构（如 3090/A100）建议开启
    report_to=["wandb"],
    run_name=run.name,
    remove_unused_columns=False,          # 配合 SFTTrainer 使用
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_small["train"],
    eval_dataset=dataset_small["valid"],
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=config["max_seq_length"],
    packing=False,                        # 设为 False 才能让 DataCollator 准确 Mask 掉 Prompt
)

# =========================
# 6. 训练与测试
# =========================
try:
    print("🚀 Starting Training...")
    trainer.train()

    print("\n--- 推理测试 ---")
    def test_model(idx=0):
        item = dataset_small["test"][idx]
        prompt = item["processed_prompt"]
        gt = item["label"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 TL;DR: 之后的内容
        summary = decoded.split("TL;DR:")[-1].strip()

        print(f"\n[Test Item {idx}]")
        print(f"Input: {prompt[:150]}...")
        print(f"Ground Truth: {gt}")
        print(f"Model Generated: {summary}")

    for i in [0, 1]: test_model(i)

    # 保存
    trainer.save_model(f"{output_dir}/lora_adapter_final")
    print(f"✅ Saved to: {output_dir}/lora_adapter_final")

except Exception as e:
    print(f"❌ Error: {e}")
finally:
    wandb.finish()