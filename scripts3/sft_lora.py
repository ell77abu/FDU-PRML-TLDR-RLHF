import os
import time
import torch
import wandb

# ===============================
# 1. 环境与模型配置
# ===============================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "/workspace/pj-RL/models/Qwen3-1.7B"   # base model
output_dir = "./trial2-sft-qwen3-1.7b-lora"

# ===============================
# 2. wandb
# ===============================
run = wandb.init(
    project="prml-sft",
    name=f"Qwen3-1.7B-SFT-LoRA-POSTONLY-{int(time.time())}",
    config={
        "model": model_id,
        "train_samples": 10000,
        "eval_samples": 750,
        "epochs": 2,
        "lr": 5e-5,              # LoRA 下略大一点更好
        "batch_size": 4,
        "grad_accum": 16,
        "max_seq_len": 1024,
        "lora_r": 16,
    },
)

# ===============================
# 3. Tokenizer
# ===============================
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

# ===============================
# 4. 数据集加载
# ===============================
from datasets import load_from_disk

dataset = load_from_disk("/workspace/pj-RL/datasets/openai_summarize_tldr")

train_n = 10000
eval_n = 750
test_n = 750

dataset_small = {
    "train": dataset["train"].shuffle(seed=42).select(range(train_n)),
    "valid": dataset["valid"].select(range(eval_n)),
    "test": dataset["test"].select(range(test_n)),
}

# ===============================
# 5. Prompt 预处理（POST-only）
# ===============================
def extract_post_only(prompt: str) -> str:
    if "POST:" in prompt:
        prompt = prompt.split("POST:", 1)[1]

    if "TL;DR:" in prompt:
        post, _ = prompt.split("TL;DR:", 1)
        prompt = post.strip() + "\n\nTL;DR:"
    else:
        prompt = prompt.strip() + "\n\nTL;DR:"

    return prompt


def build_prompt_completion(example):
    clean_prompt = extract_post_only(example["prompt"])
    completion = " " + example["label"].strip() + tokenizer.eos_token
    return {
        "prompt": clean_prompt,
        "completion": completion,
    }


dataset_for_sft = {
    split: dataset_small[split].map(build_prompt_completion)
    for split in ["train", "valid", "test"]
}

# ===============================
# 6. 模型加载 + LoRA 注入
# ===============================
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
)

model.config.pad_token_id = tokenizer.pad_token_id

# ---- LoRA 配置 ----
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================
# 7. SFT 训练参数
# ===============================
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir=output_dir,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,

    learning_rate=5e-5,          # LoRA 推荐
    num_train_epochs=2,

    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    logging_steps=5,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",

    bf16=True,
    fp16=False,

    completion_only_loss=True,
    max_length=1024,

    report_to=["wandb"],
    run_name=run.name,
)

# ===============================
# 8. SFT Trainer
# ===============================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset_for_sft["train"],
    eval_dataset=dataset_for_sft["valid"],
)

# ===============================
# 9. 开始训练
# ===============================
print("\n--- Sample Check ---")
sample = dataset_for_sft["train"][0]
print("Prompt:\n", sample["prompt"])
print("Completion:\n", sample["completion"])
print("--------------------")

print("🚀 Start LoRA-SFT (Qwen3-1.7B, POST-only)")
trainer.train()

# ===============================
# 10. 推理测试（LoRA 模型）
# ===============================
def test_model(idx=0):
    raw_prompt = dataset_small["test"][idx]["prompt"]
    gold = dataset_small["test"][idx]["label"]

    prompt = extract_post_only(raw_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = decoded.split("TL;DR")[-1].strip()

    print("\n==============================")
    print("[POST tail]")
    print(prompt[-200:])
    print("\n[Human TL;DR]")
    print(gold)
    print("\n[Model TL;DR]")
    print(summary)

    wandb.log({
        "sample_idx": idx,
        "human_summary": gold,
        "model_summary": summary,
    })


test_model(5)
test_model(7)

# ===============================
# 11. 保存 LoRA Adapter
# ===============================
trainer.save_model(f"{output_dir}/final_checkpoint")
wandb.finish()
