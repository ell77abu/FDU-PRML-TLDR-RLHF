import os
import time
import torch
import wandb

# ===============================
# 1. 环境与模型配置
# ===============================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "/workspace/pj-RL/models/Qwen3-1.7B"   # ⚠️ base model
output_dir = "./trial2-sft-qwen3-1.7b"

# ===============================
# 2. wandb（可选）
# ===============================
run = wandb.init(
    project="prml-sft",
    name=f"Qwen3-1.7B-SFT-POSTONLY-{int(time.time())}",
    config={
        "model": model_id,
        "train_samples": 10000,
        "eval_samples": 750,
        "epochs": 2,
        "lr": 1e-5,
        "batch_size": 4,
        "grad_accum": 16,
        "max_seq_len": 1024,
    },
)

# ===============================
# 3. Tokenizer
# ===============================
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True,
)

# Qwen3 必须显式处理 pad_token
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
# 5. Prompt 预处理（核心）
# ===============================
def extract_post_only(prompt: str) -> str:
    """
    输入：
        SUBREDDIT: ...
        TITLE: ...
        POST: xxx
        TL;DR:

    输出：
        xxx
        TL;DR:
    """
    # 去掉 SUBREDDIT / TITLE
    if "POST:" in prompt:
        prompt = prompt.split("POST:", 1)[1]

    # 保留 TL;DR:
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
# 6. 模型加载
# ===============================
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device)

model.config.pad_token_id = tokenizer.pad_token_id

# ===============================
# 7. 训练参数
# ===============================
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,

    learning_rate=1e-5,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,

    logging_steps=5,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",

    bf16=True,
    fp16=False,

    report_to=["wandb"],
    run_name=run.name,
    max_length=1024,
    completion_only_loss=True,
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
# 打印一个样本看看格式
print("--- Sample Check ---")
sample_preview = dataset_for_sft["train"][0]
print("Prompt:\n", sample_preview["prompt"])
print("Completion:\n", sample_preview["completion"])
print("Prompt + Completion (model input):\n", sample_preview["prompt"] + sample_preview["completion"])
print("--------------------")

print("🚀 Start SFT (POST-only, Qwen3-1.7B)")
trainer.train()

# ===============================
# 10. 推理测试
# ===============================
def test_model(idx=0):
    raw_prompt = dataset_small["test"][idx]["prompt"]
    gold = dataset_small["test"][idx]["label"]

    prompt = extract_post_only(raw_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
        "post_tail": prompt[-200:],
        "human_summary": gold,
        "model_summary": summary,
    })


test_model(5)
test_model(7)

# ===============================
# 11. 保存模型
# ===============================
trainer.save_model(f"{output_dir}/final_checkpoint")
wandb.finish()
