import os
import time
import torch
import wandb

# =========================
# 改进版 LoRA SFT 训练配置
# 主要改进：
# - 增加数据量到50000个样本
# - 提高LoRA rank到16，扩展到MLP层
# - 优化学习率和批次配置
# - 添加数据质量过滤
# - 增加训练轮数到3
# =========================
# 1. 环境与基本配置
# =========================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_id = './models/Qwen1.5-1.8B'
output_dir = "./sft-tldr-lora-improved"

run = wandb.init(
    project="prml-sft-lora-improved",
    name=f"qwen1.5-1.8b-lora-improved-{int(time.time())}",
    config={
        "train_sample": 50000,      # 从10000增加到50000
        "eval_sample": 2000,        # 相应增加验证集
        "learning_rate": 5e-5,      # 从1e-4降低到5e-5
        "num_train_epochs": 3,       # 从1增加到3
        "batch_size": 2,            # 从1增加到2
        "grad_accum": 4,            # 从8降低到4
        "max_seq_length": 512,      # 从768降低到512
        "lora_r": 16,               # 从8增加到16
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

# 增加数据量
train_sample = 50000
eval_sample = 2000
test_sample = 1000

# 数据预处理和过滤 - 提取POST内容，提高训练效果
def preprocess_and_filter(example):
    """预处理数据：提取POST内容，过滤低质量样本"""

    # 提取POST内容
    if "POST:" in example['prompt'] and "TL;DR:" in example['prompt']:
        post_start = example['prompt'].find("POST:")
        tldr_start = example['prompt'].find("TL;DR:")
        if post_start != -1 and tldr_start != -1:
            post_content = example['prompt'][post_start + 5:tldr_start].strip()
            # 创建简洁的输入格式
            example['processed_prompt'] = f"{post_content}\nTL;DR:"
        else:
            example['processed_prompt'] = example['prompt']
    else:
        example['processed_prompt'] = example['prompt']

    # 数据质量过滤
    prompt_len = len(example['processed_prompt'].split())
    label_len = len(example['label'].split())

    # 保留高质量样本
    keep = (prompt_len > 30 and prompt_len < 200 and      # POST内容合理长度
            label_len > 5 and label_len < 50 and           # 摘要合理长度
            label_len / prompt_len < 0.5)                  # 压缩比合理

    return keep

print(f"原始数据集大小: {len(dataset['train'])}")
dataset = dataset.filter(preprocess_and_filter)
print(f"过滤后数据集大小: {len(dataset['train'])}")

dataset_small = {
    "train": dataset["train"].select(range(min(train_sample, len(dataset["train"])))),
    "valid": dataset["valid"].select(range(min(eval_sample, len(dataset["valid"])))),
    "test": dataset["test"].select(range(min(test_sample, len(dataset["test"])))),
}

def formatting_prompts_func(example):
    """格式化训练样本：使用处理后的prompt + 标签"""
    texts = []
    for i in range(len(example["processed_prompt"])):
        text = f"{example['processed_prompt'][i]} {example['label'][i]}{tokenizer.eos_token}"
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

# 改进的LoRA配置 - 提高表达能力和训练效果
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # 从8增加到16，提高表达能力
    lora_alpha=32,           # 从16增加到32，优化缩放比例
    lora_dropout=0.05,
    bias="none",
    target_modules=[         # 扩展到attention和MLP层，提高适应性
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
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
    output_dir=f"{output_dir}-improved",  # 新的输出目录
    per_device_train_batch_size=2,        # 从1增加到2
    gradient_accumulation_steps=4,        # 从8降低到4
    gradient_checkpointing=True,          # 启用梯度检查点节省显存

    learning_rate=5e-5,                   # 从1e-4降低到5e-5
    num_train_epochs=3,                   # 从1增加到3
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    logging_steps=50,                     # 减少日志频率
    eval_strategy="steps",
    eval_steps=500,                       # 增加评估间隔
    save_strategy="steps",
    save_steps=1000,                      # 增加保存间隔


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
    max_seq_length=512,  # 序列最大长度
)

# =========================
# 6. 开始训练
# =========================
try:
    print("🚀 Starting LoRA SFT training...")
    trainer.train()

    # =========================
    # 7. 推理测试
    # =========================
    print("\n--- 推理测试 ---")

    def test_model(idx=0):
        # 使用处理后的prompt进行测试
        original_prompt = dataset_small["test"][idx]["prompt"]
        processed_prompt = dataset_small["test"][idx]["processed_prompt"]
        gt = dataset_small["test"][idx]["label"]

        inputs = tokenizer(processed_prompt, return_tensors="pt").to(device)

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

        print("\n[原始Prompt]")
        print(original_prompt[-200:])
        print("\n[处理后输入]")
        print(processed_prompt)
        print("\n[Human摘要]")
        print(gt)
        print("\n[Model生成]")
        print(summary)

    test_model(7)
    test_model(5)

    # =========================
    # 8. 保存 LoRA Adapter
    # =========================
    trainer.save_model(f"{output_dir}/lora_adapter_final")
    print(f"✅ 模型已保存到: {output_dir}/lora_adapter_final")

except Exception as e:
    print(f"❌ 训练过程中出现错误: {e}")
    print("请检查配置参数是否正确")

finally:
    # 确保wandb连接正确关闭
    wandb.finish()
