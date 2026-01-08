import os
import time
import torch
import wandb  # 先禁用 wandb 记录，避免联网

# --- 1. 配置与环境 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用本地基础模型（非 Chat）；已下载到 models/Qwen1.5-1.8B
model_id = './models/Qwen1.5-1.8B'
output_dir = "./sft-tldr"

# --- 0. （可选）初始化 Weights & Biases ---
run = wandb.init(
    project="prml-sft",
    name=f"{model_id.split('/')[-1]}-{int(time.time())}",
    config={
        "model_id": model_id,
        "train_sample": 10000,
        "eval_sample": 750,
        "test_sample": 750,
        "learning_rate": 2e-5,
        "num_train_epochs": 2,
        "batch_size": 4,
        "grad_accum": 32,
        "max_seq_length": 1024,
    },
)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False,
    trust_remote_code=True,
    local_files_only=True,  # 强制使用本地模型文件
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

from datasets import load_dataset
from trl import DataCollatorForCompletionOnlyLM
# --- 2. 数据加载与预处理（小样本快速验证） ---
dataset = load_dataset('CarperAI/openai_summarize_tldr')

# 小样本子集，便于快速验证格式和训练流程
train_sample = 10000
eval_sample = 750
test_sample = 750
dataset_small = {
    "train": dataset["train"].select(range(min(train_sample, len(dataset["train"])))),
    "valid": dataset["valid"].select(range(min(eval_sample, len(dataset["valid"])))),
    "test": dataset["test"].select(range(min(test_sample, len(dataset["test"])))),
}

# 数据预处理：提取并格式化原文（Post）和 TL;DR 摘要
def preprocess_and_filter(example):
    # 提取内容
    if "POST:" in example['prompt'] and "TL;DR:" in example['prompt']:
        post_start = example['prompt'].find("POST:") + 5
        tldr_start = example['prompt'].find("TL;DR:")
        post_content = example['prompt'][post_start:tldr_start].strip()
        
        # 使用 TL;DR 格式，保持与 RM 训练和 PPO 训练一致
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

dataset = dataset.filter(preprocess_and_filter)

# 抽样
dataset_small = {
    "train": dataset["train"].select(range(min(train_sample, len(dataset["train"])))),
    "valid": dataset["valid"].select(range(min(eval_sample, len(dataset["valid"])))),
    "test": dataset["test"].select(range(min(100, len(dataset["test"])))),
}

# 格式化函数，确保 prompt 和 label 之间有清晰的界限
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['processed_prompt'])):
        text = f"{example['processed_prompt'][i]} {example['label'][i]}{tokenizer.eos_token}"
        output_texts.append(text)
    return output_texts

# 确保只学习 TL;DR: 之后的内容
response_template = "TL;DR:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

from transformers import AutoModelForCausalLM, TrainingArguments
# --- 3. 加载模型 ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,  # 避免联网
).to(device)

# --- 4. 训练参数设定 (只跑 1 个 Epoch) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,  # 降低显存占用
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,

    learning_rate=2e-5,
    num_train_epochs=2,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=False,
    bf16=True,
    report_to=["wandb"],  # 启用 wandb 日志
    run_name=run.name,
)

from trl import SFTTrainer
# --- 5. 启动训练 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_small['train'],
    eval_dataset=dataset_small['valid'],
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    max_seq_length=1024,  # 降低序列长度以节省显存
)

print("Starting training...")
trainer.train()

# --- 6. 训练结果展示与测试 ---
print("\n--- 训练完成，开始推理测试 ---")

def test_model(sample_idx=0):
    test_text = dataset_small['test'][sample_idx]['prompt']
    ground_truth = dataset_small['test'][sample_idx]['label']
    
    # 构建推理提示，显式加入 TL;DR: 引导模型输出摘要
    infer_prompt = f"{test_text}"
    inputs = tokenizer(infer_prompt, return_tensors="pt").to(device)
    model.config.pad_token_id = tokenizer.pad_token_id # new add

    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,  # 限制生成内容的长度
            do_sample=True,  # 使用贪婪解码False
            top_p=0.9, 
            temperature=0.8,
            repetition_penalty=1.1, # 增加重复惩罚
            eos_token_id=tokenizer.eos_token_id
        )
    
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的摘要部分（去掉输入部分）
    summary_only = decoded_output.split("TL;DR:")[-1].strip()
    
    print(f"\n[原文提示词]: {test_text[-200:]}...")
    print(f"\n[真实标签 (Human)]: {ground_truth}")
    print(f"\n[模型生成 (Model)]: {summary_only}")

    wandb.log({
        "sample_idx": sample_idx,
        "prompt_tail": test_text[-200:],
        "ground_truth": ground_truth,
        "model_summary": summary_only,
    })

# 测试两条数据看看效果
test_model(7)
test_model(5)

# 保存最终结果
trainer.save_model(f"{output_dir}/final_checkpoint")
wandb.finish()

# --- 7. ROUGE 测试：对比训练前后的模型性能 ---
from datasets import load_metric

# 加载 ROUGE 指标
rouge = load_metric("rouge")

def evaluate_rouge(model, dataset, tokenizer):
    model.eval()
    preds, labels = [], []
    for item in dataset:
        prompt = item['prompt']
        label = item['label']
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(decoded_output)
        labels.append(label)

    results = rouge.compute(predictions=preds, references=labels)
    return results

# 对比训练前后的模型 ROUGE 分数
print("\n--- 训练前模型测试 ---")
# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device)

original_rouge_results = evaluate_rouge(original_model, dataset_small['test'], tokenizer)
print(f"Original Model ROUGE Results: {original_rouge_results}")

print("\n--- 训练后模型测试 ---")
# 使用训练后的模型进行 ROUGE 测试
trained_rouge_results = evaluate_rouge(model, dataset_small['test'], tokenizer)
print(f"Trained Model ROUGE Results: {trained_rouge_results}")
