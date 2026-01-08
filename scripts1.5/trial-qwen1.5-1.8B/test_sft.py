# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# from tqdm import tqdm

# # --- 1. 配置 ---
# # 使用你训练好的 SFT 模型路径
# model_path = "./models/sft-tldr/final_checkpoint"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"正在加载模型: {model_path}")
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# # SFT 训练时使用的是右填充，推理时通常建议左填充以配合 generate
# tokenizer.padding_side = "left" 
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True
# ).eval()

# # --- 2. 加载测试数据 ---
# print("正在加载测试集...")
# dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test")

# # 随机选几个样本进行测试
# num_samples = 5
# test_samples = dataset.shuffle(seed=42).select(range(num_samples))

# # --- 3. 生成与对比 ---
# generation_config = {
#     "max_new_tokens": 50,
#     "do_sample": True,
#     "top_p": 0.9,
#     "temperature": 0.8,
#     "repetition_penalty": 1.1,
#     "eos_token_id": tokenizer.eos_token_id,
#     "pad_token_id": tokenizer.pad_token_id,
# }

# print("\n" + "="*50)
# print("SFT 模型摘要效果测试")
# print("="*50 + "\n")

# for i, sample in enumerate(test_samples):
#     prompt = sample["prompt"]
#     ground_truth = sample["label"]
    
#     # 编码输入
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # 生成摘要
#     with torch.no_grad():
#         output_ids = model.generate(**inputs, **generation_config)
    
#     # 解码输出（只取生成的部分）
#     input_len = inputs.input_ids.shape[1]
#     generated_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    
#     print(f"【测试样本 {i+1}】")
#     # print(f"原文提示词: {prompt[-200:]}...") # 只打印末尾部分
#     print(f"--- 原始摘要 (Ground Truth) ---\n{ground_truth}")
#     print(f"--- 模型生成摘要 ---\n{generated_text.strip()}")
#     print("-" * 30 + "\n")

# print("测试完成！")



# # --- 7. ROUGE 测试：对比训练前后的模型性能 ---
# from datasets import load_metric

# # 加载 ROUGE 指标
# rouge = load_metric("rouge")

# def evaluate_rouge(model, dataset, tokenizer):
#     model.eval()
#     preds, labels = [], []
#     for item in dataset:
#         prompt = item['prompt']
#         label = item['label']
        
#         inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=50)
        
#         decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         preds.append(decoded_output)
#         labels.append(label)

#     results = rouge.compute(predictions=preds, references=labels)
#     return results

# # 对比训练前后的模型 ROUGE 分数
# print("\n--- 训练前模型测试 ---")
# orgin_model_path = "./models/Qwen1.5-1.8B"
# # 加载原始模型
# original_model = AutoModelForCausalLM.from_pretrained(
#     orgin_model_path,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
#     local_files_only=True,
# ).to(device)

# original_rouge_results = evaluate_rouge(original_model, dataset_small['test'], tokenizer)
# print(f"Original Model ROUGE Results: {original_rouge_results}")

# print("\n--- 训练后模型测试 ---")
# # 使用训练后的模型进行 ROUGE 测试
# trained_rouge_results = evaluate_rouge(model, dataset_small['test'], tokenizer)
# print(f"Trained Model ROUGE Results: {trained_rouge_results}")




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from datasets import load_metric

# --- 1. 配置 ---
# 使用你训练好的 SFT 模型路径
model_path = "./models/sft-tldr/final_checkpoint"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"正在加载模型: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# SFT 训练时使用的是右填充，推理时通常建议左填充以配合 generate
tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# --- 2. 加载测试数据 ---
print("正在加载测试集...")
dataset = load_dataset("CarperAI/openai_summarize_tldr", split="test")

# 随机选几个样本进行测试
num_samples = 50
test_samples = dataset.shuffle(seed=42).select(range(num_samples))

# # --- 3. 生成与对比 ---
# generation_config = {
#     "max_new_tokens": 50,
#     "do_sample": True,
#     "top_p": 0.9,
#     "temperature": 0.8,
#     "repetition_penalty": 1.1,
#     "eos_token_id": tokenizer.eos_token_id,
#     "pad_token_id": tokenizer.pad_token_id,
# }

# print("\n" + "="*50)
# print("SFT 模型摘要效果测试")
# print("="*50 + "\n")

# for i, sample in enumerate(test_samples):
#     prompt = sample["prompt"]
#     ground_truth = sample["label"]
    
#     # 编码输入
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     # 生成摘要
#     with torch.no_grad():
#         output_ids = model.generate(**inputs, **generation_config)
    
#     # 解码输出（只取生成的部分）
#     input_len = inputs.input_ids.shape[1]
#     generated_text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    
#     print(f"【测试样本 {i+1}】")
#     # print(f"原文提示词: {prompt[-200:]}...") # 只打印末尾部分
#     print(f"--- 原始摘要 (Ground Truth) ---\n{ground_truth}")
#     print(f"--- 模型生成摘要 ---\n{generated_text.strip()}")
#     print("-" * 30 + "\n")

# print("测试完成！")

# --- 4. ROUGE 测试：对比训练前后的模型性能 ---

# 加载 ROUGE 指标
rouge = load_metric("rouge")

def evaluate_rouge(model, dataset, tokenizer):
    model.eval()
    preds, labels = [], []
    for item in tqdm(dataset, desc="评估中..."):
        prompt = item['prompt']
        label = item['label']
        
        # 对数据进行编码并推理
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        # 解码输出（只取生成的部分）
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(decoded_output)
        labels.append(label)

    # 使用 ROUGE 指标进行计算
    results = rouge.compute(predictions=preds, references=labels)
    return results

# --- 5. 对比训练前后的模型 ROUGE 分数 ---

print("\n--- 训练前模型测试 ---")
orgin_model_path = "./models/Qwen1.5-1.8B"
# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    orgin_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device)

# 测试原始模型的 ROUGE
original_rouge_results = evaluate_rouge(original_model, test_samples, tokenizer)
print(f"Original Model ROUGE Results: {original_rouge_results}")

print("\n--- 训练后模型测试 ---")
# 使用训练后的模型进行 ROUGE 测试
trained_rouge_results = evaluate_rouge(model, test_samples, tokenizer)
print(f"Trained Model ROUGE Results: {trained_rouge_results}")
