import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 定义模型路径
# model_path = "./models/Qwen2.5-1.5B-Instruct"
# model_path = "./models/Qwen2.5-1.5B"
# model_path = "./models/Qwen1.5-1.8B"
model_path = "./models/Qwen3-1.7B"

print("正在加载模型到 4090 D...")
# 使用 half precision (float16) 以节省显存并匹配 409 D 特性
# 尝试使用 use_fast=False 解决某些版本分词器报错问题
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)

# 准备一个简单的提示词
prompt = "Explain what is a Large Language Model."
# 自动检测是否有 chat_template，没有则直接拼接
if tokenizer.chat_template:
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
else:
    text = f"User: {prompt}\nAssistant:"

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("模型正在生成回答...")
generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 30)
print(f"回答内容:\n{response.split('assistant')[-1]}")
print("-" * 30)

# 显示显存占用
memory_used = torch.cuda.memory_allocated() / 1024**3
print(f"当前 4090 D 显存占用: {memory_used:.2f} GB")