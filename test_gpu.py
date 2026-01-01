import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 定义模型路径
# model_path = "./models/Qwen2.5-1.5B-Instruct"
# model_path = "./models/Qwen2.5-1.5B"
model_path = "./models/Qwen1.5-1.8B"

print("正在加载模型到 4090 D...")
# 使用 half precision (float16) 以节省显存并匹配 4090 特性
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 准备一个简单的提示词
prompt = "请解释一下音乐理论中的“关系大小调”概念"
messages = [
    {"role": "system", "content": "你是一个人工智能助手。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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