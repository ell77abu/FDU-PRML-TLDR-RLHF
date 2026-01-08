import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 配置路径 - 请确保这些路径与你服务器上的实际路径一致
base_model_path = "/workspace/pj-RL/models/Qwen3-1.7B"
lora_adapter_path = "/workspace/pj-RL/experiments3/qwen3-sft-lora/final_checkpoint"
output_path = "/workspace/pj-RL/experiments3/qwen3-sft-lora-merged"

def merge():
    print(f"🔄 正在加载基座模型...")
    # 使用 bfloat16 节省内存并保持精度
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 合并建议在 CPU 上进行，避免占用训练用的显存
        trust_remote_code=True
    )

    print(f"⚙️ 正在加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print(f"🏗️ 正在合并权重 (Merge and Unload)...")
    # 这一步是将 LoRA 参数加回到基座参数中
    model = model.merge_and_unload()

    print(f"💾 正在保存完整模型至: {output_path}")
    model.save_pretrained(output_path)
    
    # 保存分词器，ROUGE 脚本需要它
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print("✅ 合并成功！")

if __name__ == "__main__":
    merge()