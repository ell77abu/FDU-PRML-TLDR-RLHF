# import os
# import torch
# import re
# from tqdm import tqdm
# from transformers import (
#     AutoTokenizer, 
#     AutoModelForSequenceClassification, 
#     BitsAndBytesConfig
# )
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
# from datasets import load_dataset
# from peft import LoraConfig

# # --- 1. 基础配置 ---
# sft_model_path = "./models/sft-tldr/final_checkpoint"
# rm_model_path = "./models/rm-tldr/final_rm" 

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# config = PPOConfig(
#     model_name="qwen-1.8b-ppo-lora-clean",
#     learning_rate=1e-8,          # LoRA 建议比全参数略高
#     batch_size=16,               # 4090 显存适配
#     mini_batch_size=1,           
#     gradient_accumulation_steps=16, 
#     optimize_cuda_cache=True,
#     target_kl=0.1,               # 限制偏离 SFT 太远
#     init_kl_coef=0.2,           # 初始 KL 惩罚，防止模型一上来就乱写
#     whiten_rewards=True,         
# )

# # --- 2. 模型加载与显存压缩 ---
# tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left" 

# # LoRA 配置：覆盖全部线性层以保证表达力
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
#     bias="none",
# )

# # A. 策略与价值模型 (Policy & Value)：共用 LoRA 基座
# model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     sft_model_path,
#     peft_config=lora_config,
#     torch_dtype=torch.bfloat16,
#     device_map={"": device}
# )

# # B. 奖励模型 (RM)：使用 4-bit 量化，将显存占用压到 1.5GB 左右
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4"
# )
# reward_model = AutoModelForSequenceClassification.from_pretrained(
#     rm_model_path,
#     quantization_config=bnb_config,
#     device_map={"": device}
# ).eval()

# # --- 3. 奖励整形逻辑 (针对你的 CSV 问题定制) ---
# def get_shaped_reward(raw_score, response_text):
#     penalty = 0.0
    
#     # 惩罚 1：表情包堆砌 (针对 :) :) :) :) )
#     # 如果任意符号重复出现 3 次以上，重罚
#     if re.search(r'([:\)\!\?\.])\1{2,}', response_text):
#         penalty += 5.0
        
#     # 惩罚 2：数据噪声 (针对 "10 points for...", "SUBREDDIT:")
#     black_list = ["points for", "SUBREDDIT", "POST:", "TITLE:", "Thanks for", "Help!"]
#     for word in black_list:
#         if word.lower() in response_text.lower():
#             penalty += 10.0  # 核心惩罚：禁止复读原文格式
            
#     # 惩罚 3：长度冗余 (摘要应在 15-50 词之间)
#     word_count = len(response_text.split())
#     if word_count > 60:
#         penalty += (word_count - 60) * 0.2  # 线性惩罚长文本
#     elif word_count < 5:
#         penalty += 2.0  # 惩罚过短

#     return raw_score - penalty

# # --- 4. 数据预处理 ---
# dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")
# dataset = dataset.shuffle(seed=42).select(range(512))

# def tokenize_fn(sample):
#     # 1. 先清理掉可能存在的末尾空格
#     query = sample["prompt"].rstrip() 
    
#     # 2. 检查是否已经包含了 TL;DR: 
#     # 如果没有，才添加。如果有，直接使用。
#     if not query.endswith("TL;DR:"):
#         query += "\nTL;DR:"
    
#     sample["query"] = query
#     # 3. 编码转换
#     sample["input_ids"] = tokenizer.encode(query, truncation=True, max_length=512)
#     return sample

# dataset = dataset.map(tokenize_fn, batched=False)
# dataset.set_format(type="torch")

# ppo_trainer = PPOTrainer(
#     config=config,
#     model=model,
#     ref_model=None, # PEFT 模式下传 None 开启显存共享模式
#     tokenizer=tokenizer,
#     dataset=dataset,
#     data_collator=lambda data: {key: [d[key] for d in data] for key in data[0]},
# )

# # --- 5. 训练循环 ---
# generation_kwargs = {
#     "do_sample": True,
#     "temperature": 0.8,         # 保持低温，减少随机乱码
#     "repetition_penalty": 1.5,  # 强力压制原文重复
#     "max_new_tokens": 60,
#     "pad_token_id": tokenizer.pad_token_id,
# }

# print("🚀 启动纠偏版 LoRA-PPO 训练...")

# for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
#     query_tensors = batch["input_ids"]

#     # 生成
#     response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
#     batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

#     # 打分
#     tokenizer.padding_side = "right" 
#     texts = [q + r + tokenizer.eos_token for q, r in zip(batch["query"], batch["response"])]
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
#     with torch.no_grad():
#         raw_rewards = reward_model(**inputs).logits.squeeze(-1)
#         # 应用针对性整形
#         rewards = [get_shaped_reward(r.float(), resp) for r, resp in zip(raw_rewards, batch["response"])]

#     tokenizer.padding_side = "left" 

#     # 更新
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
#     # 日志输出
#     if epoch % 1 == 0:
#         print(f"\nStep {epoch} | Reward: {torch.mean(torch.tensor(rewards)).item():.4f}")
#         print(f"Sample: {batch['response'][0][:150]}") # 监控是否有 :) :)

# # --- 6. 保存 ---
# ppo_trainer.save_pretrained("./qwen-1.8b-ppo-lora-final")




import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import pandas as pd

# --- 1. 配置路径 ---
model_path = "./models/sft-tldr/final_checkpoint"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 加载基础 SFT 模型 ---
print("正在加载原始 SFT 模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
# 必须统一 padding 侧，否则分布会有细微差异
tokenizer.padding_side = "left" 

model_sft = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map={"": device}
).eval()

# --- 3. 创建带有初始 LoRA 的模型 ---
print("正在构建 SFT + LoRA 模型...")
# 使用你 PPO 脚本中完全一样的配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)
# get_peft_model 会在原模型基础上包装 LoRA 层
model_lora = get_peft_model(model_sft, lora_config).to(device).eval()

# --- 4. 准备测试样本 ---
test_samples = [
    "SUBREDDIT: r/relationships TITLE: Me [13 M] and my crush [12 F]. How do I ask her to the upcoming school dance? POST: Hey r/relationships! So this past Thursday my seventh grade class went on a school trip to Boston, and during this trip my crush ended up breaking up with her eighth grader boyfriend, I'll refer to him as Ian. Now I moved to this school this past year and Ian was my first friend, and what he ended up doing was dating my crush, Lily, so what happened on Thursday night is, according to Lily's friend, he sent her a picture of a pornstar in a quite revealing outfit, with a crude message something along the lines of \"if you wear this I'll f*** you\". She immediately broke up with him and things were quite awkward between them today as this was the first school day back. Now I've had a crush on Lily all year, but I found out Ian was dating her so I waited. Now she's free and seems to be over him, and is acting quite nice to me, which is very odd. Now the school dance is approaching in May, so I was curious if I should ask her, how, and when?  If I left anything out feel free to ask! TL;DR:",
    "SUBREDDIT: r/personalfinance TITLE: Prioritize student debt or saving for down payment? POST: I have $25k in student debt. One private loan at 9.5% (highest priority obviously) and nine others federal between 3.4% and 6.8%. Minimum payment per month total is $301.16. Over the next 9 months, I will pay off $11k of these, which will get rid of everything above 5% interest and will drop the total minimum payment to $150.   At the end of the 9 months, our savings will be around $35k. At that time my husband will need to purchase a car so some of that will be his down payment. So more realistically $25-30k.   Sometime in the future, between a year to two years from now, my husband and I may be moving. Typical single family homes in this area go for around $300k.   At the end of the 9 months, should I continue to focus on paying down student debt (which will be a balance of $14k by then) or growing our savings/down payment? I have $5200/mo to somehow split between debt and down payment and I'm not sure how best to allocate it. TL;DR:",
]

# --- 5. 执行对比推理 ---
# 使用固定的生成参数，确保结果可复现
gen_kwargs = {
    "max_new_tokens": 50,
    "do_sample": False,  # 强制贪婪搜索以验证权重一致性
    "repetition_penalty": 1.0,
    "pad_token_id": tokenizer.pad_token_id,
}

results = []

print("\n开始对比推理...\n")
for i, prompt in enumerate(test_samples):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 原始 SFT 输出
        # 注意：由于 model_lora 包装了 model_sft，
        # 我们需要用 model_lora.disable_adapter() 来模拟原始 SFT
        with model_lora.disable_adapter():
            out_sft = model_lora.generate(**inputs, **gen_kwargs)
            text_sft = tokenizer.decode(out_sft[0], skip_special_tokens=True)
        
        # 初始 LoRA 输出
        out_lora = model_lora.generate(**inputs, **gen_kwargs)
        text_lora = tokenizer.decode(out_lora[0], skip_special_tokens=True)
    
    results.append({
        "ID": i + 1,
        "SFT_Output": text_sft.split("TL;DR:")[-1].strip(),
        "LoRA_Init_Output": text_lora.split("TL;DR:")[-1].strip(),
        "Match": text_sft == text_lora
    })

# --- 6. 结果展示 ---
df = pd.DataFrame(results)
print(df.to_string())
# --- 7. 逻辑判定 ---
if df["Match"].all():
    print("\n✅ 验证通过：初始 LoRA 矩阵对模型输出没有任何影响（$\Delta W=0$）。")
    print("这意味着 PPO 刚开始时的复读问题与 LoRA 结构无关，是由于训练超参数或更新导致的。")
else:
    print("\n❌ 警告：两者输出不一致！")
    print("可能原因：1. LoRA 包含了非零初始化的层；2. 某些 target_modules 在加载时改变了原始精度。")