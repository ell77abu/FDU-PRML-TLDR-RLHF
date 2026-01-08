# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
# from datasets import load_dataset

# # --- 1. 配置 ---
# model_id = "./sft-tldr/final_checkpoint" # SFT 模型路径
# rm_model_id = "./rm-tldr/final_rm"       # 训练好的 RM 路径
# device = "cuda" if torch.cuda.is_available() else "cpu"

# config = PPOConfig(
#     model_name="qwen-1.8b-ppo",
#     learning_rate=1.41e-5,
#     batch_size=32,          # 每次 PPO 更新使用的样本总数
#     mini_batch_size=4,      # 显存限制下的微批次
#     gradient_accumulation_steps=8,
#     optimize_cuda_cache=True,
#     early_stopping=False,
#     target_kl=0.1,          # 目标 KL 散度
#     init_kl_coeff=0.1,      # 初始 KL 惩罚系数
#     adap_kl_ctrl=True,      # 动态调整 KL
# )

# # --- 2. 加载模型与分词器 ---
# # 注意：PPO 生成通常用 left padding
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left" 

# # 加载带 Value Head 的策略模型
# model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     model_id, 
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# ).to(device)

# # 创建参考模型（冻结的 SFT，用于计算 KL）
# ref_model = create_reference_model(model)

# # 加载奖励模型（RM）
# from transformers import AutoModelForSequenceClassification
# reward_model = AutoModelForSequenceClassification.from_pretrained(
#     rm_model_id, 
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True
# ).to(device).eval()

# # --- 3. 数据处理 ---
# dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train")
# dataset = dataset.shuffle(seed=42).select(range(5000)) # PPO 迭代通常不需要海量数据

# def tokenize(sample):
#     # 严格对齐训练时的 Prompt 格式
#     prompt = f"Post: {sample['info']['post']}\nTL;DR:"
#     sample["input_ids"] = tokenizer.encode(prompt, truncation=True, max_length=512)
#     sample["query"] = prompt
#     return sample

# dataset = dataset.map(tokenize, batched=False)
# dataset.set_format(type="torch")

# def collator(data):
#     return {key: [d[key] for d in data] for key in data[0]}

# # --- 4. 初始化 PPO Trainer ---
# ppo_trainer = PPOTrainer(
#     config=config,
#     model=model,
#     ref_model=ref_model,
#     tokenizer=tokenizer,
#     dataset=dataset,
#     data_collator=collator,
# )

# # --- 5. 训练循环 ---
# generation_kwargs = {
#     "min_length": -1,
#     "top_k": 0.0,
#     "top_p": 1.0,
#     "do_sample": True,
#     "pad_token_id": tokenizer.pad_token_id,
#     "max_new_tokens": 100, # 摘要长度控制
# }



# print("🚀 开始 PPO 训练...")
# for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
#     query_tensors = batch["input_ids"]

#     # A. Policy 模型生成摘要
#     response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
#     batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

#     # B. 构造 RM 的输入并打分
#     # 注意：RM 评估时通常需要对齐训练时的拼接格式
#     texts = [q + r + tokenizer.eos_token for q, r in zip(batch["query"], batch["response"])]
    
#     # 切换到 RM 需要的右填充模式进行推理（临时操作）
#     tokenizer.padding_side = "right"
#     inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
#     with torch.no_grad():
#         rewards = reward_model(**inputs).logits.squeeze(-1)
    
#     # 奖励归一化处理（可选：减去均值，让奖励有正有负）
#     rewards = [torch.tensor(r) for r in rewards]
#     tokenizer.padding_side = "left" # 切回生成模式

#     # C. 执行 PPO 步进
#     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
#     # D. 打印关键指标
#     if epoch % 10 == 0:
#         ppo_trainer.log_stats(stats, batch, rewards)
#         print(f"Epoch {epoch} | Mean Reward: {stats['ppo/returns/mean']:.4f} | Mean KL: {stats['objective/kl']:.4f}")

# # --- 6. 保存模型 ---
# ppo_trainer.save_pretrained("./ppo-tldr-final")
# print("✅ PPO 训练完成并保存！")

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import LoraConfig, get_peft_model

# model_path = "./models/sft-tldr/final_checkpoint"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # 1. 加载原始 SFT 模型
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model_sft = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     torch_dtype=torch.bfloat16
# ).to(device)

# # 2. 准备测试数据
# test_prompt = "SUBREDDIT: r/relationships\nTITLE: My boyfriend is depressed...\nPOST: [此处省略原文内容]\nTL;DR:"
# inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

# # 3. 原始 SFT 生成
# with torch.no_grad():
#     out_sft = model_sft.generate(**inputs, max_new_tokens=30, do_sample=False)
#     text_sft = tokenizer.decode(out_sft[0], skip_special_tokens=True)

# # 4. 挂载 LoRA (模拟 PPO 刚开始的状态)
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )
# model_lora = get_peft_model(model_sft, lora_config)

# # 5. LoRA 版本生成
# with torch.no_grad():
#     out_lora = model_lora.generate(**inputs, max_new_tokens=50, do_sample=False)
#     text_lora = tokenizer.decode(out_lora[0], skip_special_tokens=True)

# print(f"\n--- 原始 SFT 输出 ---\n{text_sft}")
# print(f"\n--- 挂载 LoRA 后输出 ---\n{text_lora}")

# # 6. 逻辑检查
# is_same = text_sft == text_lora
# print(f"\n结论：两者输出是否一致? {'✅ 是' if is_same else '❌ 否'}")





import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

# --- 1. 配置路径 ---
sft_model_path = "./models/sft-tldr/final_checkpoint"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 加载模型与 Tokenizer ---
print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
# SFT 模型用于生成摘要
sft_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path, torch_dtype=torch.bfloat16
).to(device).eval()


# --- 3. 加载测试数据集 ---
dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
# 选取 50 条数据进行对比测试
test_samples = dataset["validation"].shuffle(seed=42).select(range(10))

results = []

print("开始生成...")
for i, sample in enumerate(tqdm(test_samples)):
    prompt = f"Post: {sample['info']['post']}\nTL;DR:"
    human_summary = f" {sample['summaries'][sample['choice']]['text']}{tokenizer.eos_token}"
    
    # --- Step A: SFT 模型生成摘要 ---
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sft_model.generate(
            **inputs, 
            max_new_tokens=80, 
            # do_sample=True, 
            do_sample=False, 
            # temperature=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    # 提取生成的摘要部分
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sft_summary = full_text.split("TL;DR:")[-1].strip()
    print(f"\n样本 {i+1}:\nHuman: {human_summary}\nSFT: {sft_summary}")