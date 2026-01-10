import os
import time
import torch
import wandb
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from peft import LoraConfig

# ===============================
# 0. 基础配置
# ===============================
device = "cuda"
torch_dtype = torch.float16
sft_model_path = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
rm_model_path  = "/workspace/pj-RL/experiments3/qwen3-rm-normalized" # 使用归一化后的奖励模型
output_dir = "/workspace/pj-RL/experiments3/qwen3-ppo-final"

os.makedirs(output_dir, exist_ok=True)

# ===============================
# 0.1 wandb 配置
# ===============================
run = wandb.init(
    project="prml-norm-ppo-5000",
    name=f"Qwen3-PPO-demote-norm-5000-{int(time.time())}",
    config={
        "model": sft_model_path,
        "reward_model": rm_model_path,
        "train_samples": 5000,
        "learning_rate": 5e-6,
        "batch_size":32,
        "mini_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "ppo_epochs": 1,
        "target_kl": 0.05,
        "init_kl_coef": 0.04,
        "max_new_tokens": 60,
        "rm_offload": "cpu",  # RM 放在 CPU，推理时临时移到 GPU
    },
)

# ===============================
# 1. Tokenizer (修复 Mistral Regex)
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    sft_model_path,
    trust_remote_code=True,
    fix_mistral_regex=True,
)
# PPO 训练建议左侧填充，以便 generate 正常工作
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# ===============================
# 2. PPO Config (针对 TRL 0.9.6 优化)
# ===============================
config = PPOConfig(
    learning_rate=5e-6,
    batch_size=64,               # 减小到 16 避免 OOM
    mini_batch_size=2,           # 叛逆
    gradient_accumulation_steps=32,
    ppo_epochs=1,                # 每一批数据重复优化的次数
    #B. 为什么建议 ppo_epochs=1？在小数据集（500条）时，为了让模型“吃透”数据，我们设为 2。但在 5000 条时，数据量足够丰富，设为 1 可以显著降低 Reward Hacking 的风险。模型每步只看一次新数据，KL 增长会线性且平稳，而不是指数级跳变。
    target_kl=0.05,              # 限制模型与 SFT 模型的偏差
    init_kl_coef=0.1,           # KL散度惩罚系数
    # 针对大数据量新增：adap_kl_ctrl
    adap_kl_ctrl=True,
    optimize_cuda_cache=True,    # 0.9.6 特有：每步清理显存碎片
    seed=42,
    # wandb 配置
    log_with="wandb",
    # tracker_project_name="qwen3-ppo",
)

# ===============================
# 3. 加载模型 (Policy + Reference)
# ===============================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 自动创建带 Value Head 的 Causal LM
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_path,
    peft_config=lora_config,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="auto",
)

# 开启梯度检查点 (24GB 显存必须开启)
policy_model.gradient_checkpointing_enable()

# 创建冻结的参考模型
ref_model = create_reference_model(policy_model)
for param in ref_model.parameters():
    param.requires_grad = False
# ===============================
# 4. 加载奖励模型 (Reward Model)
# ===============================
print("\n🔹 Loading Reward Model (on CPU to save GPU memory)...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="cpu",  # 放在 CPU，推理时临时移到 GPU
)

# 🔧 归一化模型 bias 加载适配
if hasattr(reward_model, "score") and reward_model.score.bias is None:
    print("  ⚠️  Score head has no bias, loading from state dict...")
    state_dict_path = os.path.join(rm_model_path, "pytorch_model.bin")
    if os.path.exists(state_dict_path):
        state_dict = torch.load(state_dict_path, map_location="cpu")
        if "score.bias" in state_dict:
            old_score = reward_model.score
            new_score = torch.nn.Linear(old_score.in_features, old_score.out_features, bias=True)
            new_score.weight.data = old_score.weight.data
            new_score.bias.data = state_dict["score.bias"].to(dtype=torch_dtype)
            reward_model.score = new_score
            print(f"  ✅ Loaded normalized RM with bias = {new_score.bias.item():.6f}")
        else:
            print("  ⚠️  WARNING: No bias found, RM may not be normalized!")
elif hasattr(reward_model, "score") and reward_model.score.bias is not None:
    print(f"  ✅ RM loaded with bias = {reward_model.score.bias.item():.6f}")

reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False
print("  💡 RM will be moved to GPU only during inference")

# ===============================
# 5. 数据集准备 
# tokenizer.decode(output_ids[0, len(input_ids[0]):], skip_special_tokens=True)
# ===============================
raw_dataset = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")["train"]

def tokenize_fn(example):
    prompt = f"{example['info']['post']}\n\nTL;DR:" # POST: 开头改为正常格式
    # 注意：这里只处理 input_ids，不进行 padding
    inputs = tokenizer(prompt, truncation=True, max_length=1024)
    return {
        "input_ids": inputs["input_ids"],
        "query": prompt
    }

# 选取 500 条进行 Baseline 训练
ppo_dataset = raw_dataset.shuffle(seed=42).select(range(5000)).map(tokenize_fn, remove_columns=raw_dataset.column_names)
ppo_dataset.set_format(type="torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# ===============================
# 6. 初始化 PPOTrainer
# ===============================
ppo_trainer = PPOTrainer(
    config=config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=ppo_dataset,
    data_collator=collator,
)
# ===============================
# 7. 训练循环
# ===============================
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.95,
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 60,  # 增加长度，解决摘要写不完的问题
}

print("\n🚀 Starting PPO Training Baseline...\n")

for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # --- Step 1: Rollout (模型生成) ---
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # --- Step 2: Scoring (RM 打分) ---
    # 奖励模型通常预期格式为: Prompt + Response
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 临时将 RM 移到 GPU 进行推理
    reward_model.to(device)
    with torch.no_grad():
        # 假设 RM 输出的 logits 的第一个维度是奖励分数
        outputs = reward_model(**inputs)
        # 获取分数并转为 tensor list
        # rewards = [torch.tensor(score.item()) for score in outputs.logits]
        # 显式地放大奖励信号，或者进行动态标准化
        rewards = [torch.tensor(score.item() * 2.0) for score in outputs.logits] # 尝试放大 2 倍
    # 推理完成后移回 CPU 释放显存
    reward_model.to("cpu")
    torch.cuda.empty_cache()

    # --- Step 3: PPO Step (更新模型) ---
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # 打印监控指标
    ppo_trainer.log_stats(stats, batch, rewards)

    # --- Step 4: wandb 记录 ---
    # TRL 的 log_stats 已自动记录大部分指标，这里只补充关键的自定义指标
    reward_scores = [r.item() for r in rewards]
    
    # 计算生成文本长度
    response_lengths = [len(r) for r in response_tensors]
    
    wandb.log({
        "epoch": epoch,
        # === 最重要：Reward 统计 ===
        "reward/mean": sum(reward_scores) / len(reward_scores),
        "reward/max": max(reward_scores),
        "reward/min": min(reward_scores),
        "reward/std": torch.tensor(reward_scores).std().item(),
        
        # === 生成质量指标 ===
        "generation/length_mean": sum(response_lengths) / len(response_lengths),
        "generation/length_max": max(response_lengths),
        "generation/sample": batch["response"][0] if batch["response"] else "",
        
        # === 核心 PPO 指标（从 stats 提取）===
        "ppo/loss/total": stats.get("ppo/loss/total", 0),
        "ppo/loss/policy": stats.get("ppo/loss/policy", 0),
        "ppo/loss/value": stats.get("ppo/loss/value", 0),
        "ppo/policy/entropy": stats.get("ppo/policy/entropy", 0),  # 多样性指标
        "ppo/policy/approxkl": stats.get("ppo/policy/approxkl", 0),  # 实际 KL
        "ppo/policy/clipfrac": stats.get("ppo/policy/clipfrac", 0),  # clip 比例
        "ppo/returns/mean": stats.get("ppo/returns/mean", 0),
        "ppo/val/vpred": stats.get("ppo/val/vpred", 0),
        "ppo/val/error": stats.get("ppo/val/error", 0),
    })

    # --- Step 4: 保存 Checkpoint ---
    if (epoch + 1) % 50 == 0:
        ppo_trainer.save_pretrained(os.path.join(output_dir, f"step_{epoch+1}"))

# 最终保存
ppo_trainer.save_pretrained(os.path.join(output_dir, "final_ppo_model"))
print(f"\n✅ Training finished. Model saved to {output_dir}")

# 结束 wandb 记录
wandb.finish()