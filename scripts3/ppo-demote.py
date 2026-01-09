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
rm_model_path  = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
output_dir = "/workspace/pj-RL/experiments3/qwen3-ppo-final"

os.makedirs(output_dir, exist_ok=True)

# ===============================
# 0.1 wandb 配置
# ===============================
run = wandb.init(
    project="prml-ppo",
    name=f"Qwen3-PPO-demote-{int(time.time())}",
    config={
        "model": sft_model_path,
        "reward_model": rm_model_path,
        "train_samples": 500,
        "learning_rate": 1.41e-5,
        "batch_size": 32,
        "mini_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "ppo_epochs": 4,
        "target_kl": 0.1,
        "init_kl_coef": 0.2,
        "max_new_tokens": 60,
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
    learning_rate=1.41e-5,
    batch_size=32,               # 每 32 条数据执行一次 PPO 更新
    mini_batch_size=2,           # 24GB 显存单卡建议设为 2，防止 OOM
    gradient_accumulation_steps=16,
    ppo_epochs=4,                # 每一批数据重复优化的次数
    target_kl=0.1,               # 限制模型与 SFT 模型的偏差
    init_kl_coef=0.2,
    optimize_cuda_cache=True,    # 0.9.6 特有：每步清理显存碎片
    seed=42,
    # wandb 配置
    log_with="wandb",
    tracker_project_name="qwen3-ppo",
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

# ===============================
# 4. 加载奖励模型 (Reward Model)
# ===============================
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="auto",
)
reward_model.eval()

# ===============================
# 5. 数据集准备
# ===============================
raw_dataset = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")["train"]

def tokenize_fn(example):
    prompt = f"{example['info']['post']}\n\nTL;DR:" # POST: 开头改为正常格式
    # 注意：这里只处理 input_ids，不进行 padding
    inputs = tokenizer(prompt, truncation=True, max_length=512)
    return {
        "input_ids": inputs["input_ids"],
        "query": prompt
    }

# 选取 500 条进行 Baseline 训练
ppo_dataset = raw_dataset.shuffle(seed=42).select(range(20000)).map(tokenize_fn, remove_columns=raw_dataset.column_names)
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
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
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
    
    with torch.no_grad():
        # 假设 RM 输出的 logits 的第一个维度是奖励分数
        outputs = reward_model(**inputs)
        # 获取分数并转为 tensor list
        rewards = [torch.tensor(score.item()) for score in outputs.logits]

    # --- Step 3: PPO Step (更新模型) ---
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # 打印监控指标
    ppo_trainer.log_stats(stats, batch, rewards)

    # --- Step 4: wandb 记录 ---
    # 计算奖励模型分数的平均值
    reward_scores = [r.item() for r in rewards]
    avg_reward = sum(reward_scores) / len(reward_scores)

    # 记录到 wandb
    wandb.log({
        "epoch": epoch,
        "loss": stats.get("ppo/loss/total", 0),
        "reward_score": avg_reward,
        "kl_divergence": stats.get("ppo/loss/kl", 0),
        "generated_text": batch["response"][0] if batch["response"] else "",  # 记录第一个生成的文本作为示例
        "learning_rate": stats.get("ppo/learning_rate", config.learning_rate),
        "ppo/policy/advantages_mean": stats.get("ppo/policy/advantages_mean", 0),
        "ppo/returns/mean": stats.get("ppo/returns/mean", 0),
        "ppo/val/vpred": stats.get("ppo/val/vpred", 0),
    })

    # --- Step 4: 保存 Checkpoint ---
    if (epoch + 1) % 50 == 0:
        ppo_trainer.save_pretrained(os.path.join(output_dir, f"step_{epoch+1}"))

# 最终保存
ppo_trainer.save_pretrained(os.path.join(output_dir, "final_ppo_model"))
print(f"\n✅ Training finished. Model saved to {output_dir}")

# 结束 wandb 记录
wandb.finish()