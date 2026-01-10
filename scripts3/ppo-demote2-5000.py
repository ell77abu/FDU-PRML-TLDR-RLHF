import os
import time
import torch
import wandb
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from peft import LoraConfig

# ============================================================
# 0. Config
# ============================================================
TRAIN_SAMPLES = 5000
PPO_EPOCHS = 2
LEARNING_RATE = 5e-6  # 降低学习率以提高稳定性

device = "cuda"
torch_dtype = torch.float16

sft_model_path = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
rm_model_path  = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"
output_dir = f"/workspace/pj-RL/experiments3/qwen3-ppo-{TRAIN_SAMPLES}-kl"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. wandb
# ============================================================
wandb.init(project="prml-ppo", name=f"qwen3-ppo-{int(time.time())}")

# ============================================================
# 2. Tokenizers
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_tokenizer.padding_side = "left"

# ============================================================
# 3. PPO Config
# ============================================================
config = PPOConfig(
    learning_rate=LEARNING_RATE,
    batch_size=32,
    mini_batch_size=2,
    gradient_accumulation_steps=16,
    ppo_epochs=PPO_EPOCHS,
    target_kl=0.05,         # 提高到合理的目标KL
    init_kl_coef=0.05,      # 提高初始KL系数
    adap_kl_ctrl=True,      # 启用自适应KL控制
    kl_penalty="abs",       # 使用绝对值KL惩罚
    max_kl_coef=0.5,        # 限制KL系数的最大值
    log_with="wandb",
)

# ============================================================
# 4. Policy + Reference
# ============================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_path,
    peft_config=lora_config,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="auto",
)

policy_model.gradient_checkpointing_enable()
policy_model.enable_input_require_grads()

ref_model = create_reference_model(policy_model)

# ============================================================
# 5. Reward Model
# ============================================================
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    trust_remote_code=True,
    torch_dtype=torch_dtype,
    device_map="auto",
)
reward_model.eval()

# ============================================================
# 6. Dataset (PROMPT ONLY)
# ============================================================
raw_dataset = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")["train"]

def tokenize_fn(example):
    prompt = f"{example['info']['post']}\n\nTL;DR:"
    tokens = tokenizer(prompt, truncation=True, max_length=512)
    return {
        "input_ids": tokens["input_ids"],
        "query": prompt,
    }

ppo_dataset = (
    raw_dataset.shuffle(seed=42)
    .select(range(TRAIN_SAMPLES))
    .map(tokenize_fn, remove_columns=raw_dataset.column_names)
)
ppo_dataset.set_format(type="torch")

def collator(data):
    return {k: [d[k] for d in data] for k in data[0]}

# ============================================================
# 7. PPO Trainer
# ============================================================
ppo_trainer = PPOTrainer(
    config=config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=ppo_dataset,
    data_collator=collator,
)

# ============================================================
# 8. Generation (CRITICAL)
# ============================================================
generation_kwargs = dict(
    do_sample=True,         # 必须使用采样，否则没有策略梯度
    top_p=0.5,              # 非常保守的top_p
    top_k=1,                # 极小的top_k
    temperature=0.1,        # 极低的温度，几乎确定性
    max_new_tokens=60,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    return_prompt=False,    # 只返回response，不包含prompt
)

# ============================================================
# 9. PPO Loop
# ============================================================
print("🚀 PPO training")

for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):

    query_tensors = batch["input_ids"]

    # ---- Generate responses only ----
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # ---- Reward ----
    # 拼接完整文本用于奖励计算
    full_texts = [q + r for q, r in zip(batch["query"], responses)]
    rm_inputs = rm_tokenizer(full_texts, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        scores = reward_model(**rm_inputs).logits.squeeze(-1)

    # normalize rewards (防KL负数归一化)
    scores_mean = scores.mean()
    scores_std = scores.std()

    # 对于大数据集，使用更稳定的归一化
    if TRAIN_SAMPLES >= 1000:
        # 使用移动平均来稳定统计
        global global_reward_mean, global_reward_std
        if global_reward_mean == 0.0:  # 第一次初始化
            global_reward_mean = scores_mean.item()
            global_reward_std = max(scores_std.item(), 0.1)
        else:
            alpha = 0.1  # 移动平均系数
            global_reward_mean = alpha * scores_mean.item() + (1-alpha) * global_reward_mean
            global_reward_std = alpha * max(scores_std.item(), 0.1) + (1-alpha) * global_reward_std

        scores_std = torch.tensor(global_reward_std, device=scores.device)

    # 确保数值稳定性
    scores_std = torch.clamp(scores_std, min=0.1, max=10.0)

    scores = (scores - scores_mean) / scores_std

    # 非常严格的奖励裁剪，防止极端值
    scores = torch.clamp(scores, -0.5, 0.5)  # 更严格的裁剪
    rewards = [s.detach() for s in scores]

    # ---- PPO ----
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # 检查KL值，如果异常则跳过此步骤
    kl_value = stats.get("objective/kl", 0)
    if kl_value < -1.0 or kl_value > 10.0:  # KL值异常
        print(f"⚠️ 跳过异常KL值: {kl_value}")
        continue  # 跳过此训练步骤

    batch["response"] = responses
    ppo_trainer.log_stats(stats, batch, rewards)

    wandb.log({
        "reward_mean": scores.mean().item(),
        "reward_std": scores.std().item(),
        "kl": stats.get("objective/kl", 0),
        "loss": stats.get("ppo/loss/total", 0),
        "sample": responses[0],
    })

    if (step + 1) % 200 == 0:
        ppo_trainer.save_pretrained(os.path.join(output_dir, f"step_{step+1}"))

# ============================================================
# 10. Save
# ============================================================
ppo_trainer.save_pretrained(os.path.join(output_dir, "final"))
wandb.finish()
