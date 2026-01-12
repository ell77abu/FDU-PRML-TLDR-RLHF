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
rm_model_path  = "/workspace/pj-RL/experiments3/qwen3-rm-normalized" 
output_dir = "/workspace/pj-RL/experiments3/qwen3-ppo-valuehead" 

os.makedirs(output_dir, exist_ok=True)

# ===============================
# 1. Tokenizer 
# ===============================
tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

# ===============================
# 2. PPO Config (按照 OpenAI 论文建议微调)
# ===============================
config = PPOConfig(
    learning_rate=3e-6,
    batch_size=64,               # 增大有效 Batch 以稳定训练
    mini_batch_size=1,           # 24GB 显存保命配置
    gradient_accumulation_steps=64,
    ppo_epochs=1,                
    target_kl=0.05,              # 略微放宽目标，给模型探索空间
    init_kl_coef=0.04,          # 降低初始KL系数。。。😄
    adap_kl_ctrl=True,
    optimize_cuda_cache=True,    
    seed=42,
    log_with="wandb",
    whiten_rewards=True,        # TRL 特性：将一个 Batch 内的奖励归一化到均值 0，标准差 1
)

# ===============================
# 3. 模型加载：策略隔离与 Value Head 初始化
# ===============================

# 配置 LoRA 参数
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["v_head"], # 确保 Value Head 独立训练
)

# 加载 Policy 模型
policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_path,
    peft_config=lora_config,
    torch_dtype=torch_dtype,
    device_map="auto",
)
policy_model.gradient_checkpointing_enable()
policy_model.train()

print("\n🔹 Initializing Value Function with Normalized RM weights...")

# 1. 加载原始权重字典
rm_state_dict = torch.load(os.path.join(rm_model_path, "pytorch_model.bin"), map_location="cpu")

# 2. 从 config.json 中读取手动保存的归一化偏置 (_normalization_bias)
import json
with open(os.path.join(rm_model_path, "config.json"), "r") as f:
    rm_config = json.load(f)

# 获取保存的 bias，如果没找到则默认为 0
norm_bias_shift = rm_config.get("_normalization_bias", 0.0)
print(f"  🔍 Found _normalization_bias in config: {norm_bias_shift}")

with torch.no_grad():
    # 拷贝权重 (Weight 不需要变)
    if "score.weight" in rm_state_dict:
        policy_model.v_head.summary.weight.copy_(rm_state_dict["score.weight"])
        
        # 拷贝偏置 (Bias 需要加上归一化偏移量)
        # 注意：如果原始 RM 有 bias，我们要加上偏移；如果原始 RM 没 bias，就直接设为偏移值
        if "score.bias" in rm_state_dict:
            original_bias = rm_state_dict["score.bias"]
            # 这里的逻辑是：归一化后的 Bias = 原始 Bias + 修正值
            policy_model.v_head.summary.bias.copy_(original_bias + norm_bias_shift)
            print(f"Value Head bias initialized: {original_bias.item():.6f} + ({norm_bias_shift:.6f})")
        else:
            # 如果原始 RM 没 bias (Linear 层 bias=False)，TRL 的 v_head 默认是有 bias 的
            policy_model.v_head.summary.bias.fill_(norm_bias_shift)
            print(f"Value Head bias initialized with shift: {norm_bias_shift:.6f}")
            
    else:
        print("Error: 'score.weight' not found in RM state_dict!")

# 释放临时显存
del rm_state_dict
torch.cuda.empty_cache()

# 创建冻结的参考模型
ref_model = create_reference_model(policy_model)
for param in ref_model.parameters():
    param.requires_grad = False

# 加载独立的奖励模型用于评分
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    torch_dtype=torch_dtype,
    device_map="cpu", 
)
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False

# ===============================
# 4. 数据集准备 (保持 1024 长度)
# ===============================
raw_dataset = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")["train"]

def tokenize_fn(example):
    prompt = f"{example['info']['post']}\n\nTL;DR:"
    inputs = tokenizer(prompt, truncation=True, max_length=1024)
    return {"input_ids": inputs["input_ids"], "query": prompt}

ppo_dataset = raw_dataset.shuffle(seed=42).select(range(15000)).map(tokenize_fn, remove_columns=raw_dataset.column_names)
ppo_dataset.set_format(type="torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# ===============================
# 5. 初始化 PPOTrainer
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
# 6. 训练循环与 WandB 监控增强
# ===============================
generation_kwargs = {
    "top_k": 0.0, "top_p": 0.95, "do_sample": True,
    "temperature": 0.7, "max_new_tokens": 60,
    "pad_token_id": tokenizer.pad_token_id,
}

wandb.init(project="qwen3-ppo-valuehead", name=f"Qwen3-PPO-ValueHead-{int(time.time())}")

for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # --- Step 1: Rollout ---
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # --- Step 2: Scoring ---
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    reward_model.to(device)
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # 根据你的实验结论：这里可以使用归一化分数，如发现 KL 下降再考虑 * 2.0
        rewards = [score for score in outputs.logits.flatten()]
    reward_model.to("cpu")

    # --- Step 3: PPO Step ---
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # --- Step 4: 自定义 WandB 记录 (增加对 Value Function 的监控) ---
    wandb.log({
        "ppo/val/vpred_mean": stats.get("ppo/val/vpred", 0).mean() if hasattr(stats.get("ppo/val/vpred"), 'mean') else 0,
        "ppo/val/error_mean": stats.get("ppo/val/error", 0),
        "reward/mean_batch": torch.stack(rewards).mean().item(),
        "generation/sample_text": wandb.Html(f"<b>Prompt:</b> {batch['query'][0]}<br><b>Response:</b> {batch['response'][0]}")
    })

    if (epoch + 1) % 200 == 0:
        ppo_trainer.save_pretrained(os.path.join(output_dir, f"step_{epoch+1}"))

ppo_trainer.save_pretrained(os.path.join(output_dir, "final_model"))
wandb.finish()