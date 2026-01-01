import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import load_dataset

# --- 1. 配置 ---
sft_model_path = "./models/sft-tldr/final_checkpoint"
rm_model_path = "./models/rm-tldr/final_rm"  # 你最新训练好的 RM

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = PPOConfig(
    model_name="qwen-1.8b-ppo",
    learning_rate=1.41e-5,
    batch_size=64,           # 采样总数
    mini_batch_size=1,       # 【显存优化】4090 建议设为 1，防止 Value Head 计算时 OOM
    gradient_accumulation_steps=16,
    optimize_cuda_cache=True,
    target_kl=0.1,           # 限制策略偏离 SFT 太远
    init_kl_coeff=0.03,      # 【针对 0.43 分差优化】初始 KL 设小一点，给微弱奖励留出空间
    reward_whitening=True,   # 【核心】开启奖励归一化（白化），将 Batch 内奖励转为均值0，方差1
)

# --- 2. 加载模型 (针对 24GB 显存优化) ---
tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # PPO 生成必须左填充

# 策略模型 (Policy)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)

# 参考模型 (Reference) - 用于计算 KL 散度
ref_model = create_reference_model(model)

# 奖励模型 (Reward Model)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device}
).eval()

# --- 3. 数据加载 ---
# 使用 TL;DR 任务的 Prompt 部分
dataset = load_dataset("CarperAI/openai_summarize_tldr", split="train")

def tokenize_fn(sample):
    # 构造与 RM 训练一致的 Prompt
    # 原始数据集中 prompt 字段通常已包含 "Post: ... TL;DR:"
    sample["input_ids"] = tokenizer.encode(sample["prompt"], truncation=True, max_length=512)
    sample["query"] = sample["prompt"]
    return sample

dataset = dataset.shuffle(seed=42).map(tokenize_fn, batched=False)
dataset.set_format(type="torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# --- 4. 初始化 PPO Trainer ---
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# --- 5. 训练循环 ---
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "max_new_tokens": 80, 
}

print("🚀 PPO 训练启动...")

for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Step A: 生成响应
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

    # Step B: 计算奖励 (奖励归一化与缩放)
    tokenizer.padding_side = "right" # RM 评估建议右填充
    texts = [q + r + tokenizer.eos_token for q, r in zip(batch["query"], batch["response"])]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        # 获取原始 Logits 分数
        raw_rewards = reward_model(**inputs).logits.squeeze(-1)
        
        # 【重要：奖励缩放】
        # 既然你的平均分差只有 0.43，为了让 PPO 感觉到明显的奖惩差异，
        # 我们在这里乘以一个缩放系数（2.0~3.0），放大信号强度。
        rewards = [r * 2.5 for r in raw_rewards] 

    tokenizer.padding_side = "left" # 恢复左填充准备下一轮

    # Step C: PPO 优化步
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    # 打印监控
    if epoch % 10 == 0:
        ppo_trainer.log_stats(stats, batch, rewards)
        print(f"Step {epoch} | Reward: {torch.mean(raw_rewards).item():.4f} | KL: {stats['objective/kl']:.4f}")

# --- 6. 保存模型 ---
model.save_pretrained("./qwen-1.8b-ppo-final")
tokenizer.save_pretrained("./qwen-1.8b-ppo-final")