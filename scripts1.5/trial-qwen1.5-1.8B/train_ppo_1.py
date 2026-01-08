import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification

# ==========================================
# 1. Configuration
# ==========================================
SFT_MODEL_PATH = "/workspace/pj-RL/models/sft-tldr/final_checkpoint"
RM_MODEL_PATH = "/workspace/pj-RL/models/rm-tldr/final_rm"

config = PPOConfig(
    model_name="ppo-1.5-1.8B",
    learning_rate=1.41e-5,
    batch_size=4,           
    mini_batch_size=1,      
    gradient_accumulation_steps=4,
    optimize_cuda_cache=True,
    init_kl_coef=0.05,      
    target_kl=6.0,         
    ppo_epochs=1,
    seed=42,
    whiten_rewards=True,    # 开启奖励白中化（归一化）
)

# ==========================================
# 2. Load Models & Tokenizers
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # Key for generation

# LoRA Config for Policy/Value Model
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Quantization for Policy/Value Model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"Loading SFT model from {SFT_MODEL_PATH}...")
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    SFT_MODEL_PATH,
    quantization_config=bnb_config,
    peft_config=lora_config,
    device_map={"": device},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

ref_model = create_reference_model(model)

print(f"Loading RM model from {RM_MODEL_PATH}...")
# Quantization for Reward Model (8-bit to fit in memory)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    RM_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map={"": device},
    load_in_8bit=True,
    trust_remote_code=True
).eval()

# ==========================================
# 3. Dataset & Processing
# ==========================================
def build_dataset(config):
    # load_from_disk 不支持 split 参数，它返回一个 DatasetDict
    ds_dict = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")
    ds = ds_dict["train"]
    # Take a small subset for quick debugging as requested
    ds = ds.select(range(100)) 

    def tokenize(sample):
        # Extract info
        info = sample['info']
        post = info['post']
        title = info['title']
        subreddit = info['subreddit']
        
        # Format for SFT (Policy) - matches CarperAI structure
        # Note: Ensure format matches exactly what SFT expects
        # "/workspace/pj-RL/datasets/summarize_from_feedback"
        prompt = f"POST: {post}\nTL;DR:"
        
        sample["input_ids"] = tokenizer.encode(prompt, truncation=True, max_length=512)
        sample["query"] = prompt
        sample["post_content"] = post # Store raw post for RM formatting using train_rm.py logic
        return sample

    ds = ds.map(tokenize, batched=False)
    # 核心修复：显式保留需要的列，移除其它所有原始干扰列
    columns_to_keep = ["input_ids", "query", "post_content"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in columns_to_keep])
    
    # 强制不使用 set_format(type="torch")，因为那会剔除字符串列
    # 我们在 collator 里手动转 tensor
    print(f"Dataset columns after filtering: {ds.column_names}")
    return ds

dataset = build_dataset(config)
# 将 Dataset 对象转为 List[dict]，防止 PPOTrainer 内部自动过滤非 input_ids 列
dataset = [sample for sample in dataset]

def collator(data):
    return {
        "input_ids": [torch.tensor(d["input_ids"]) for d in data],
        "query": [d["query"] for d in data],
        "post_content": [d["post_content"] for d in data]
    }

# ==========================================
# 4. Training Loop
# ==========================================
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model,
    tokenizer,
    dataset=dataset,
    data_collator=collator,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 60,
    "repetition_penalty": 1.1,  # 增加重复惩罚，防止模型复读
}

print("Starting PPO training loop...")

for step, batch in enumerate(tqdm(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]
    
    # 1. Generate text
    with torch.no_grad():
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            return_prompt=False,
            **generation_kwargs
        )
    
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    
    # 2. Compute Rewards using RM
    # RM expects: "Post: {post}\nTL;DR: {summary}"
    rm_texts = []
    for post, response in zip(batch["post_content"], batch["response"]):
        # Match train_rm.py formatting
        rm_input = f"Post: {post}\nTL;DR: {response}{tokenizer.eos_token}"
        rm_texts.append(rm_input)
        
    # RM Inference
    # Trick: Use tokenizer with right padding for RM batch inference
    tokenizer.padding_side = "right"
    rm_inputs = tokenizer(
        rm_texts, 
        padding=True, 
        truncation=True, 
        max_length=1024, 
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        rm_outputs = reward_model(**rm_inputs)
        # Assuming Regression output (num_labels=1)
        rewards_tensor = rm_outputs.logits.squeeze(-1)
        
    # --- 奖励归一化与处理 (Reward Normalization & Scaling) ---
    # 1. 简单的奖励缩放 (Reward Scaling)
    # 很多时候 RM 输出的量级不一，可以乘以一个系数
    reward_scale = 1.0 
    rewards_tensor = rewards_tensor * reward_scale

    # 2. 奖励截断 (Reward Clipping)
    # 防止极端分数破坏训练，通常限制在 [-5, 5] 或 [-10, 10]
    rewards_tensor = torch.clamp(rewards_tensor, min=-5.0, max=5.0)

    # Convert back to list of tensors for PPO
    rewards = [r for r in rewards_tensor]
    
    # Restore padding side for Policy
    tokenizer.padding_side = "left"
    
    # 3. PPO Step
    stats = ppo_trainer.step(query_tensors, list(response_tensors), rewards)
    
    # 4. Logging & Diagnostics
    batch_reward = torch.stack(rewards).mean().item()
    
    print(f"\n[Step {step}] Mean Processed Reward: {batch_reward:.4f}")
    if step % 1 == 0:
        print(f"  Raw RM Scores (first 2): {rewards_tensor[:2].cpu().tolist()}")
        print(f"  Query (SFT Input): {batch['query'][0][-50:].replace(chr(10), ' ')}...") # Last 50 chars
        print(f"  Response: {batch['response'][0]}")
        print(f"  RM Input: {rm_texts[0][:50].replace(chr(10), ' ')}...")
        
    # Stop early for checking output
    if step >= 5:
        print("Stopping early for inspection.")
        break
