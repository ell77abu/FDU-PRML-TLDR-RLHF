# import os
# import torch
# from datasets import load_from_disk
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
# from trl.experimental.ppo.ppo_trainer import PolicyAndValueWrapper
# from peft import LoraConfig
# from types import SimpleNamespace

# # ===============================
# # 0. 基础配置
# # ===============================
# device = "cuda"
# torch_dtype = torch.float16


# # patch PolicyAndValueWrapper to expose gradient checkpoint toggles (needed by unwrap_model_for_generation)
# def _gc_disable(self):
#     if hasattr(self.policy, "gradient_checkpointing_disable"):
#         self.policy.gradient_checkpointing_disable()
#     if hasattr(self.value_model, "gradient_checkpointing_disable"):
#         self.value_model.gradient_checkpointing_disable()


# def _gc_enable(self):
#     if hasattr(self.policy, "gradient_checkpointing_enable"):
#         self.policy.gradient_checkpointing_enable()
#     if hasattr(self.value_model, "gradient_checkpointing_enable"):
#         self.value_model.gradient_checkpointing_enable()


# PolicyAndValueWrapper.gradient_checkpointing_disable = _gc_disable
# PolicyAndValueWrapper.gradient_checkpointing_enable = _gc_enable

# # monkey patch forward to always return an object with logits (and optional past_key_values)
# def _pv_forward(self, **kwargs):
#     policy_out = self.policy(**kwargs)
#     # policy_out may be tuple from ValueHead wrapper; extract logits and pkv
#     if isinstance(policy_out, tuple):
#         logits = policy_out[0]
#         pkv = policy_out[3] if len(policy_out) > 3 else None
#         policy_ns = SimpleNamespace(logits=logits, past_key_values=pkv)
#     else:
#         policy_ns = policy_out
#     # value model output already returned separately
#     output = self.critic_backbone(**kwargs)
#     logits_v = self.value_model.score(output.hidden_states[-1])
#     return policy_ns, logits_v


# PolicyAndValueWrapper.forward = _pv_forward

# sft_model_path = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
# rm_model_path  = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"

# # ===============================
# # 1. Tokenizer（修 regex）
# # ===============================
# tokenizer = AutoTokenizer.from_pretrained(
#     sft_model_path,
#     trust_remote_code=True,
#     fix_mistral_regex=True,
# )
# tokenizer.pad_token = tokenizer.eos_token

# # ===============================
# # 2. Policy Model + LoRA
# # ===============================
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
#     sft_model_path,
#     peft_config=lora_config,
#     trust_remote_code=True,
#     dtype=torch_dtype,
#     device_map="auto",
# )
# policy_model.config.use_cache = False

# # ---------- 【修改 1】确保生成config存在 ----------
# if not hasattr(policy_model, "generation_config"):
#     policy_model.generation_config = policy_model.pretrained_model.generation_config

# # ---------- 【修改 2】强制模型返回 ModelOutput 而不是 tuple，并输出 hidden_states ----------
# policy_model.config.return_dict = True
# policy_model.config.output_hidden_states = True
# policy_model.pretrained_model.config.return_dict = True
# policy_model.pretrained_model.config.output_hidden_states = True
# # ensure v_head uses last hidden_state
# policy_model.config.output_attentions = False
# policy_model.pretrained_model.config.output_attentions = False

# # enable gradient checkpointing to save memory
# if hasattr(policy_model.pretrained_model, "gradient_checkpointing_enable"):
#     policy_model.pretrained_model.gradient_checkpointing_enable()

# # expose is_gradient_checkpointing flag if missing
# if not hasattr(policy_model, "is_gradient_checkpointing"):
#     policy_model.is_gradient_checkpointing = getattr(policy_model.pretrained_model, "is_gradient_checkpointing", False)

# # ===============================
# # 3. Reference Model（冻结）
# # ===============================
# ref_model = create_reference_model(policy_model.pretrained_model)
# ref_model.eval()
# for p in ref_model.parameters():
#     p.requires_grad = False

# # ===============================
# # 4. Reward Model（冻结）
# # ===============================
# reward_model = AutoModelForSequenceClassification.from_pretrained(
#     rm_model_path,
#     trust_remote_code=True,
#     dtype=torch_dtype,
#     device_map="auto",
# )
# reward_model.eval()
# for p in reward_model.parameters():
#     p.requires_grad = False

# # ===============================
# # 5. Value model wrapper（PPO 需要 score 接口）
# # ===============================
# class ValueModelWrapper(torch.nn.Module):
#     def __init__(self, policy):
#         super().__init__()
#         self.policy = policy
#         self.v_head = policy.v_head
#         self.base_model_prefix = policy.pretrained_model.base_model_prefix
#         setattr(self, self.base_model_prefix, policy.pretrained_model)

#     def score(self, hidden_states):
#         return self.v_head(hidden_states).squeeze(-1)


# value_model = ValueModelWrapper(policy_model)

# # ===============================
# # 5. PPO Config（0.26.2）
# # ===============================
# ppo_config = PPOConfig(
#     batch_size=1,
#     mini_batch_size=1,
#     gradient_accumulation_steps=1,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     learning_rate=1e-5,
#     num_ppo_epochs=1,
#     response_length=32,
# )

# # ===============================
# # 6. Dataset（tokenized 供 PPOTrainer 使用）
# # ===============================
# raw_dataset = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")["train"]


# def preprocess(example):
#     prompt = f"{example['info']['post']}\n\nTL;DR:"
#     toks = tokenizer(
#         prompt,
#         truncation=True,
#         max_length=128,
#         return_attention_mask=True,
#     )
#     return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}


# ppo_train_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
# ppo_train_dataset = ppo_train_dataset.select(range(50))
# ppo_train_dataset.set_format(type="torch")
# ppo_eval_dataset = ppo_train_dataset.select(range(5))

# # ===============================
# # 7. PPOTrainer（需提供 reward_model / dataset / value_model）
# # ===============================
# ppo_trainer = PPOTrainer(
#     ppo_config,
#     tokenizer,
#     policy_model,
#     ref_model,
#     reward_model,
#     ppo_train_dataset,
#     value_model,
#     eval_dataset=ppo_eval_dataset,
# )

# # ===============================
# # 8. 直接使用 PPOTrainer 内部 train
# # ===============================
# print("\n🚀 Starting PPO training...\n")
# ppo_trainer.train()
# print("\n✅ Finished PPO training.")




import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.experimental.ppo.ppo_trainer import PolicyAndValueWrapper
from peft import LoraConfig
from types import SimpleNamespace

# ===============================
# 0. 基础配置
# ===============================
device = "cuda"
torch_dtype = torch.float16

# patch PolicyAndValueWrapper to expose gradient checkpoint toggles
def _gc_disable(self):
    if hasattr(self.policy, "gradient_checkpointing_disable"):
        self.policy.gradient_checkpointing_disable()
    if hasattr(self.value_model, "gradient_checkpointing_disable"):
        self.value_model.gradient_checkpointing_disable()

def _gc_enable(self):
    if hasattr(self.policy, "gradient_checkpointing_enable"):
        self.policy.gradient_checkpointing_enable()
    if hasattr(self.value_model, "gradient_checkpointing_enable"):
        self.value_model.gradient_checkpointing_enable()

PolicyAndValueWrapper.gradient_checkpointing_disable = _gc_disable
PolicyAndValueWrapper.gradient_checkpointing_enable = _gc_enable

# monkey patch forward to always return an object with logits (and optional past_key_values)
def _pv_forward(self, **kwargs):
    policy_out = self.policy(**kwargs)
    if isinstance(policy_out, tuple):
        logits = policy_out[0]
        pkv = policy_out[3] if len(policy_out) > 3 else None
        policy_ns = SimpleNamespace(logits=logits, past_key_values=pkv)
    else:
        policy_ns = policy_out
    output = self.critic_backbone(**kwargs)
    logits_v = self.value_model.score(output.hidden_states[-1])
    return policy_ns, logits_v

PolicyAndValueWrapper.forward = _pv_forward

sft_model_path = "/workspace/pj-RL/experiments3/qwen3-sft/final_checkpoint"
rm_model_path  = "/workspace/pj-RL/experiments3/qwen3-rm/final_rm"

# ===============================
# 1. Tokenizer（修 regex）
# ===============================
tokenizer = AutoTokenizer.from_pretrained(
    sft_model_path,
    trust_remote_code=True,
    fix_mistral_regex=True,
)
tokenizer.pad_token = tokenizer.eos_token

# ===============================
# 2. Policy Model + LoRA
# ===============================
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
    dtype=torch_dtype,
    device_map="auto",
)
policy_model.config.use_cache = False

# ---------- 确保生成config存在 ----------
if not hasattr(policy_model, "generation_config"):
    policy_model.generation_config = policy_model.pretrained_model.generation_config

# ---------- 强制输出 dict + hidden_states ----------
policy_model.config.return_dict = True
policy_model.config.output_hidden_states = True
policy_model.pretrained_model.config.return_dict = True
policy_model.pretrained_model.config.output_hidden_states = True
policy_model.config.output_attentions = False
policy_model.pretrained_model.config.output_attentions = False

if hasattr(policy_model.pretrained_model, "gradient_checkpointing_enable"):
    policy_model.pretrained_model.gradient_checkpointing_enable()

if not hasattr(policy_model, "is_gradient_checkpointing"):
    policy_model.is_gradient_checkpointing = getattr(policy_model.pretrained_model, "is_gradient_checkpointing", False)

# ===============================
# 3. Reference Model（冻结）
# ===============================
ref_model = create_reference_model(policy_model.pretrained_model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ===============================
# 4. Reward Model（冻结）
# ===============================
reward_model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_path,
    trust_remote_code=True,
    dtype=torch_dtype,
    device_map="auto",
)
reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad = False

# ===============================
# 5. Value model wrapper
# ===============================
class ValueModelWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.v_head = policy.v_head
        self.base_model_prefix = policy.pretrained_model.base_model_prefix
        setattr(self, self.base_model_prefix, policy.pretrained_model)

    def score(self, hidden_states):
        return self.v_head(hidden_states).squeeze(-1)

value_model = ValueModelWrapper(policy_model)

# ===============================
# 5. PPO Config
# ===============================
ppo_config = PPOConfig(
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=1e-5,
    num_ppo_epochs=1,
    response_length=32,
)

# ===============================
# 6. Dataset（只用 5 条进行测试）
# ===============================
raw_dataset = load_from_disk("/workspace/pj-RL/datasets/summarize_from_feedback")["train"]

def preprocess(example):
    prompt = f"{example['info']['post']}\n\nTL;DR:"
    toks = tokenizer(
        prompt,
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    )
    return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "prompt": prompt}

# 只取前5条数据
test_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)
test_dataset = test_dataset.select(range(5))
test_dataset.set_format(type="torch")

# ===============================
# 7. 使用 policy + RM 测试生成内容和奖励
# ===============================
print("\n🚀 Testing generation and RM scoring for 5 examples...\n")

for i, batch in enumerate(test_dataset):
    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
    prompt_text = batch["prompt"]

    # 生成文本
    with torch.no_grad():
        outputs = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
        )
    gen_text = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)

    # 奖励模型打分
    with torch.no_grad():
        rm_inputs = tokenizer(gen_text, return_tensors="pt").to(device)
        reward_score = reward_model(**rm_inputs).logits.squeeze().item()

    print(f"Example {i+1}")
    print(f"Prompt: {prompt_text}")
    print(f"Generated: {gen_text}")
    print(f"Reward score: {reward_score}")
    print("-" * 80)

print("\n✅ Finished testing.")
