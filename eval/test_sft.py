import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载模型和tokenizer
ppo_model_path = "/workspace/pj-RL/experiments3/qwen3-ppo-merged5000"
tokenizer = AutoTokenizer.from_pretrained(ppo_model_path)
model = AutoModelForCausalLM.from_pretrained(ppo_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 确保模型使用GPU，如果可用
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. 定义你的输入（Prompt）
prompt = """So me and my now ex met online July 2013. From the start I knew he was a really disciplined individual because of his eating habits ( very strict; non GMO, only free range meats etc..), his education and his martial arts career.   Things were great up until 4 months ago, we would argue all the time about everything. We would have super intense arguments over the stupidest things like me wearing make up (he thought I looked hideous and super fake) or like me talking about something stressful in my life without warning him first that I was going to do that. He restricted my ability to talk about my stress in life to the weekends because he "couldn't handle my emotions all the time" and when it came down to discussing my stress or an argument, he would dissect every emotion to action to reaction to emotion. I just couldn't do that anymore, even after letting him know all I wanted him to do is just to listen, not to analyze in explicit detail everything I was feeling and my actions because of those feelings.   I even went to counseling for the last 3 weeks because he said my communication is poor and I don't understand the English Language ( even though it is my first language). I was just going to counseling by myself. He never offered to come along with me.   So I broke up with him yesterday, and it wasn't until I said " I realized through the counseling that I am just done trying to make us work" was he then willing to commit and go to counseling and "Do whatever it takes to keep you here with me" I didn't give in to that because I feel like if he really did want to make us work, he would have been that committed since the beginning of our problems.   But now I feel really guilty. I feel like I should have given him that opportunity. I don't know if it's just a post break up feeling or if I genuinely made a huge mistake.\n\nTL;DR:"""

# 3. 将Prompt进行编码
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 4. 生成文本
# 调整生成的参数
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=60,
    do_sample=False,
    top_p=0.9,
    temperature=0.8,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id
)

# 5. 解码生成的文本
generated_text = tokenizer.decode(output_ids[0, len(input_ids[0]):], skip_special_tokens=True)

# 6. 打印结果
print("Prompt: ", prompt)
print("Generated Text: ", generated_text)
# human label
# broke up with bf, now feeling really guilty and that I didn't give him the opportunity to help fix the relationship.
