import os

# 配置 HF 镜像
os.environ['HF_ENDPOINT'] = 'https://mirrors.aliyun.com/huggingface'

output_dir = "/workspace/pj-RL/datasets"
os.makedirs(output_dir, exist_ok=True)
from datasets import load_dataset

# 1. CarperAI/openai_summarize_tldr
print("Processing CarperAI/openai_summarize_tldr...")
try:
    ds_tldr = load_dataset("CarperAI/openai_summarize_tldr")
    save_path = os.path.join(output_dir, "openai_summarize_tldr")
    ds_tldr.save_to_disk(save_path)
    print(f"Successfully saved to {save_path}")
except Exception as e:
    print(f"Error processing tldr dataset: {e}")

# 2. openai/summarize_from_feedback
print("\nProcessing openai/summarize_from_feedback (comparisons)...")
try:
    # 使用 'comparisons' 配置，这是之前代码中使用的
    ds_feedback = load_dataset("openai/summarize_from_feedback", "comparisons")
    save_path = os.path.join(output_dir, "summarize_from_feedback")
    ds_feedback.save_to_disk(save_path)
    print(f"Successfully saved to {save_path}")
except Exception as e:
    print(f"Error processing feedback dataset: {e}")

# 3. openai/summarize_from_feedback (axis)
print("\nProcessing openai/summarize_from_feedback (axis)...")
try:
    # 使用 'axis' 配置
    ds_axis = load_dataset("openai/summarize_from_feedback","axis") #, "axis"
    save_path = os.path.join(output_dir, "summarize_axis")
    ds_axis.save_to_disk(save_path)
    print(f"Successfully saved to {save_path}")
except Exception as e:
    print(f"Error processing axis dataset: {e}")

# from datasets import load_dataset
# ds = load_dataset("openai/summarize_from_feedback", "axis")