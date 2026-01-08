from datasets import load_dataset
import os

# 配置 HF 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

output_dir = "/workspace/pj-RL/datasets"
os.makedirs(output_dir, exist_ok=True)

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
