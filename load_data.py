import os
from datasets import load_dataset

# 1. 配置镜像源（国内必须）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 2. 设置路径
dataset_name = "openai/summarize_from_feedback"
subset = "comparisons"  # RM训练通常使用这个子集
local_save_path = "./datasets/rm_local"

print(f"开始下载数据集: {dataset_name}...")

# 3. 加载并缓存
# cache_dir 是临时存放下载原始文件的地方
dataset = load_dataset(
    dataset_name, 
    subset, 
    cache_dir="./huggingface_cache", 
    trust_remote_code=True
)

# 4. 将处理好的 Dataset 格式永久保存到指定目录
# 这会把数据转化为 Arrow 格式，方便下次免网络直接读取
dataset.save_to_disk(local_save_path)

print(f"✅ 数据集已成功保存至本地文件夹: {local_save_path}")