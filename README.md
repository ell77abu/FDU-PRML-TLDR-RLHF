# FDU-PRML-PJ Qwen3 RLHF Summarization

面向 TL;DR 摘要任务的强化学习奖励设计优化项目。

> 训练流程 (SFT → 奖励模型（RM）→ GRPO) 全部在本地 Qwen3-1.7B 上运行; 数据集[CarperAI/openai_summarize_tldr](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/viewer/default/train?views%5B%5D=train)和[openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback/viewer/axis?views%5B%5D=axis_validation)均下载到本地。

## 目录结构
- scripts3/ baseline代码
  - sft.py：监督微调（SFT)
  - rm.py：奖励模型训练，使用 pairwise 选择数据（chosen/rejected）。
  - grpo.py：基于 RM 奖励的 GRPO 强化训练(LoRA)。
  - ppo-valuehead.py：PPO 的 训练代码。
  - merge_lora.py：合并 LoRA 权重到基座模型。
  - normalize_rm.py：奖励归一化辅助脚本。
- scripts-improve/ improved代码
- datasets/ 下载到本地的数据集
  - openai_summarize_tldr/：CarperAI/openai_summarize_tldr
  - summarize_from_feedback/：openai/summarize_from_feedback comparison
  - summarize_axis/：openai/summarize_from_feedback axis
- experiments3/：SFT、RM、GRPO 训练结果、检查点示例。
- models/Qwen3-1.7B/：本地基座模型。
- eval/：奖励模型准确率与摘要 ROUGE评测脚本


## 环境准备
1) 使用提供的 conda 环境：
```bash
conda env create -f environment.yml
conda activate prml_pj
```
2) 硬件设备：单张 24GB+ NVIDIA GPU，已安装 CUDA 11.7/12.1 兼容驱动。

## 数据准备(本地)
- SFT数据集：[CarperAI/openai_summarize_tldr](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr/viewer/default/train?views%5B%5D=train)
- RM数据集：[openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback/viewer/axis?views%5B%5D=axis_validation)


## 训练运行流程
### 1) 监督微调（SFT）
脚本：scripts3/sft.py
```bash
python scripts3/sft.py 
```
### 2) 奖励模型（RM）
脚本：scripts3/rm.py
```bash
python scripts3/rm.py
```
### 3) GRPO训练
脚本：scripts3/grpo.py
```bash
python scripts3/grpo.py
```
## 评测工具
- 运行 GPU/GRPO导入：
```bash
python test_gpu.py
python test_env.py
```
- 奖励模型与 ROUGE 测试：eval/ 目录下的 `test_rm_*.py`、`test_rouge.py`（按需修改路径后运行）。
## 日志与模型产出
- 采用wandb可视化记录实验结果

