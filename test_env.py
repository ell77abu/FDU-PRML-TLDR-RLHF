try:
    from trl import GRPOTrainer, GRPOConfig
    print("✅ 恭喜！GRPO 模块已成功加载，你可以开始实验了。")
except ImportError:
    print("❌ 错误：TRL 版本依然不支持 GRPO。请检查是否安装到了正确的 Python 环境。")