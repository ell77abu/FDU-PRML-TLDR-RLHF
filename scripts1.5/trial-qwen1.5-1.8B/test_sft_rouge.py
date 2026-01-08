
# --- 7. ROUGE 测试：对比训练前后的模型性能 ---
from datasets import load_metric

# 加载 ROUGE 指标
rouge = load_metric("rouge")

def evaluate_rouge(model, dataset, tokenizer):
    model.eval()
    preds, labels = [], []
    for item in dataset:
        prompt = item['prompt']
        label = item['label']
        
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(decoded_output)
        labels.append(label)

    results = rouge.compute(predictions=preds, references=labels)
    return results

# 对比训练前后的模型 ROUGE 分数
print("\n--- 训练前模型测试 ---")
# 加载原始模型
original_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    local_files_only=True,
).to(device)

original_rouge_results = evaluate_rouge(original_model, dataset_small['test'], tokenizer)
print(f"Original Model ROUGE Results: {original_rouge_results}")

print("\n--- 训练后模型测试 ---")
# 使用训练后的模型进行 ROUGE 测试
trained_rouge_results = evaluate_rouge(model, dataset_small['test'], tokenizer)
print(f"Trained Model ROUGE Results: {trained_rouge_results}")