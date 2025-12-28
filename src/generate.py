import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import json
import numpy as np

# 导入自定义层以确保 Keras 能够反序列化它们
from train.train import TokenAndPositionEmbedding, TransformerBlock

def generate_text(model, start_str, char2id, id2char, gen_len=100, temp=1.0):
    # 将起始文本转换为 ID
    input_ids = [char2id.get(c, char2id['<UNK>']) for c in start_str]
    generated = input_ids[:]

    for _ in range(gen_len):
        # 准备输入，确保长度不超过模型允许的 seq_len (127)
        x = generated[-127:]
        if len(x) < 127:
            pad_id = char2id.get('<PAD>', 0)
            x = [pad_id] * (127 - len(x)) + x
        
        curr_input = np.array([x])
        
        # 预测结果现在是 Logits (未经过 Softmax)，可能包含负数
        logits = model.predict(curr_input, verbose=0)[0][-1]
        
        # 1. Temperature Scaling (直接除以 temp)
        logits = logits / temp
        
        # 2. 手动实现 Softmax (增加数值稳定性)
        # 减去最大值防止 exp 溢出
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
        
        # 3. 随机选择
        # 再次归一化以防浮点误差导致和不完全为1
        probs = probs / np.sum(probs)
        next_id = np.random.choice(len(probs), p=probs)
        
        generated.append(next_id)
        if next_id == char2id.get('<EOS>', -1):
            break
            
    return "".join([id2char[str(idx)] for idx in generated])

if __name__ == "__main__":
    # 1. 加载模型
    model_path = "best_model.keras" 
    
    if not os.path.exists(model_path):
        model_path = "wenyan_transformer.keras"

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    # 2. 加载词表
    with open("./data/processed/vocab.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        char2id = vocab_data['char2id']
        id2char = vocab_data['id2char']

    # 3. 进行实验
    prompts = ["黄帝者", "太史公曰", "项羽乃"]
    for p in prompts:
        print(f"\n--- Prompt: {p} ---")
        # temp 越高，生成越随机；temp 越低，生成越保守
        result = generate_text(model, p, char2id, id2char, gen_len=50, temp=0.8)
        print(result)