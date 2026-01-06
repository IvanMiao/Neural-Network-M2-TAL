import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import json
import numpy as np

# 导入自定义层以确保 Keras 能够反序列化它们
from train.train import TokenAndPositionEmbedding, TransformerBlock

def generate_text(model, start_str, char2id, id2char, gen_len=100, temp=1.0, top_k=50, repetition_penalty=1.2, seq_len=511):
    """生成文言文本
    
    Args:
        model: 训练好的模型
        start_str: 起始文本
        char2id: 字符到ID的映射
        id2char: ID到字符的映射
        gen_len: 生成长度
        temp: 温度参数 (越高越随机，越低越保守)
        top_k: Top-K 采样参数 (0=禁用，只保留概率最高的K个token)
        repetition_penalty: 重复惩罚参数 (>1.0 降低重复概率)
        seq_len: 模型序列长度 (应与训练时的 seq_len 一致)
    """
    # 将起始文本转换为 ID
    input_ids = [char2id.get(c, char2id['<UNK>']) for c in start_str]
    generated = input_ids[:]

    for _ in range(gen_len):
        # 准备输入，确保长度不超过模型允许的 seq_len
        x = generated[-(seq_len):]
        if len(x) < seq_len:
            pad_id = char2id.get('<PAD>', 0)
            x = [pad_id] * (seq_len - len(x)) + x
        
        curr_input = np.array([x])
        
        # 预测结果现在是 Logits (未经过 Softmax)，可能包含负数
        logits = model.predict(curr_input, verbose=0)[0][-1]
        
        # 0. Repetition Penalty (重复惩罚)
        # 对已经生成的 token 进行惩罚
        if repetition_penalty != 1.0:
            for token_id in set(generated):
                if token_id < len(logits):
                    if logits[token_id] < 0:
                        logits[token_id] *= repetition_penalty
                    else:
                        logits[token_id] /= repetition_penalty

        # 1. Temperature Scaling
        logits = logits / temp
        
        # 2. Top-K 过滤：只保留概率最高的 K 个 token
        if top_k > 0 and top_k < len(logits):
            indices_to_remove = np.argsort(logits)[:-top_k]
            logits[indices_to_remove] = -float('inf')
        
        # 3. 手动实现 Softmax (增加数值稳定性)
        exp_preds = np.exp(logits - np.max(logits))
        probs = exp_preds / np.sum(exp_preds)
        
        # 4. 随机选择
        probs = probs / np.sum(probs)  # 再次归一化以防浮点误差
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
        # 增加 repetition_penalty 防止复读
        result = generate_text(model, p, char2id, id2char, gen_len=50, temp=0.8, repetition_penalty=1.2)
        print(result)