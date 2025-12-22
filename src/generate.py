import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import json
import numpy as np

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
        
        # 预测概率
        preds = model.predict(curr_input, verbose=0)[0][-1]
        
        # Temperature Sampling
        preds = np.log(preds + 1e-10) / temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        # 随机选择下一个字
        next_id = np.random.choice(len(preds), p=preds)
        
        generated.append(next_id)
        if next_id == char2id.get('<EOS>', -1):
            break
            
    return "".join([id2char[str(idx)] for idx in generated])

if __name__ == "__main__":
    # 1. 加载模型
    model = keras.models.load_model("shiji_transformer.keras")
    
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