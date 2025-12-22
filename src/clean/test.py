import os
import json
import torch
from collections import Counter

def clean_shiji(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 1. Jump tables
    content = lines[134:] 
    
    cleaned_text = ""
    for line in content:
        line = line.strip()
        if not line:
            continue
        # Add more 过滤逻辑
        cleaned_text += line + "\n"
    
    return cleaned_text


def build_vocab(text, vocab_size=8000):
    counter = Counter(text)
    # High-freq words
    most_common = counter.most_common(vocab_size - 4)
    tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] + [char for char, _ in most_common]
    char2id = {char: i for i, char in enumerate(tokens)}
    id2char = {i: char for i, char in enumerate(tokens)}
    return char2id, id2char


def encode_and_chunk(text, char2id, seq_len=128):
    # Text to id
    data = [char2id.get(char, char2id['<UNK>']) for char in text]
    
    # fixed length sequence
    chunks = []
    for i in range(0, len(data) - seq_len, seq_len // 2): # 50% overlap to add datas
        chunks.append(data[i:i + seq_len])
    
    return torch.tensor(chunks, dtype=torch.long)

if __name__ == "__main__":
    raw_data_path = "./data/史记.txt"
    save_dir = "./data/processed"
    os.makedirs(save_dir, exist_ok=True)

    print("Cleaning data...")
    text = clean_shiji(raw_data_path)
    
    print("Building vocab...")
    char2id, id2char = build_vocab(text)
    
    print("Encoding text...")
    dataset = encode_and_chunk(text, char2id)
    
    # Save data
    torch.save(dataset, os.path.join(save_dir, "train_data.pt"))
    with open(os.path.join(save_dir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump({"char2id": char2id, "id2char": id2char}, f, ensure_ascii=False)
    
    print(f"Preprocessing finished. Dataset shape: {dataset.shape}")