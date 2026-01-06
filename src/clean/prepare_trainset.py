import os
import json
import torch
import glob
from collections import Counter

def clean_data(input_paths):
    cleaned_text = ""
    for path in input_paths:
        print(f"Processing {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip headers starting with ##
            if line.startswith("##"):
                continue
            
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


def encode_and_chunk(text, char2id, seq_len=512):
    # Text to id
    data = [char2id.get(char, char2id['<UNK>']) for char in text]
    
    # fixed length sequence
    chunks = []
    for i in range(0, len(data) - seq_len, seq_len // 2): # 50% overlap to add datas
        chunks.append(data[i:i + seq_len])
    
    # Check if we have enough data for at least one chunk
    if not chunks:
        print("Warning: Not enough data to create even one chunk!")
        return torch.tensor([], dtype=torch.long)

    return torch.tensor(chunks, dtype=torch.long)

if __name__ == "__main__":
    # Use glob to find all cleaned text files
    raw_data_paths = glob.glob("./data/processed/*_cleaned.txt")
    if not raw_data_paths:
        print("No cleaned data files found in ./data/processed/")
        exit(1)
        
    save_dir = "./data/processed"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Found files: {raw_data_paths}")
    print("Cleaning data...")
    text = clean_data(raw_data_paths)
    print(f"Total cleaned text length: {len(text)} characters")
    
    print("Building vocab...")
    char2id, id2char = build_vocab(text)
    
    print("Encoding text...")
    dataset = encode_and_chunk(text, char2id)
    
    # Save data
    if dataset.numel() > 0:
        torch.save(dataset, os.path.join(save_dir, "train_data.pt"))
        with open(os.path.join(save_dir, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump({"char2id": char2id, "id2char": id2char}, f, ensure_ascii=False)
        
        print(f"Preprocessing finished. Dataset shape: {dataset.shape}")
    else:
        print("Preprocessing failed: Empty dataset.")