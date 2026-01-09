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
    # Define paths
    prose_paths = glob.glob("./data/data_prose/processed/*_cleaned.txt")
    poetry_paths = ["./data/data_poem/poems_all_cleaned.txt"]
    
    save_dir = "./data/processed"
    os.makedirs(save_dir, exist_ok=True)

    print("Cleaning Prose data...")
    prose_text = clean_data(prose_paths)
    print(f"Prose length: {len(prose_text)} chars")

    print("Cleaning Poetry data...")
    poetry_text = clean_data(poetry_paths)
    print(f"Poetry length: {len(poetry_text)} chars")
    
    full_text = prose_text + poetry_text
    print(f"Total combined length: {len(full_text)} chars")

    print("Building unified vocab...")
    char2id, id2char = build_vocab(full_text)
    
    # Save vocab
    with open(os.path.join(save_dir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump({"char2id": char2id, "id2char": id2char}, f, ensure_ascii=False)

    print("Encoding Prose dataset...")
    prose_dataset = encode_and_chunk(prose_text, char2id)
    if prose_dataset.numel() > 0:
        torch.save(prose_dataset, os.path.join(save_dir, "train_prose.pt"))
        print(f"Saved train_prose.pt: {prose_dataset.shape}")
    else:
        print("Warning: Prose dataset is empty")

    print("Encoding Poetry dataset...")
    poetry_dataset = encode_and_chunk(poetry_text, char2id)
    if poetry_dataset.numel() > 0:
        torch.save(poetry_dataset, os.path.join(save_dir, "train_poetry.pt"))
        print(f"Saved train_poetry.pt: {poetry_dataset.shape}")
    else:
        print("Warning: Poetry dataset is empty")
