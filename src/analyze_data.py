import json
import torch
import numpy as np

def analyze_vocab():
    print("--- Vocab Analysis ---")
    try:
        with open("./data/processed/vocab.json", 'r', encoding='utf-8') as f:
            vocab = json.load(f)
            char2id = vocab.get('char2id', {})
            print(f"Vocab size: {len(char2id)}")
            print(f"Sample items: {list(char2id.items())[:10]}")
    except Exception as e:
        print(f"Error reading vocab: {e}")

def analyze_text_file(filename):
    print(f"\n--- Analyzing {filename} ---")
    lengths = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("##"): # Skip empty and headers if any
                    lengths.append(len(line))
        
        if lengths:
            lengths = np.array(lengths)
            print(f"Total non-empty lines: {len(lengths)}")
            print(f"Min length: {np.min(lengths)}")
            print(f"Max length: {np.max(lengths)}")
            print(f"Mean length: {np.mean(lengths):.2f}")
            print(f"Median length: {np.median(lengths)}")
            print(f"90th percentile: {np.percentile(lengths, 90)}")
            print(f"95th percentile: {np.percentile(lengths, 95)}")
            print(f"99th percentile: {np.percentile(lengths, 99)}")
        else:
            print("No valid lines found.")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

def analyze_combined_files(filenames):
    print(f"\n--- Analyzing Combined: {' + '.join([f.split('/')[-1] for f in filenames])} ---")
    all_lengths = []
    total_chars = 0
    total_lines = 0
    
    for filename in filenames:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("##"):
                        length = len(line)
                        all_lengths.append(length)
                        total_chars += length
                        total_lines += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return

    if all_lengths:
        lengths = np.array(all_lengths)
        print(f"Total files: {len(filenames)}")
        print(f"Total non-empty lines: {total_lines}")
        print(f"Total characters: {total_chars}")
        print(f"Min length: {np.min(lengths)}")
        print(f"Max length: {np.max(lengths)}")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Median length: {np.median(lengths)}")
        print(f"90th percentile: {np.percentile(lengths, 90)}")
        print(f"95th percentile: {np.percentile(lengths, 95)}")
        print(f"99th percentile: {np.percentile(lengths, 99)}")
    else:
        print("No valid lines found in combined files.")

def analyze_train_tensor():
    print("\n--- Analyzing train_prose.pt ---")
    try:
        data = torch.load("./data/processed/train_prose.pt")
        print(f"Tensor shape: {data.shape}")
        print(f"Datatype: {data.dtype}")
        print(f"Memory usage (MB): {data.element_size() * data.nelement() / 1024 / 1024:.2f}")
    except Exception as e:
        print(f"Error reading train_prose.pt: {e}")

if __name__ == "__main__":
    analyze_vocab()
    
    prose_files = [
        "./data/data_prose/processed/shiji_cleaned.txt",
        "./data/data_prose/processed/hanshu_cleaned.txt"
    ]
    poem_files = ["./data/data_poem/poems_all_cleaned.txt"]
    analyze_combined_files(poem_files)
    analyze_combined_files(prose_files)
    
    analyze_train_tensor()
