import os
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
    """Analyze one-line-per-poem text file. Returns list of non-empty poem lines and their lengths."""
    print(f"\n--- Analyzing {filename} ---")
    lines = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("##"):
                    lines.append(s)

        if lines:
            lengths = np.array([len(l) for l in lines])
            print(f"Total poems: {len(lines)}")
            print(f"Min length: {np.min(lengths)}")
            print(f"Max length: {np.max(lengths)}")
            print(f"Mean length: {np.mean(lengths):.2f}")
            print(f"Median length: {np.median(lengths)}")
            print(f"90th percentile: {np.percentile(lengths, 90)}")
            print(f"95th percentile: {np.percentile(lengths, 95)}")
            print(f"99th percentile: {np.percentile(lengths, 99)}")
            return lines, lengths
        else:
            print("No valid poems found.")
            return [], np.array([])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [], np.array([])


# NOTE: Deletion of longest poems was removed per request. Analysis-only script retained.
# The function `filter_out_max_length` has been intentionally removed; use the analysis functions to inspect lengths.

def analyze_train_tensor():
    print("\n--- Analyzing train_data.pt ---")
    try:
        data = torch.load("./data/processed/train_data.pt")
        print(f"Tensor shape: {data.shape}")
        print(f"Datatype: {data.dtype}")
        print(f"Memory usage (MB): {data.element_size() * data.nelement() / 1024 / 1024:.2f}")
    except Exception as e:
        print(f"Error reading train_data.pt: {e}")

if __name__ == "__main__":
    analyze_vocab()

    # Analyze the merged poems file (analysis only; no deletions)
    poems_file = 'poems_all.txt'
    analyze_text_file(poems_file)

    analyze_train_tensor()
