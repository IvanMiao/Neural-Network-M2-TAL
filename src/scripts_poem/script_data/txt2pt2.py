#!/usr/bin/env python3
"""
Usage example:
    python3 txt2trainpt.py --input_dir corpus_poems --out_pt data/processed/train_data.pt --vocab_json data/processed/vocab.json --seq_len 128 --stride 1
"""

import argparse
import json
import os
from collections import Counter
import torch

def build_vocab(texts):
    chars = set()
    for t in texts:
        chars.update(t)
    chars = sorted(chars)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {str(i): c for c, i in char2id.items()}
    return char2id, id2char

def texts_to_windows(texts, char2id, L, stride=1):
    windows = []
    for t in texts:
        ids = [char2id[c] for c in t if c in char2id]
        n = len(ids)
        if n < L:
            continue
        for i in range(0, n - L + 1, stride):
            win = ids[i : i + L]
            windows.append(win)
    return windows

def collect_texts(input_dir, encoding="utf-8"):
    texts = []
    files = []
    for root, _, fnames in os.walk(input_dir):
        for name in fnames:
            if not name.lower().endswith(".txt"):
                continue
            p = os.path.join(root, name)
            try:
                with open(p, "r", encoding=encoding, errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                print("warning: could not read", p, "->", e)
                continue
            if not text.endswith("\n"):
                text = text + "\n"
            texts.append(text)
            files.append(p)
    return texts, files

def save_vocab(char2id, id2char, path):
    d = {"char2id": char2id, "id2char": id2char}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def save_data(windows, out_pt):
    if len(windows) == 0:
        raise RuntimeError("No windows to save. Try lowering seq_len or lowering stride.")
    arr = torch.tensor(windows, dtype=torch.long)
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save(arr, out_pt)

def parse_args():
    p = argparse.ArgumentParser(description="Build train dataset from .txt files (char-level)")
    p.add_argument("--input_dir", default="corpus_poems", help="Directory with .txt files")
    p.add_argument("--out_pt", default="data/processed/train_data.pt", help="Output .pt path")
    p.add_argument("--vocab_json", default="data/processed/vocab.json", help="Output vocab json")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length for X (actual saved windows length will be seq_len + 1)")
    p.add_argument("--stride", type=int, default=1, help="Stride between windows (1 = fully overlapping)")
    p.add_argument("--encoding", default="utf-8", help="File encoding")
    return p.parse_args()

def main():
    args = parse_args()
    L = args.seq_len + 1
    texts, files = collect_texts(args.input_dir, args.encoding)
    print(f"Found {len(texts)} text files.")
    char2id, id2char = build_vocab(texts)
    print(f"Vocab size: {len(char2id)} chars")

    windows = texts_to_windows(texts, char2id, L, stride=args.stride)
    print(f"Created {len(windows)} windows (shape: ({len(windows)}, {L}))")

    print("Saving vocab to:", args.vocab_json)
    save_vocab(char2id, id2char, args.vocab_json)

    print("Saving data to:", args.out_pt)
    save_data(windows, args.out_pt)


if __name__ == "__main__":
    main()
