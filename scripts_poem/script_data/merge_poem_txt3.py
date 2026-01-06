#!/usr/bin/env python3
"""
Simple cleaning for poems corpus.
- Read all .txt files from a directory (default: corpus_poems)
- For each file: remove blank lines, join remaining lines into a single line (no extra spaces)
- Combine all cleaned poems into one output file, separated by a single blank line

Usage:
    python3 clean_poems.py --input_dir corpus_poems --out cleaned_poems.txt
"""

import os
import argparse


def collect_txt_paths(input_dir):
    paths = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith('.txt'):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, input_dir)
                paths.append((rel, full))
    paths.sort(key=lambda x: x[0])
    return paths


def clean_file(path, encoding='utf-8'):
    """Read file and return a single-line string with blank lines removed."""
    try:
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f'warning: could not read {path}: {e}')
        return None

    # strip whitespace and remove empty lines
    lines = [ln.strip() for ln in lines]
    lines = [ln for ln in lines if ln != '']

    if not lines:
        return None

    # join into one line without extra spaces (suitable for Chinese poems)
    merged = ''.join(lines)
    return merged


def clean_and_merge(input_dir='corpus_poems', out='cleaned_poems.txt', encoding='utf-8', include_filename=False):
    paths = collect_txt_paths(input_dir)
    if not paths:
        raise RuntimeError(f'No .txt files found in {input_dir}')

    cleaned = []
    for rel, full in paths:
        text = clean_file(full, encoding=encoding)
        if text is None:
            continue
        if include_filename:
            cleaned.append(f'=== {rel} ===\n' + text)
        else:
            cleaned.append(text)

    # write with exactly one blank line between poems (no extra trailing blank lines)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out, 'w', encoding=encoding) as f:
        for i, poem in enumerate(cleaned):
            f.write(poem)
            # end the poem line
            f.write('\n')
            # add single blank separator between poems
            if i < len(cleaned) - 1:
                f.write('\n')

    print(f'saved {out} ({len(cleaned)} poems)')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Clean poems: remove blank lines and merge into one file')
    p.add_argument('--input_dir', default='corpus_poems', help='Directory containing .txt files')
    p.add_argument('--out', default='cleaned_poems.txt', help='Output file path')
    p.add_argument('--encoding', default='utf-8', help='File encoding')
    p.add_argument('--include_filename', action='store_true', help='Include filename header before each cleaned poem')
    args = p.parse_args()

    clean_and_merge(input_dir=args.input_dir, out=args.out, encoding=args.encoding, include_filename=args.include_filename)
