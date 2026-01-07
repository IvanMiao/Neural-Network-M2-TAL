#!/usr/bin/env bash
set -euo pipefail

echo "Step 1/4: clean raw poems -> corpus_clean_poem"
if [[ ! -d "corpus_brut_poem" ]]; then
  echo "warning: corpus_brut_poem not found; clean_poem_brut.py will do nothing"
fi
python3 clean_poem_brut.py

echo "Step 2/4: merge cleaned poems -> poems_all.txt"
if [[ ! -d "corpus_clean_poem" ]]; then
  echo "error: corpus_clean_poem not found; cleaning may have failed"
  exit 1
fi
python3 merge_poem_txt.py --input_dir corpus_clean_poem --out poems_all.txt

echo "Step 3/4: build vocab + dataset -> data/processed"
python3 prepare_trainset.py --include_poems_all

echo "Step 4/5: train model"
python3 train_poem.py

echo "Step 5/5: generate text"
python3 generate.py

echo "Pipeline finished."
