"""
Usage:
    python3 clean_poems.py --input_dir corpus_poems --out cleaned_poems.txt
"""

from pathlib import Path
files = sorted(Path("corpus_poems").rglob("*.txt"))
with open("cleaned_poems.txt", "w", encoding="utf-8") as out:
    for i, p in enumerate(files):
        txt = ''.join(line.strip() for line in p.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())
        if not txt:
            continue
        out.write(txt)
        if i != len(files) - 1:
            out.write("\n\n")