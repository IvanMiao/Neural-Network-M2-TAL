import os
import re

INPUT_DIR = "corpus_p"
OUTPUT_DIR = "corpus_poems"

HEADER_SCAN_LINES = 120

PUNCT_FOR_TITLE_CHECK = set("，。；：？！、,.!?;:—…「」『』（）()《》〈〉【】[]{}“”‘’\"\'")
HEADER_HINTS = (
    "欽定", "四庫", "全書", "文集", "詩集", "巻", "卷", "目錄", "序", "序言", "校", "注",
    "撰", "編", "纂", "輯", "評", "譯", "增訂", "重修",
    "唐", "宋", "元", "明", "清"
)

def _has_punct(s: str) -> bool:
    return any(ch in PUNCT_FOR_TITLE_CHECK for ch in s)

def _strip_tags_and_notes(line: str) -> str:
    line = line.replace("¶", "")
    if "#" in line:
        line = line.split("#", 1)[0]
    line = re.sub(r"<[^>]*>", "", line)
    while True:
        new_line = re.sub(r"\([^()]*\)", "", line)
        if new_line == line:
            break
        line = new_line
    while True:
        new_line = re.sub(r"（[^（）]*）", "", line)
        if new_line == line:
            break
        line = new_line

    return line

def _looks_like_title_line(original_line: str) -> bool:
    if not (original_line.startswith("　　") or original_line.startswith("  ")):
        return False
    s = original_line.strip()
    if s == "":
        return False
    if len(s) > 30:
        return False
    if _has_punct(s):
        return False
    return True

def _compress_blank_lines(lines):
    out = []
    last_blank = False
    for x in lines:
        if x.strip() == "":
            if not last_blank:
                out.append("")
            last_blank = True
        else:
            out.append(x.rstrip())
            last_blank = False
    return out

def clean_and_split_poems(text: str):
    raw_lines = text.splitlines()
    poems = []
    cur = []
    seen_first_title = False

    for i, line in enumerate(raw_lines):
        ls = line.lstrip()
        if ls.startswith("#") or ls.startswith("+"):
            continue
        line = _strip_tags_and_notes(line)
        s = line.strip()
        if s == "":
            if seen_first_title and cur:
                cur.append("")
            continue
        if (not seen_first_title) and (i < HEADER_SCAN_LINES):
            if (len(s) <= 40) and (not _has_punct(s)) and any(h in s for h in HEADER_HINTS):
                continue
        if _looks_like_title_line(line):
            seen_first_title = True
            if cur:
                cur2 = _compress_blank_lines(cur)
                poem = "\n".join(cur2).strip()
                if poem:
                    poems.append(poem + "\n")
            cur = []
            continue

        if seen_first_title:
            cur.append(s)
    if cur:
        cur2 = _compress_blank_lines(cur)
        poem = "\n".join(cur2).strip()
        if poem:
            poems.append(poem + "\n")

    return poems

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_in = 0
    total_poems = 0

    for root, _, files in os.walk(INPUT_DIR):
        for name in files:
            if not name.lower().endswith(".txt"):
                continue

            in_path = os.path.join(root, name)
            rel_dir = os.path.relpath(root, INPUT_DIR)
            out_dir = os.path.join(OUTPUT_DIR, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()

            poems = clean_and_split_poems(raw)
            base = os.path.splitext(name)[0]
            for idx, poem in enumerate(poems, start=1):
                out_name = f"{base}__poem_{idx:04d}.txt"
                out_path = os.path.join(out_dir, out_name)
                with open(out_path, "w", encoding="utf-8") as w:
                    w.write(poem)
            total_in += 1
            total_poems += len(poems)

    print("Done for poem, total txt：", total_in, "\nhave：", total_poems, "poem")

if __name__ == "__main__":
    main()
