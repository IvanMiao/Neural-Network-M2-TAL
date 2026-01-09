import re
from pathlib import Path


def contains_problematic_chars(line: str) -> bool:
    # check &KR
    if re.search(r'&KR\d+;', line):
        return True

    problematic_chars = [
        '\u4A0F',  # 𤨏
        '\U00024A0F',  # 𤨏 (另一种表示)
        '\U0002466F',  # 𤙯
        '\U00024613',  # 𤘓
        '\U000244E3',  # 𤓣
        '\U00024923',  # 𤤣
        '\U00024663',  # 𤙣
        '\U00024627',  # 𤘧
    ]

    for char in problematic_chars:
        if char in line:
            return True

    if re.search(r'[\uE000-\uF8FF]|[\U000F0000-\U000FFFFD]|[\U00100000-\U0010FFFD]', line):
        return True

    if re.search(r'[\U00020000-\U0002A6DF]', line):
        for char in line:
            code_point = ord(char)
            if 0x20000 <= code_point <= 0x2A6DF:
                pass

    return False


def clean_poem_file(input_file: Path, output_file: Path):
    total_lines = 0
    kept_lines = 0
    removed_lines = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            total_lines += 1
            line = line.rstrip('\n')

            if line and not contains_problematic_chars(line):
                f_out.write(line + '\n')
                kept_lines += 1
            else:
                if line:  # 只计算非空行
                    removed_lines += 1

    print(f"Lines: {total_lines}")
    print(f"Reserved lines: {kept_lines}")
    print(f"Deleted lines: {removed_lines}")
    print(f"Delete propotion: {removed_lines/total_lines*100:.2f}%")


def main():
    input_file = Path("data/data_poem/poems_all_cleaned.txt")
    output_file = Path("data/data_poem/poems_further_cleaned.txt")

    print(f"Cleaning: {input_file}")
    print(f"Output file: {output_file}")
    print()

    clean_poem_file(input_file, output_file)


if __name__ == "__main__":
    main()
