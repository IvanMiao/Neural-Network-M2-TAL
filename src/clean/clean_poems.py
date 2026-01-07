import re
from pathlib import Path


def contains_problematic_chars(line: str) -> bool:
    """检查一行是否包含需要过滤的字符"""
    # 检查 &KR
    if re.search(r'&KR\d+;', line):
        return True

    # 检查罕见的或不常用的 Unicode 字符
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
    """清理诗歌文件"""
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

    print(f"处理完成！")
    print(f"总行数: {total_lines}")
    print(f"保留行数: {kept_lines}")
    print(f"删除行数: {removed_lines}")
    print(f"删除比例: {removed_lines/total_lines*100:.2f}%")


def main():
    input_file = Path("data/data_poem/poems_all_cleaned.txt")
    output_file = Path("data/data_poem/poems_further_cleaned.txt")

    if not input_file.exists():
        print(f"错误：输入文件不存在：{input_file}")
        return

    print(f"开始清理文件：{input_file}")
    print(f"输出文件：{output_file}")
    print()

    clean_poem_file(input_file, output_file)

    print(f"\n清理后的文件已保存到：{output_file}")


if __name__ == "__main__":
    main()
