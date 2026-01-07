import re

file_path = './data/data_prose/processed/hanshu_cleaned.txt'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove 〈...〉 including content
content = re.sub(r'〈[^〉]*〉', '', content)

# 2. Remove （...） and its content
content = re.sub(r'（[^）]*）', '', content)

# 3. Remove 〔...〕 but keep content
content = re.sub(r'〔([^〕]*)〕', r'\1', content)

# 4. Remove lines containing any remaining brackets
lines = content.split('\n')
original_count = len(lines)
lines = [line for line in lines if not re.search(r'[〈〉（）〔〕]', line)]
removed_count = original_count - len(lines)
print(f"Removed {removed_count} lines containing leftover brackets.")
content = '\n'.join(lines)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"processed {file_path}")
