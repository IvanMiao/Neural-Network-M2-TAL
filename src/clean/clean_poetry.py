import os
import re

def clean_poetry_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    
    # 匹配括号及其内容 (用于去除注释/补注)
    # 模式说明：
    # \(.*?\) : 匹配英文/半角括号
    # （.*?）: 匹配中文/全角括号 (如果存在)
    # 注意：需要处理跨行的情况与嵌套的情况比较复杂，
    # 但在此特定数据集中，大部分注释都在一行内或者以(...)形式分段。
    # 我们先使用简单的非贪婪匹配。如果要处理嵌套，可能需要循环去除。
    annotation_pattern = re.compile(r'\(.*?\)')
    
    # 匹配页面标记 <pb:...>
    pb_pattern = re.compile(r'<pb:.*?>')
    # 匹配 Mandoku 实例实体 (如 &KR2664;)
    entity_pattern = re.compile(r'&[A-Z0-9*]+\;')
    
    for line in lines:
        # 保留原始缩进信息用于判断，但先去除换行符
        original_line_content = line.strip('\n').strip('\r')
        
        # 1. 跳过元数据行 (# 开头)
        if original_line_content.startswith('#'):
            continue
            
        # 2. 关键启发式规则：跳过以空格（半角或全角）开头的行
        # 这些通常是标题、卷名等
        # 当遇到这些行时，意味着上一首诗结束，下一首（或下一卷）即将开始
        # 我们插入一个分隔符
        if original_line_content.startswith(' ') or original_line_content.startswith('　') or "欽定四庫全書" in original_line_content:
            # 只有当之前已经有正文内容，且最后一行不是分隔符时，才插入分隔符
            if cleaned_lines and cleaned_lines[-1] != "----------------":
                cleaned_lines.append("----------------")
            continue
            
        # 3. 现在的行应该就是正文（或者包含正文的混合行）
        # 开始清洗内容
        content = original_line_content
        
        # 去除页面标记
        content = pb_pattern.sub('', content)
        # 去除段落符
        content = content.replace('¶', '')
        # 去除 Mandoku 实体
        content = entity_pattern.sub('', content)
        # 去除括号注释
        # 可能有多个注释，全部替换
        content = annotation_pattern.sub('', content)
        
        # 4. 去除首尾空白
        content = content.strip()
        
        # 5. 如果清洗后还有内容，且不是纯标点或异常符号，则保留
        if content:
            # 再次检查是否只剩下括号（防止 regex 没处理干净嵌套）
            if content.startswith(')') and len(content) < 5:
                # 有时候 regex 替换后可能剩下孤立的括号，简单过滤
                continue
                
            cleaned_lines.append(content)

    # 写入输出文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Cleaned poetry saved to {output_path}")
    print(f"Extracted {len(cleaned_lines)} lines.")

if __name__ == "__main__":
    input_file = "data/KR4c0012_002.txt"
    output_file = "data/processed/poetry_cleaned.txt"
    clean_poetry_file(input_file, output_file)
