import os

def clean_file(input_path, output_path, is_hanshu=False):
    print(f"Processing {input_path} -> {output_path}")
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        last_line_empty = False
        
        for line in fin:
            original_line = line
            stripped_line = line.strip()
            
            # Handle empty lines
            if not stripped_line:
                if not last_line_empty:
                    fout.write('\n')
                    last_line_empty = True
                continue
            
            last_line_empty = False
            
            if is_hanshu:
                # Hanshu specific logic
                if stripped_line.startswith("漢書卷"):
                    fout.write(f"## {stripped_line}\n")
                else:
                    # Body text: remove indentation
                    fout.write(f"{stripped_line}\n")
            else:
                # Shiji specific logic
                fout.write(f"{stripped_line}\n")

def main():
    base_dir = "data"
    output_dir = os.path.join(base_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process Shiji
    clean_file(
        os.path.join(base_dir, "史記.txt"),
        os.path.join(output_dir, "shiji_cleaned.txt"),
        is_hanshu=False
    )
    
    # Process Hanshu
    clean_file(
        os.path.join(base_dir, "漢書.txt"),
        os.path.join(output_dir, "hanshu_cleaned.txt"),
        is_hanshu=True
    )

if __name__ == "__main__":
    main()
