import re
import sys
import argparse

def replace_inline_formulas(text):
    # 匹配 $$...$$ 形式的行内公式，且 $$ 后没有换行符
    # Step 1: 在 $$ 后没有 \n 的情况下，添加 \n
    text = re.sub(r'\$\$([^\n])', r'$$\n\1', text)
    
    # Step 2: 在 $$ 前没有 \n 的情况下，添加 \n
    text = re.sub(r'([^\n])\$\$', r'\1\n$$', text)
    
    return text

def process_markdown(input_file, output_file):
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 执行公式替换
        result = replace_inline_formulas(text)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        print(f"Successfully processed {input_file} and saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except PermissionError:
        print(f"Error: Permission denied when accessing files.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Replace inline LaTeX formulas in Markdown file with newlines.")
    parser.add_argument("input_file", help="Path to the input Markdown file")
    parser.add_argument("output_file", help="Path to the output Markdown file")
    
    args = parser.parse_args()
    
    # 调用处理函数
    process_markdown(args.input_file, args.output_file)