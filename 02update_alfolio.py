import os
import sys
import re

def update_post(post_path):
    if not os.path.exists(post_path):
        print(f"错误: 找不到文件 {post_path}")
        return

    base_name = os.path.basename(post_path)
    # 获取不含后缀的文件名，用于路径拼接
    name_without_ext = os.path.splitext(base_name)[0]

    with open(post_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # --- 1. 修复 Markdown 链接中的 | 符号 ---
    def link_pipe_fix(match):
        title, link = match.group(1), match.group(2)
        fixed_title = title.replace('|', r'\|')
        fixed_link = link.replace('|', '%7C')
        return f'[{fixed_title}]({fixed_link})'
    
    content = re.sub(r'\[(.*?)\]\((.*?)\)', link_pipe_fix, content)

    # --- 2. 修改 Markdown 图片 + 响应式类 (增加重复检查) ---
    def md_img_replace(match):
        alt = match.group(1)
        src = match.group(2)
        full_match = match.group(0)
        
        # 检查是否已有 .img-fluid 标记
        # 这里的检查范围扩大到匹配项后的 60 个字符，防止中间有空格干扰
        post_match_text = content[match.end():match.end()+60]
        if '{: .img-fluid' in post_match_text:
            return full_match
        
        new_src = src
        if '/images/' not in src:
            new_src = f"/images/{name_without_ext}/{os.path.basename(src)}"
            
        return f"![{alt}]({new_src}){{: .img-fluid data-zoomable=\"\"}}"

    content = re.sub(r'!\[(.*?)\]\((.*?)\)', md_img_replace, content)

    # --- 3. 修改 HTML 图片 (增加重复检查) ---
    def html_img_replace(match):
        before, src_path, after = match.group(1), match.group(2), match.group(3)
        
        new_src = src_path
        if '/images/' not in src_path:
            new_src = f"/images/{name_without_ext}/{os.path.basename(src_path)}"
            
        # 检查 HTML 标签内是否已含 img-fluid
        if 'img-fluid' in before or 'img-fluid' in after:
            return f'<img{before}src="{new_src}"{after}>'
        
        return f'<img{before}class="img-fluid" src="{new_src}"{after}>'

    content = re.sub(r'<img([^>]+)src=["\']([^"\']+)["\']([^>]*)>', html_img_replace, content)

    # --- 4. 公式处理 ---
    # 仅将 $...$ 转换为 $$...$$，不再替换内部的 | 符号
    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$$\1$$', content)

    # --- 5. 写回原文件 ---
    with open(post_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✅ 原地更新完成: {base_name}")
    print("   - 链接 | 符号修复: 已完成")
    print("   - 图片路径及类名: 已检查并更新 (跳过已存在的 img-fluid)")
    print("   - 公式转换: $ -> $$ (公式内 | 保持原样)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python update_alfolio.py <文件路径>")
        sys.exit(1)
    update_post(sys.argv[1])