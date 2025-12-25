import os
import sys
import re
import shutil
from datetime import datetime

def transform_post(post_path, image_position=0):
    if not os.path.exists(post_path):
        print(f"错误: 找不到文件 {post_path}")
        return

    # --- 1. 日期逻辑 ---
    if image_position == 1:
        current_date = "2024-12-30"
    else:
        current_date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    base_name = os.path.basename(post_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # --- 2. 文件名空格处理 ---
    safe_name = name_without_ext.replace(" ", "-")
    new_name_prefix = f"{current_date}-{safe_name}"
    new_post_filename = f"{new_name_prefix}.md"
    
    target_img_root = "./images"
    if not os.path.exists(target_img_root):
        os.makedirs(target_img_root)

    # --- 3. 搬运图片文件夹 ---
    if image_position == 1:
        old_img_dir = os.path.join(os.path.dirname(post_path), name_without_ext)
    else:
        old_img_dir = os.path.join(target_img_root, name_without_ext)

    final_img_path = os.path.join(target_img_root, new_name_prefix)

    img_sync_status = "未发现目录"
    if os.path.exists(old_img_dir):
        shutil.copytree(old_img_dir, final_img_path, dirs_exist_ok=True)
        img_sync_status = f"已同步至 {final_img_path}"

    # --- 4. 读取并处理内容 ---
    with open(post_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # A. 修复 Markdown 链接中的 | 符号 (修正版)
    def link_pipe_fix(match):
        title = match.group(1)
        link = match.group(2)
        # 标题中使用 \| 转义，链接中使用 %7C 编码
        fixed_title = title.replace('|', r'\|')
        fixed_link = link.replace('|', '%7C')
        return f'[{fixed_title}]({fixed_link})'
    
    # 匹配 [标题](链接)
    content = re.sub(r'\[(.*?)\]\((.*?)\)', link_pipe_fix, content)

    # B. 修改 Markdown 图片 + 响应式类
    content = re.sub(r'!\[(.*?)\]\((.*?)\)', 
                     lambda m: f"![{m.group(1)}](/images/{new_name_prefix}/{os.path.basename(m.group(2))}){{: .img-fluid}}", 
                     content)

    # C. 修改 HTML 图片
    def html_img_replace(match):
        b, src_path, a = match.group(1), match.group(2), match.group(3)
        new_src = f"/images/{new_name_prefix}/{os.path.basename(src_path)}"
        if 'class=' not in b and 'class=' not in a:
            return f'<img{b}class="img-fluid" src="{new_src}"{a}>'
        return f'<img{b}src="{new_src}"{a}>'

    content = re.sub(r'<img([^>]+)src=["\']([^"\']+)["\']([^>]*)>', html_img_replace, content)

    # --- 5. 公式处理 ---
    content = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$$\1$$', content)
    content = re.sub(r'\$\$(.*?)\$\$', lambda m: f"$${m.group(1).replace('|', r'\mid ')}$$", content, flags=re.DOTALL)

    # --- 6. 生成 Front Matter ---
    display_title = name_without_ext.replace('-', ' ').title()
    front_matter = f"""---
layout: post
title: "{display_title}"
date: {current_date} {current_time} +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---

"""
    # --- 7. 写入并输出报告 ---
    new_post_path = os.path.join(os.path.dirname(post_path), new_post_filename)
    with open(new_post_path, 'w', encoding='utf-8') as f:
        f.write(front_matter + content)

    print("\n" + "="*45)
    print(f"🚀 al-folio 转换任务完成！")
    print("="*45)
    print(f" [文件信息]\n  ├─ 原始: {base_name}\n  ├─ 目标: {new_post_filename}\n  └─ 日期: {current_date}")
    print(f" [图片处理]\n  ├─ 状态: {img_sync_status}\n  └─ 类名: 已注入 .img-fluid 响应式支持")
    print(f" [内容优化]\n  ├─ 公式: $->$$ 升级, |->\\mid 修正")
    print(f"  └─ 链接: 标题内 | 已转义为 \\|, 链接内 | 已编码为 %7C")
    print("="*45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python to_alfolio.py <文件路径> [image_position]")
        sys.exit(1)
    transform_post(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)