---
layout: post
title: "Matplotlib输出中文"
date: 2025-12-25 22:06:17 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left # or right
typora-root-url: ../
---

#### 步骤一：在 Ubuntu 系统上安装中文字体并清理缓存

您需要在 Ubuntu 24 上安装一个包含中文支持的字体包。

1. **安装中文字体（推荐：文泉驿）：** 打开您的终端，执行以下命令安装常用的开源中文字体包：

   Bash

   ```
   sudo apt update
   sudo apt install fonts-wqy-zenhei
   ```

2. **清理 Matplotlib 缓存：** Matplotlib 会缓存系统中的字体信息。如果您不清理缓存，即使安装了新字体，它也可能无法立即识别。

   Bash

   ```
   # 找到缓存目录的位置
   python -c "import matplotlib; print(matplotlib.get_cachedir())"
   
   # 通常该目录是 ~/.cache/matplotlib，清理它
   rm -rf ~/.cache/matplotlib
   
   # （重要）清理后，请确保重新启动您的 Cursor 终端或整个 IDE，以使新的环境变量和字体生效。
   ```

#### 步骤二：修改 Python 代码配置 Matplotlib 字体

在您的 Python 脚本开头，导入 Matplotlib 之后，添加字体配置代码，指定 Matplotlib 优先使用支持中文的字体。

**请用下面这段代码替换您脚本开头的所有 `import` 语句以及它们之后的代码（在 `visualize_dinov2_features_variable_res` 函数定义之前）。**

```python
# =========================================================
# Matplotlib 中文字体配置（解决 UserWarning 问题）
# =========================================================
try:
    # 尝试设置 Matplotlib 的字体
    # 使用一个字体列表，优先尝试安装的文泉驿字体，确保兼容性
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    
    # 解决负号显示问题 (可选，但推荐)
    plt.rcParams['axes.unicode_minus'] = False 
    print("Matplotlib 已配置中文字体。")
except Exception as e:
    print(f"配置中文字体失败: {e}。将使用默认字体。")
# =========================================================
```

