#!/bin/bash
# 用法: ./new_post.sh "My New Post"
TITLE=$1
SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
DATE=$(date +%Y-%m-%d)
FILENAME="_posts/${DATE}-${SLUG}.md"

cat <<EOF > "$FILENAME"
---
layout: post
title: "$TITLE"
date: $DATE +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
typora-root-url: ../
---

## 二级标题
EOF

echo "已创建新文章: $FILENAME"