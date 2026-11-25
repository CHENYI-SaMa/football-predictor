# -*- coding: utf-8 -*-
"""打包源码全量 zip：包含隐藏文件，排除构建产物，保持目录结构。"""
import os
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\CHENYI\AppData\Roaming\TRAE SOLO CN\ModularData\ai-agent\work-mode-projects\6a895a6a31f95ef2795f8b06\football-prediction")
OUT = Path(r"C:\Users\CHENYI\Desktop\竞彩足球预测_源码全量_20260910.zip")

# 排除：构建产物 / 虚拟环境 / 缓存 / 安装包目录
EXCLUDE_DIRS = {
    ".workbuddy", "__pycache__", "build", "buildBayes", "buildV4_predict",
    "dist3", "dist4", "dist5", "dist6", "dist7", "dist8", "dist9",
    "distBayes", "distV4", "distV4_new", "打包",
}
EXCLUDE_FILES = {"_md2docx.py", "_md2docx_v2.py", "_inspect_docx.py", "_inspect2.py", "_verify_docx.py", "_md2docx_v3.py", "_md2docx_final.py"}

count = 0
total = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in sorted(files):
            if f in EXCLUDE_FILES:
                continue
            fp = Path(root) / f
            rel = fp.relative_to(SRC).as_posix()
            try:
                z.write(fp, rel)
                count += 1
                total += fp.stat().st_size
            except OSError as e:
                print("skip:", rel, e)

print(f"打包文件数: {count} | 原始大小: {total/1024/1024:.1f} MB")
print("输出:", OUT)
print("zip 大小: %.1f MB" % (OUT.stat().st_size / 1024 / 1024))