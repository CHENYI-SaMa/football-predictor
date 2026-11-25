# -*- coding: utf-8 -*-
"""从 GitHub 下载 UI-UX-Pro-Max skill 并安装到 ~/.workbuddy/skills/"""
import io
import json
import os
import shutil
import sys
import zipfile

import requests

SKILL_NAME = "ui-ux-pro-max"
ZIP_URL = "https://codeload.github.com/nextlevelbuilder/ui-ux-pro-max-skill/zip/refs/heads/main"
TMP = os.path.expanduser("~/.workbuddy/.tmp_uiux")
DEST = os.path.expanduser(f"~/.workbuddy/skills/{SKILL_NAME}")


def main():
    H = {"User-Agent": "Mozilla/5.0"}
    data = None
    for att in range(4):
        try:
            r = requests.get(ZIP_URL, headers=H, timeout=180, stream=True)
            if r.status_code == 200:
                chunks = []
                total = 0
                for c in r.iter_content(1 << 20):
                    chunks.append(c)
                    total += len(c)
                data = b"".join(chunks)
                print(f"下载完成 {total/1024:.0f} KB", flush=True)
                break
            print(f"HTTP {r.status_code}", flush=True)
        except Exception as e:
            print(f"第{att+1}次失败: {type(e).__name__}: {e}", flush=True)
    if data is None:
        print("下载失败")
        sys.exit(1)

    if os.path.exists(TMP):
        shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(TMP)
    print("解压完成:", os.listdir(TMP), flush=True)

    # 找含 SKILL.md 的根目录
    root = None
    for d in os.listdir(TMP):
        p = os.path.join(TMP, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "SKILL.md")):
            root = p
            break
    if root is None:
        for d in os.listdir(TMP):
            p = os.path.join(TMP, d)
            if os.path.isdir(p):
                for d2 in os.listdir(p):
                    if os.path.exists(os.path.join(p, d2, "SKILL.md")):
                        root = os.path.join(p, d2)
                        break
            if root:
                break
    print("SKILL.md 根:", root, flush=True)
    if not root:
        sys.exit(1)

    # 安装到 skills 目录
    if os.path.exists(DEST):
        shutil.rmtree(DEST, ignore_errors=True)
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    shutil.copytree(root, DEST)
    print(f"已安装到 {DEST}", flush=True)

    # 打印 SKILL.md 头部（检查结构）
    sk = open(os.path.join(DEST, "SKILL.md"), encoding="utf-8", errors="ignore").read()
    print("\n== SKILL.md 前 800 字 ==", flush=True)
    print(sk[:800], flush=True)
    print("\n== 目录结构 ==", flush=True)
    for root2, dirs, files in os.walk(DEST):
        level = root2.replace(DEST, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root2)}/", flush=True)
        for f in files[:8]:
            print(f"{indent}  {f}", flush=True)


if __name__ == "__main__":
    main()
