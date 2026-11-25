# -*- coding: utf-8 -*-
"""统一资源路径：打包后从 exe 目录读模型/数据，源码模式从项目目录读。"""
import os
import sys
from pathlib import Path


def get_base_dir() -> Path:
    # 1) 菜单入口注入的环境变量（最高优先级，保证打包后所有模块一致）
    env = os.environ.get("FOOTBALL_BASE_DIR")
    if env:
        return Path(env)
    # 2) PyInstaller 打包模式：_internal 数据目录（PyInstaller 6.x onedir）
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    # 3) 源码模式：path_helper.py 位于项目根目录
    return Path(__file__).resolve().parent


def get_outputs_dir() -> Path:
    """输出/存档目录：桌面/竞彩预测/存档。
    打包版（frozen）下 BASE_DIR 指向临时解压目录会丢数据，
    统一用桌面存档保证跨版本持久、可写、与自动化任务一致。"""
    d = Path.home() / "Desktop" / "竞彩预测" / "存档"
    d.mkdir(parents=True, exist_ok=True)
    return d
