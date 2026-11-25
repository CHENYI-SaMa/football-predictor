@echo off
chcp 65001 >nul
title 竞彩足球预测系统 - 源码版
cd /d %~dp0

echo ==============================================
echo    竞彩足球预测系统 (源码版) 一键启动
echo ==============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 Python！
    echo 请先安装 Python 3.12 并勾选 "Add python.exe to PATH"
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
python -c "import pandas, numpy, scipy, sklearn, lightgbm, requests, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo [2/3] 首次运行，正在安装依赖（约 1-3 分钟，请耐心等待）...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络后重试
        pause
        exit /b 1
    )
) else (
    echo [2/3] 依赖已就绪
)

echo [3/3] 启动软件...
echo.
python gui.py
pause
