# -*- coding: utf-8 -*-
"""ab_form.py —— form 特征 A/B 对比（同一数据、同一参数，仅切换 form 特征）

方案A: 全特征（含 form 抢分率/净胜球）
方案B: 去掉 form 特征（188）
结论用于判断新增 form 特征的真实贡献。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from featsel import build_data, train_eval          # noqa: E402
from src.evaluation import odds_baseline_probs      # noqa: E402


def main():
    tr, va, te, cols = build_data()
    print(f"[分割] 训练 {len(tr):,} | 验证 {len(va):,} | 测试 {len(te):,}")
    print(f"全特征: {len(cols)} | form 特征: {sum(1 for c in cols if 'form' in c)}")
    print()

    # 市场基线
    o = odds_baseline_probs(te["odds_h"].values, te["odds_d"].values, te["odds_a"].values)
    yh_r = te["fthg"].values + te["rang"].values
    yoh = np.zeros((len(yh_r), 3))
    yoh[yh_r > te["ftag"].values, 0] = 1
    yoh[yh_r == te["ftag"].values, 1] = 1
    yoh[yh_r < te["ftag"].values, 2] = 1
    print(f"{'市场SP基线':<24} 让球 Acc={np.mean(np.argmax(o,1)==np.argmax(yoh,1)):.3f}\n")

    # 方案A：全特征（含 form）
    train_eval(cols, tr, va, te, "A 全特征(含form)")

    # 方案B：去掉 form 特征
    no_form = [c for c in cols if "form" not in c]
    train_eval(no_form, tr, va, te, "B 去form特征")

    print()
    print("注意：两次训练同一 fit 参数（seed 固定），结果可复现。")


if __name__ == "__main__":
    main()
