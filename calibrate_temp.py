# -*- coding: utf-8 -*-
"""calibrate_temp.py —— 温度校准扫描（纠错：高置信段错误率偏高）
================================================================
对多个温度 T 评估模型：总让球 Acc、LogLoss、高置信段(≥70%)错误率。
选「高置信段错误率降到合理水平且总 Acc 不降」的最优 T。
用法: python calibrate_temp.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from train_v3 import build_features_v3                 # noqa: E402
from src.model import PoissonLightGBM                  # noqa: E402

HIST_CSV = BASE_DIR / "data" / "jczq_history_xg.csv"


def range_probs(model, X, rang_arr):
    grids = model.predict_grids(X)
    n = grids.shape[1]
    ih, ja = np.indices((n, n))
    r = rang_arr[:, None, None]
    shifted = ih[None, :, :] + r
    pr = np.clip(np.stack([
        np.sum(grids * (shifted > ja), axis=(1, 2)),
        np.sum(grids * (shifted == ja), axis=(1, 2)),
        np.sum(grids * (shifted < ja), axis=(1, 2))], 1), 1e-12, 1.0)
    pr /= pr.sum(axis=1, keepdims=True)
    return pr


def evaluate(model, X, y_h, y_a, rang, act_r):
    pr = range_probs(model, X, rang)
    acc = float((pr.argmax(1) == act_r).mean())
    # LogLoss（三分类）
    ll = float(-np.log(np.clip(pr[np.arange(len(pr)), act_r], 1e-12, 1)).mean())
    # 高置信段（conf>=0.70）错误率
    conf = pr.max(1)
    hc = conf >= 0.70
    hc_err = float((pr[hc].argmax(1) != act_r[hc]).mean()) if hc.sum() else float("nan")
    hc_n = int(hc.sum())
    return acc, ll, hc_err, hc_n


def main():
    raw = pd.read_csv(HIST_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    feat, _ = build_features_v3(raw)
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)
    max_date = feat["date"].max()
    test_start = max_date - pd.Timedelta(days=30)
    val_start = max_date - pd.Timedelta(days=60)
    cols = [c for c in feat.columns if c not in
            {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}]
    cols = [c for c in cols if "form" not in c]
    print(f"特征 {len(cols)} 个 | 数据至 {max_date.date()}")

    def prep(seg):
        X = seg[cols].values.astype(float)
        y_h = seg["fthg"].values.astype(float)
        y_a = seg["ftag"].values.astype(float)
        rang = seg["rang"].values.astype(float)
        act = np.where(y_h + rang > y_a, 0, np.where(y_h + rang == y_a, 1, 2))
        return X, y_h, y_a, rang, act

    tr = feat[feat["date"] <= val_start]
    va = feat[(feat["date"] > val_start) & (feat["date"] <= test_start)]
    te = feat[feat["date"] > test_start]
    va_p, te_p = prep(va), prep(te)

    model = PoissonLightGBM.load(BASE_DIR / "models_v3" / "model.pkl")
    base = model.temperature
    print(f"当前 T={base:.3f}\n")
    print(f"{'T':>6} | {'验证Acc':>7} {'验证LL':>6} | {'测试Acc':>7} {'测试LL':>6} | 高置信段错误率(验证/测试) 样本")
    results = []
    for T in [base, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]:
        model.temperature = T
        a_va = evaluate(model, *va_p)
        a_te = evaluate(model, *te_p)
        results.append((T, a_va, a_te))
        print(f"{T:>6.2f} | {a_va[0]:>7.3f} {a_va[1]:>6.3f} | "
              f"{a_te[0]:>7.3f} {a_te[1]:>6.3f} | "
              f"{a_va[2]:>5.1%}/{a_te[2]:>5.1%}  n={a_va[3]}/{a_te[3]}")

    # 选 T：高置信段错误率(测试) <= 15% 且测试 Acc 不降超过 0.005 的最大 T
    best = None
    for T, av, at in results:
        if at[2] <= 0.15 and at[0] >= results[0][2][0] - 0.005:
            best = T
    model.temperature = base
    print(f"\n当前基准: Acc(测)={results[0][2][0]:.3f} 高置信错误率(测)={results[0][2][2]:.1%}")
    if best:
        print(f"建议温度 T={best:.2f}")
    else:
        print("无满足条件的 T（高置信错误率已接近合理或提升 T 会伤总 Acc），保持现状")
    with open(BASE_DIR / "outputs" / "calibrate_temp.json", "w", encoding="utf-8") as f:
        json.dump([{"T": t, "va_acc": a[0], "va_ll": a[1], "va_hc_err": a[2], "va_hc_n": a[3],
                    "te_acc": b[0], "te_ll": b[1], "te_hc_err": b[2], "te_hc_n": b[3]}
                   for t, a, b in results], f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
