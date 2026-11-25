# -*- coding: utf-8 -*-
"""featsel.py —— 特征选择实验（任务12 D2/D3）
全特征 vs top-k 子集（按 models_v3 gain importance）重训对比。
用法: python featsel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.features_v3 import FeatureBuilderV3          # noqa: E402
from src.model import PoissonLightGBM                 # noqa: E402
from src.evaluation import odds_baseline_probs        # noqa: E402

HIST_CSV = BASE_DIR / "data" / "jczq_history_xg.csv"


def build_data():
    raw = pd.read_csv(HIST_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    fb = FeatureBuilderV3(raw)
    feat, _ = fb.build()
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)
    max_date = feat["date"].max()
    test_start = max_date - pd.Timedelta(days=30)
    val_start = max_date - pd.Timedelta(days=90)
    tr = feat[feat["date"] <= val_start].copy()
    va = feat[(feat["date"] > val_start) & (feat["date"] <= test_start)].copy()
    te = feat[feat["date"] > test_start].copy()
    cols = [c for c in feat.columns if c not in
            {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}]
    return tr, va, te, cols


def range_metrics(model, X, y_h, y_a, rang):
    grids = model.predict_grids(X)
    n = grids.shape[1]
    ih, ja = np.indices((n, n))
    shifted = ih[None, :, :] + rang[:, None, None]
    pr = np.clip(np.stack([
        np.sum(grids * (shifted > ja), axis=(1, 2)),
        np.sum(grids * (shifted == ja), axis=(1, 2)),
        np.sum(grids * (shifted < ja), axis=(1, 2))], 1), 1e-12, 1.0)
    pr /= pr.sum(axis=1, keepdims=True)
    yh_r = y_h + rang
    yoh = np.zeros((len(y_h), 3))
    yoh[yh_r > y_a, 0] = 1
    yoh[yh_r == y_a, 1] = 1
    yoh[yh_r < y_a, 2] = 1
    ll = float(-np.mean(np.sum(yoh * np.log(pr), axis=1)))
    acc = float(np.mean(np.argmax(pr, 1) == np.argmax(yoh, 1)))
    return ll, acc


def train_eval(subset_cols, tr, va, te, label):
    X_tr = tr[subset_cols].values
    X_va = va[subset_cols].values
    X_te = te[subset_cols].values
    yh_tr, ya_tr = tr["fthg"].values, tr["ftag"].values
    yh_va, ya_va = va["fthg"].values, va["ftag"].values
    yh_te, ya_te = te["fthg"].values, te["ftag"].values
    model = PoissonLightGBM()
    model.fit(X_tr, yh_tr, ya_tr, X_va, yh_va, ya_va,
              num_boost_round=1500, early_stopping=80)
    rll, racc = range_metrics(model, X_te, yh_te, ya_te, te["rang"].values)
    vll, vacc = range_metrics(model, X_va, yh_va, ya_va, va["rang"].values)
    print(f"{label:<24} 特征{len(subset_cols):<4} 测试让球LL={rll:.4f} "
          f"Acc={racc:.3f} | 验证Acc={vacc:.3f}")
    return racc


def main():
    tr, va, te, cols = build_data()
    print(f"[分割] 训练 {len(tr):,} | 验证 {len(va):,} | 测试 {len(te):,}")
    print(f"全特征数: {len(cols)}")

    # 市场基线（让球SP去抽水）
    o = odds_baseline_probs(te["odds_h"].values, te["odds_d"].values, te["odds_a"].values)
    yh_r = te["fthg"].values + te["rang"].values
    yoh = np.zeros((len(yh_r), 3))
    yoh[yh_r > te["ftag"].values, 0] = 1
    yoh[yh_r == te["ftag"].values, 1] = 1
    yoh[yh_r < te["ftag"].values, 2] = 1
    m_acc = float(np.mean(np.argmax(o, 1) == np.argmax(yoh, 1)))
    print(f"{'市场SP基线':<24} 让球 Acc={m_acc:.3f}\n")

    # 全特征
    train_eval(cols, tr, va, te, "全特征")

    # importance 排序（用现有 models_v3 的 gain，训练集统计无泄漏）
    model_ref = PoissonLightGBM.load(BASE_DIR / "models_v3" / "model.pkl")
    imp = np.zeros(len(cols))
    for b in (model_ref.model_h, model_ref.model_a):
        imp += b.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    # 去除 0 重要性
    nonzero = [cols[i] for i in order if imp[i] > 0]
    print(f"非零重要性特征: {len(nonzero)}\n")

    for k in (30, 50, 80, 120):
        subset = [c for c in order if imp[c] > 0][:k] if False else \
            [cols[i] for i in order if imp[i] > 0][:k]
        # 确保含基础列（赔率在 top，应该都在）
        train_eval(subset, tr, va, te, f"top{k}")

    print("\n→ 结论见上表（各特征子集 vs 全特征 vs 市场）")


if __name__ == "__main__":
    main()
