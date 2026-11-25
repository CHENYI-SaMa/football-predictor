# -*- coding: utf-8 -*-
"""train_spf_v3.py —— 三分类直连 + Stacking 融合（任务3）
================================================================
阶段B（任务3 核心）：在 V3 特征上训练：
  模型1: 双泊松（train_v3.py 已产 models_v3）
  模型2: 三分类直连 LightGBM（multiclass，直接优化 H/D/A）
  融合 : 逻辑回归元学习器（输入：双泊松3概率 + 三分类3概率 + 市场3概率 = 9维，
         输出三分类概率；验证集拟合权重，测试集评估）
评估: 赛果/让球 Log Loss、Acc，与单模型/市场基线对比
保存: models_v3_spf/（三分类 + 融合器）
用法: python train_spf_v3.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.features_v3 import FeatureBuilderV3          # noqa: E402
from src.model import PoissonLightGBM                 # noqa: E402
from src.evaluation import odds_baseline_probs        # noqa: E402

HIST_CSV = BASE_DIR / "data" / "jczq_history_xg.csv"
OUT_DIR = BASE_DIR / "models_v3_spf"


def build_data():
    raw = pd.read_csv(HIST_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    fb = FeatureBuilderV3(raw)
    feat, state = fb.build()
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)
    max_date = feat["date"].max()
    test_start = max_date - pd.Timedelta(days=30)
    val_start = max_date - pd.Timedelta(days=90)
    tr = feat[feat["date"] <= val_start].copy()
    va = feat[(feat["date"] > val_start) & (feat["date"] <= test_start)].copy()
    te = feat[feat["date"] > test_start].copy()
    cols = [c for c in feat.columns if c not in
            {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}]
    return (tr, va, te, cols, state)


def outcome_prob_from_grid(grids: np.ndarray) -> np.ndarray:
    n = grids.shape[1]
    ih, ja = np.indices((n, n))
    ph = grids[:, ih > ja].sum(axis=1)
    pd_ = grids[:, ih == ja].sum(axis=1)
    pa = grids[:, ih < ja].sum(axis=1)
    return np.clip(np.stack([ph, pd_, pa], axis=1), 1e-12, 1.0)


def onehot(y_h, y_a):
    y = np.zeros((len(y_h), 3))
    y[y_h > y_a, 0] = 1
    y[y_h == y_a, 1] = 1
    y[y_h < y_a, 2] = 1
    return y


def range_probs(grids: np.ndarray, rang: np.ndarray) -> np.ndarray:
    n = grids.shape[1]
    ih, ja = np.indices((n, n))
    shifted = ih[None, :, :] + rang[:, None, None]
    pr = np.clip(np.stack([
        grids[shifted > ja].sum(axis=1) if False else np.sum(grids * (shifted > ja), axis=(1, 2)),
        np.sum(grids * (shifted == ja), axis=(1, 2)),
        np.sum(grids * (shifted < ja), axis=(1, 2)),
    ], axis=1), 1e-12, 1.0)
    return pr / pr.sum(axis=1, keepdims=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tr, va, te, cols, state = build_data()
    print(f"[分割] 训练 {len(tr):,} | 验证 {len(va):,} | 测试 {len(te):,}")

    # ---- 模型1：双泊松（V3，即 models_v3） ----
    poisson = PoissonLightGBM.load(BASE_DIR / "models_v3" / "model.pkl")

    def poisson_probs(df):
        X = df[cols].values
        grids = poisson.predict_grids(X)
        return outcome_prob_from_grid(grids), range_probs(grids, df["rang"].values)

    p_tr, r_tr = poisson_probs(tr)
    p_va, r_va = poisson_probs(va)
    p_te, r_te = poisson_probs(te)

    # ---- 模型2：三分类直连 LightGBM ----
    y_tr = np.argmax(onehot(tr["fthg"].values, tr["ftag"].values), axis=1)
    y_va = np.argmax(onehot(va["fthg"].values, va["ftag"].values), axis=1)
    y_te = np.argmax(onehot(te["fthg"].values, te["ftag"].values), axis=1)

    dtr = lgb.Dataset(tr[cols].values, label=y_tr)
    dva = lgb.Dataset(va[cols].values, label=y_va, reference=dtr)
    cls = lgb.train({
        "objective": "multiclass", "num_class": 3, "verbosity": -1,
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 30,
        "feature_fraction": 0.8, "lambda_l2": 1.0,
    }, dtr, num_boost_round=500, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)])
    c_tr = np.clip(cls.predict(tr[cols].values), 1e-12, 1.0)
    c_va = np.clip(cls.predict(va[cols].values), 1e-12, 1.0)
    c_te = np.clip(cls.predict(te[cols].values), 1e-12, 1.0)
    print(f"[三分类] 验证 Acc={np.mean(np.argmax(c_va,1)==y_va):.3f}")

    # ---- 市场概率（让球SP 去抽水） ----
    m_tr = odds_baseline_probs(tr["odds_h"].values, tr["odds_d"].values, tr["odds_a"].values)
    m_va = odds_baseline_probs(va["odds_h"].values, va["odds_d"].values, va["odds_a"].values)
    m_te = odds_baseline_probs(te["odds_h"].values, te["odds_d"].values, te["odds_a"].values)

    # ---- Stacking：验证集拟合 LR（9 维输入） ----
    def stack_input(p, c, mkt):
        return np.concatenate([p, c, mkt], axis=1)

    meta = LogisticRegression(max_iter=1000, C=1.0)
    meta.fit(stack_input(p_va, c_va, m_va), y_va)
    s_tr = meta.predict_proba(stack_input(p_tr, c_tr, m_tr))
    s_va = meta.predict_proba(stack_input(p_va, c_va, m_va))
    s_te = meta.predict_proba(stack_input(p_te, c_te, m_te))

    # ---- 评估 ----
    yoh_te = onehot(te["fthg"].values, te["ftag"].values)
    yoh_va = onehot(va["fthg"].values, va["ftag"].values)

    def metric(p, yoh):
        p = np.clip(p, 1e-12, 1.0)
        ll = float(-np.mean(np.sum(yoh * np.log(p), axis=1)))
        acc = float(np.mean(np.argmax(p, 1) == np.argmax(yoh, 1)))
        return ll, acc

    print("\n=== 测试集 赛果（H/D/A）===")
    print(f"{'模型':<14}{'LogLoss':<10}{'Acc':<8}")
    for name, p in (("市场SP", m_te), ("双泊松", p_te), ("三分类", c_te), ("Stacking", s_te)):
        ll, acc = metric(p, yoh_te)
        print(f"{name:<14}{ll:<10.4f}{acc:<8.3f}")

    # 让球维度（用双泊松网格平移 + 融合后的赛果概率近似，见说明）
    print("\n=== 测试集 让球胜平负（双泊松网格平移）===")
    rg_te = te["rang"].values
    yh_r = te["fthg"].values + rg_te
    yohr = np.zeros((len(yh_r), 3))
    yohr[yh_r > te["ftag"].values, 0] = 1
    yohr[yh_r == te["ftag"].values, 1] = 1
    yohr[yh_r < te["ftag"].values, 2] = 1
    rll, racc = metric(r_te, yohr)
    mll, macc = metric(m_te, yohr)
    print(f"双泊松让球: LL={rll:.4f} Acc={racc:.3f} | 市场: LL={mll:.4f} Acc={macc:.3f}")

    # ---- 保存 ----
    (OUT_DIR / "cls_lgb.txt").write_text(cls.model_to_string(), encoding="utf-8")
    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"meta_lr": meta, "feature_cols": cols,
                     "cls_num_leaves": 31}, f)
    json.dump({
        "test": {
            "stack_ll": metric(s_te, yoh_te)[0], "stack_acc": metric(s_te, yoh_te)[1],
            "poisson_ll": metric(p_te, yoh_te)[0], "poisson_acc": metric(p_te, yoh_te)[1],
            "cls_ll": metric(c_te, yoh_te)[0], "cls_acc": metric(c_te, yoh_te)[1],
            "mkt_ll": metric(m_te, yoh_te)[0], "mkt_acc": metric(m_te, yoh_te)[1],
            "range_acc": racc, "range_mkt_acc": macc,
        },
        "val": {
            "stack_ll": metric(s_va, yoh_va)[0], "stack_acc": metric(s_va, yoh_va)[1],
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, open(OUT_DIR / "train_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ 已保存 {OUT_DIR}/（三分类 + Stacking 融合器）")


if __name__ == "__main__":
    main()
