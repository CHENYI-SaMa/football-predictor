# -*- coding: utf-8 -*-
"""决定性对照：赔率特征 vs 过程数据的贡献分解
模型1: 全部特征（含赔率）
模型2: 无赔率特征（implied_* 移除）
模型3: 仅赔率特征
"""
import sys, warnings, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

MAX_GOALS = 10


def build_grid(lam_h, lam_a, rho=0.0):
    maxg = MAX_GOALS
    ph = poisson.pmf(np.arange(maxg + 1), lam_h)
    pa = poisson.pmf(np.arange(maxg + 1), lam_a)
    grid = np.outer(ph, pa)
    s = grid.sum()
    return grid / s if s > 0 else grid


def grid_to_probs(grid):
    n = grid.shape[0]
    idx = np.arange(n)
    ih, ja = np.meshgrid(idx, idx, indexing="ij")
    return (grid[ih > ja].sum(), grid[ih == ja].sum(), grid[ih < ja].sum())


def evaluate(X_tr, X_te, yh_tr, ya_tr, yh_te, ya_te, name):
    params = {"objective": "poisson", "num_leaves": 50, "learning_rate": 0.036,
              "min_child_samples": 88, "feature_fraction": 0.735, "lambda_l2": 1.96,
              "verbosity": -1}
    mh = lgb.train(params, lgb.Dataset(X_tr, label=yh_tr), num_boost_round=400)
    ma = lgb.train(params, lgb.Dataset(X_tr, label=ya_tr), num_boost_round=400)
    ch = IsotonicRegression(out_of_bounds="clip"); ch.fit(np.clip(mh.predict(X_tr), 1e-6, None), yh_tr)
    ca = IsotonicRegression(out_of_bounds="clip"); ca.fit(np.clip(ma.predict(X_tr), 1e-6, None), ya_tr)
    lh = np.maximum(ch.predict(np.clip(mh.predict(X_te), 1e-6, None)), 1e-6)
    la = np.maximum(ca.predict(np.clip(ma.predict(X_te), 1e-6, None)), 1e-6)
    n = len(yh_te)
    pm = np.zeros((n, 3))
    for i in range(n):
        hw, dr, aw = grid_to_probs(build_grid(lh[i], la[i], 0.03))
        pm[i] = [hw, dr, aw]
    yoh = np.zeros((n, 3)); yoh[yh_te > ya_te, 0] = 1; yoh[yh_te == ya_te, 1] = 1; yoh[yh_te < ya_te, 2] = 1
    ll_m = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
    acc_m = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
    print(f"  [{name}] LL={ll_m:.4f} Acc={acc_m:.3f}")
    return ll_m, acc_m


def main():
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    with open("data/feature_state.pkl", "rb") as f:
        cols = pickle.load(f)["feature_cols"]
    miss_cols = [c + "_miss" for c in cols]
    test_start = df["date"].max() - pd.Timedelta(days=365)
    tr = (df["date"] < pd.Timestamp("2025-06-30")).values
    te = (df["date"] > test_start).values
    yh_tr, ya_tr = df.loc[tr, "fthg"].values.astype(float), df.loc[tr, "ftag"].values.astype(float)
    yh_te, ya_te = df.loc[te, "fthg"].values.astype(float), df.loc[te, "ftag"].values.astype(float)

    odds_cols = ["implied_H", "implied_D", "implied_A"]
    odds_miss = [c + "_miss" for c in odds_cols]
    proc_cols = [c for c in cols if c not in odds_cols]
    proc_miss = [c + "_miss" for c in proc_cols]

    print("== 特征贡献分解（测试集 5022 场，训练充分版）==")
    # 基线
    inv = 1.0 / df.loc[te, ["b365h", "b365d", "b365a"]].values
    base = inv / inv.sum(axis=1, keepdims=True)
    yoh = np.zeros((len(yh_te), 3))
    yoh[yh_te > ya_te, 0] = 1; yoh[yh_te == ya_te, 1] = 1; yoh[yh_te < ya_te, 2] = 1
    print(f"  [SP赔率基线] LL={float(-np.mean(np.sum(yoh*np.log(np.clip(base,1e-12,1)),axis=1))):.4f} "
          f"Acc={float(np.mean(np.argmax(base,1)==np.argmax(yoh,1))):.3f}")

    evaluate(df.loc[tr, cols + miss_cols].values, df.loc[te, cols + miss_cols].values,
             yh_tr, ya_tr, yh_te, ya_te, "模型1 全部特征")
    evaluate(df.loc[tr, proc_cols + proc_miss].values, df.loc[te, proc_cols + proc_miss].values,
             yh_tr, ya_tr, yh_te, ya_te, "模型2 无赔率(纯过程)")
    evaluate(df.loc[tr, odds_cols + odds_miss].values, df.loc[te, odds_cols + odds_miss].values,
             yh_tr, ya_tr, yh_te, ya_te, "模型3 仅赔率")


if __name__ == "__main__":
    main()
