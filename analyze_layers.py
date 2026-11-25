# -*- coding: utf-8 -*-
"""分层分析：模型 vs SP 基线在不同联赛/置信度区间的表现"""
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


def main():
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    with open("data/feature_state.pkl", "rb") as f:
        cols = pickle.load(f)["feature_cols"]
    miss_cols = [c + "_miss" for c in cols]
    X = df[cols + miss_cols].values
    yh = df["fthg"].values.astype(float)
    ya = df["ftag"].values.astype(float)

    params = {"objective": "poisson", "num_leaves": 50, "learning_rate": 0.036,
              "min_child_samples": 88, "feature_fraction": 0.735, "lambda_l2": 1.96,
              "verbosity": -1}
    test_start = df["date"].max() - pd.Timedelta(days=365)
    tr = (df["date"] < pd.Timestamp("2025-06-30")).values
    te = (df["date"] > test_start).values

    mh = lgb.train(params, lgb.Dataset(X[tr], label=yh[tr]), num_boost_round=400)
    ma = lgb.train(params, lgb.Dataset(X[tr], label=ya[tr]), num_boost_round=400)
    ch = IsotonicRegression(out_of_bounds="clip"); ch.fit(np.clip(mh.predict(X[tr]), 1e-6, None), yh[tr])
    ca = IsotonicRegression(out_of_bounds="clip"); ca.fit(np.clip(ma.predict(X[tr]), 1e-6, None), ya[tr])
    lh = np.maximum(ch.predict(np.clip(mh.predict(X[te]), 1e-6, None)), 1e-6)
    la = np.maximum(ca.predict(np.clip(ma.predict(X[te]), 1e-6, None)), 1e-6)

    n = te.sum()
    pm = np.zeros((n, 3))
    for i in range(n):
        hw, dr, aw = grid_to_probs(build_grid(lh[i], la[i], 0.03))
        pm[i] = [hw, dr, aw]
    yh_t, ya_t = yh[te], ya[te]
    yoh = np.zeros((n, 3)); yoh[yh_t > ya_t, 0] = 1; yoh[yh_t == ya_t, 1] = 1; yoh[yh_t < ya_t, 2] = 1
    inv = 1.0 / df.loc[te, ["b365h", "b365d", "b365a"]].values
    base = inv / inv.sum(axis=1, keepdims=True)

    def ll(a, b):
        return -np.mean(np.sum(yoh[a:b] * np.log(np.clip(pm[a:b], 1e-12, 1)), axis=1))
    def llb(a, b):
        return -np.mean(np.sum(yoh[a:b] * np.log(np.clip(base[a:b], 1e-12, 1)), axis=1))

    df_te = df[te].reset_index(drop=True)
    print("== 联赛分层（测试集）==")
    for lg, mask in [
        ("五大联赛", df_te["league"].isin({"英超", "西甲", "德甲", "意甲", "法甲"}).values),
        ("日韩荷", df_te["league"].isin({"日职", "韩职", "荷甲", "荷乙"}).values),
        ("其他", ~df_te["league"].isin({"英超", "西甲", "德甲", "意甲", "法甲", "日职", "韩职", "荷甲", "荷乙"}).values),
    ]:
        idx = np.where(mask)[0]
        if len(idx) < 50:
            continue
        print(f"  {lg}({len(idx)}场): 模型LL={ll(idx[0], idx[-1]+1):.4f} vs 基线LL={llb(idx[0], idx[-1]+1):.4f}")

    print("\n== 平局概率分箱（模型是否优于基线预测平局）==")
    # 模型 vs 基线的平局概率差异
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.01)]:
        m = (pm[:, 1] >= lo) & (pm[:, 1] < hi)
        if m.sum() < 30:
            continue
        act_draw = yoh[m, 1].mean()
        print(f"  模型平局概率{lo:.0%}-{hi:.0%}: {m.sum()}场, 实际平率{act_draw:.0%}")

    print("\n== 总体 ==")
    print(f"  模型LL={ll(0, n):.4f} vs 基线LL={llb(0, n):.4f}")


if __name__ == "__main__":
    main()
