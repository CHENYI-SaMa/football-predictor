# -*- coding: utf-8 -*-
"""对照实验：不同时序划分下模型 vs SP 基线
方案A: 任务书严格划分（2024-08-01前训练）→ 训练735场
方案B: 合理划分（训练=2024-08-01~2025-06-30, 验证=2025-07-01~2025-08-07, 测试=最近1年）
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
    if rho != 0:
        for i in range(2):
            for j in range(2):
                lam_h_c, lam_a_c = lam_h, lam_a
                if i == 0 and j == 0:
                    tau = 1 - lam_h_c * lam_a_c * rho
                elif i == 0 and j == 1:
                    tau = 1 + lam_h_c * rho
                elif i == 1 and j == 0:
                    tau = 1 + lam_a_c * rho
                elif i == 1 and j == 1:
                    tau = 1 - rho
                else:
                    tau = 1
                grid[i, j] *= tau
    s = grid.sum()
    return grid / s if s > 0 else grid


def grid_to_probs(grid):
    n = grid.shape[0]
    idx = np.arange(n)
    ih, ja = np.meshgrid(idx, idx, indexing="ij")
    return (grid[ih > ja].sum(), grid[ih == ja].sum(),
            grid[ih < ja].sum(), None)


def run_split(name, df, cols, X, yh, ya, tr, va, te, params, n_rounds=400):
    mh = lgb.train({**params, "objective": "poisson", "verbosity": -1},
                   lgb.Dataset(X[tr], label=yh[tr]),
                   num_boost_round=n_rounds,
                   valid_sets=[lgb.Dataset(X[va], label=yh[va], reference=lgb.Dataset(X[tr], label=yh[tr]))],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    ma = lgb.train({**params, "objective": "poisson", "verbosity": -1},
                   lgb.Dataset(X[tr], label=ya[tr]),
                   num_boost_round=n_rounds,
                   valid_sets=[lgb.Dataset(X[va], label=ya[va], reference=lgb.Dataset(X[tr], label=ya[tr]))],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    # isotonic 校准
    ch = IsotonicRegression(out_of_bounds="clip"); ch.fit(np.clip(mh.predict(X[tr]), 1e-6, None), yh[tr])
    ca = IsotonicRegression(out_of_bounds="clip"); ca.fit(np.clip(ma.predict(X[tr]), 1e-6, None), ya[tr])
    lh = np.maximum(ch.predict(np.clip(mh.predict(X[te]), 1e-6, None)), 1e-6)
    la = np.maximum(ca.predict(np.clip(ma.predict(X[te]), 1e-6, None)), 1e-6)
    n = te.sum()
    pm = np.zeros((n, 3))
    for i in range(n):
        hw, dr, aw, _ = grid_to_probs(build_grid(lh[i], la[i], 0.03))
        pm[i] = [hw, dr, aw]
    yh_t, ya_t = yh[te], ya[te]
    yoh = np.zeros((n, 3)); yoh[yh_t > ya_t, 0] = 1; yoh[yh_t == ya_t, 1] = 1; yoh[yh_t < ya_t, 2] = 1
    # 基线
    inv = 1.0 / df.loc[te, ["b365h", "b365d", "b365a"]].values
    base = inv / inv.sum(axis=1, keepdims=True)
    ll_m = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
    ll_b = float(-np.mean(np.sum(yoh * np.log(np.clip(base, 1e-12, 1)), axis=1)))
    acc_m = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
    acc_b = float(np.mean(np.argmax(base, 1) == np.argmax(yoh, 1)))
    print(f"\n[{name}] 训练{tr.sum()} 验证{va.sum()} 测试{n}")
    print(f"  LL: 模型{ll_m:.4f} vs 基线{ll_b:.4f} (Δ{ll_b-ll_m:+.4f})")
    print(f"  Acc: 模型{acc_m:.3f} vs 基线{acc_b:.3f} (Δ{acc_m-acc_b:+.3f})")
    return ll_m, ll_b, acc_m, acc_b


def main():
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    with open("data/feature_state.pkl", "rb") as f:
        cols = pickle.load(f)["feature_cols"]
    miss_cols = [c + "_miss" for c in cols]
    X = df[cols + miss_cols].values
    yh = df["fthg"].values.astype(float)
    ya = df["ftag"].values.astype(float)
    params = {"num_leaves": 50, "learning_rate": 0.036, "min_child_samples": 88,
              "feature_fraction": 0.735, "lambda_l2": 1.96}
    test_start = df["date"].max() - pd.Timedelta(days=365)
    te = (df["date"] > test_start).values

    # 方案A: 任务书严格
    trA = (df["date"] < pd.Timestamp("2024-08-01")).values
    vaA = ((df["date"] >= pd.Timestamp("2024-08-01")) & (df["date"] <= test_start)).values
    run_split("方案A 任务书严格划分", df, cols, X, yh, ya, trA, vaA, te, params)

    # 方案B: 合理划分（训练到2025-06-30，验证2025-07~08）
    trB = (df["date"] < pd.Timestamp("2025-06-30")).values
    vaB = ((df["date"] >= pd.Timestamp("2025-06-30")) & (df["date"] <= test_start)).values
    run_split("方案B 训练充分划分", df, cols, X, yh, ya, trB, vaB, te, params)


if __name__ == "__main__":
    main()
