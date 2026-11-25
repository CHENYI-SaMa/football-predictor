# -*- coding: utf-8 -*-
"""阶段D+E(方案A): 让球胜平负模型评估
模型输出 λ_h/λ_a → Dixon-Coles 比分网格 → 按让球数 ahh 平移 → 让球胜平负概率
对比竞彩让球SP基线（b365h/d/a = rsp_h/rsp_d/rsp_a，去抽水）
指标: 让球Log Loss / Brier / 命中率
"""
import sys, warnings, pickle, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

MAX_GOALS = 12


def build_grid(lam_h, lam_a, rho=0.0):
    maxg = MAX_GOALS
    ph = poisson.pmf(np.arange(maxg + 1), lam_h)
    pa = poisson.pmf(np.arange(maxg + 1), lam_a)
    grid = np.outer(ph, pa)
    if rho != 0:
        for i in range(2):
            for j in range(2):
                lh, la = lam_h, lam_a
                if i == 0 and j == 0:
                    tau = 1 - lh * la * rho
                elif i == 0 and j == 1:
                    tau = 1 + lh * rho
                elif i == 1 and j == 0:
                    tau = 1 + la * rho
                elif i == 1 and j == 1:
                    tau = 1 - rho
                else:
                    tau = 1
                grid[i, j] *= tau
    s = grid.sum()
    return grid / s if s > 0 else grid


def range_probs_from_grid(grid, rang):
    """按让球数平移求让球胜平负概率。rang: 主队视角（负=主让）。"""
    n = grid.shape[0]
    ih, ja = np.indices((n, n))
    shifted = ih + rang
    hw = grid[shifted > ja].sum()
    dr = grid[shifted == ja].sum()
    aw = grid[shifted < ja].sum()
    return hw, dr, aw


def range_odds_baseline(df_sub):
    """竞彩让球SP 去抽水 → 基线概率（b365h/d/a 即让球SP）。"""
    o = df_sub[["b365h", "b365d", "b365a"]].values.astype(float)
    valid = np.isfinite(o).all(axis=1) & (o > 1).all(axis=1)
    p = np.full_like(o, np.nan)
    p[valid] = 1.0 / o[valid]
    p[valid] /= p[valid].sum(axis=1, keepdims=True)
    return p


def main():
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    with open("data/feature_state.pkl", "rb") as f:
        cols = pickle.load(f)["feature_cols"]
    miss_cols = [c + "_miss" for c in cols]
    X = df[cols + miss_cols].values
    yh = df["fthg"].values.astype(float)
    ya = df["ftag"].values.astype(float)
    rang = df["rq_line"].values.astype(float)  # 让球数（主队视角）

    # 时序划分（训练充分版）
    test_start = df["date"].max() - pd.Timedelta(days=365)
    tr = (df["date"] < pd.Timestamp("2025-06-30")).values
    te = (df["date"] > test_start).values

    # 训练
    params = {"objective": "poisson", "num_leaves": 50, "learning_rate": 0.036,
              "min_child_samples": 88, "feature_fraction": 0.735, "lambda_l2": 1.96,
              "verbosity": -1}
    mh = lgb.train(params, lgb.Dataset(X[tr], label=yh[tr]), num_boost_round=400)
    ma = lgb.train(params, lgb.Dataset(X[tr], label=ya[tr]), num_boost_round=400)
    ch = IsotonicRegression(out_of_bounds="clip"); ch.fit(np.clip(mh.predict(X[tr]), 1e-6, None), yh[tr])
    ca = IsotonicRegression(out_of_bounds="clip"); ca.fit(np.clip(ma.predict(X[tr]), 1e-6, None), ya[tr])

    # rho 优化（验证集：2025-06-30 ~ test_start）
    va = ((df["date"] >= pd.Timestamp("2025-06-30")) & (df["date"] <= test_start)).values
    lh_v = np.maximum(ch.predict(np.clip(mh.predict(X[va]), 1e-6, None)), 1e-6)
    la_v = np.maximum(ca.predict(np.clip(ma.predict(X[va]), 1e-6, None)), 1e-6)
    yh_v, ya_v, rg_v = yh[va], ya[va], rang[va]
    best_rho, best_ll = 0.0, 1e9
    for rho in [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]:
        lls = []
        for i in range(va.sum()):
            g = build_grid(lh_v[i], la_v[i], rho)
            hw, dr, aw = range_probs_from_grid(g, int(round(rg_v[i])))
            p_actual = hw if yh_v[i] + rg_v[i] > ya_v[i] else (dr if yh_v[i] + rg_v[i] == ya_v[i] else aw)
            lls.append(-np.log(max(p_actual, 1e-12)))
        ll = float(np.mean(lls))
        if ll < best_ll:
            best_ll, best_rho = ll, rho
    print(f"[rho] 验证集最优 ρ={best_rho} (让球LL={best_ll:.4f})")

    # 测试集评估
    lh = np.maximum(ch.predict(np.clip(mh.predict(X[te]), 1e-6, None)), 1e-6)
    la = np.maximum(ca.predict(np.clip(ma.predict(X[te]), 1e-6, None)), 1e-6)
    yh_t, ya_t, rg_t = yh[te], ya[te], rang[te]
    n = te.sum()
    pm = np.zeros((n, 3))
    for i in range(n):
        hw, dr, aw = range_probs_from_grid(build_grid(lh[i], la[i], best_rho), int(round(rg_t[i])))
        pm[i] = [hw, dr, aw]

    yoh = np.zeros((n, 3))
    yoh[yh_t + rg_t > ya_t, 0] = 1
    yoh[yh_t + rg_t == ya_t, 1] = 1
    yoh[yh_t + rg_t < ya_t, 2] = 1

    base = range_odds_baseline(df[te])
    valid_b = np.isfinite(base).all(axis=1)
    ll_m = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
    ll_b = float(-np.mean(np.sum(yoh[valid_b] * np.log(np.clip(base[valid_b], 1e-12, 1)), axis=1)))
    br_m = float(np.mean(np.sum((pm - yoh) ** 2, axis=1)))
    br_b = float(np.mean(np.sum((base[valid_b] - yoh[valid_b]) ** 2, axis=1)))
    acc_m = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
    acc_b = float(np.mean(np.argmax(base[valid_b], 1) == np.argmax(yoh[valid_b], 1)))

    print("\n" + "=" * 60)
    print(f"让球胜平负评估（测试集 {n} 场, 基线样本 {valid_b.sum()}）")
    print("=" * 60)
    print(f"{'指标':<12}{'模型':<12}{'让球SP基线':<12}{'提升':<10}")
    print(f"{'Log Loss':<12}{ll_m:<12.4f}{ll_b:<12.4f}{ll_b-ll_m:<+10.4f}")
    print(f"{'Brier':<12}{br_m:<12.4f}{br_b:<12.4f}{br_b-br_m:<+10.4f}")
    print(f"{'命中率':<12}{acc_m:<12.3f}{acc_b:<12.3f}{acc_m-acc_b:<+10.3f}")

    # 分层：按让球方向
    print("\n按让球方向分层（模型 Acc）:")
    for label, mask in [("主让(rang<0)", rg_t < 0), ("客让(rang>0)", rg_t > 0), ("平手(rang=0)", rg_t == 0)]:
        if mask.sum() < 30:
            continue
        print(f"  {label}: {mask.sum()}场 模型Acc={np.mean(np.argmax(pm[mask],1)==np.argmax(yoh[mask],1)):.3f} "
              f"基线Acc={np.mean(np.argmax(base[mask],1)==np.argmax(yoh[mask],1)):.3f}")

    # 特征重要性
    imp = pd.DataFrame({"feature": cols + miss_cols,
                        "importance": mh.feature_importance("gain")})
    imp = imp.sort_values("importance", ascending=False).head(15)
    print("\n特征重要性 Top15:")
    print(imp.to_string(index=False))

    # 保存模型（方案A 让球版）
    import os
    os.makedirs("models_v2", exist_ok=True)
    mh.save_model("models_v2/lgb_h.txt")
    ma.save_model("models_v2/lgb_a.txt")
    with open("models_v2/calibrators.pkl", "wb") as f:
        pickle.dump({"cal_h": ch, "cal_a": ca, "rho": best_rho}, f)
    with open("models_v2/feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols + miss_cols))
    meta = {"n_train": int(tr.sum()), "n_test": int(n), "rho": float(best_rho),
            "ll_model": float(ll_m), "ll_base": float(ll_b), "brier_model": float(br_m),
            "brier_base": float(br_b), "acc_model": float(acc_m), "acc_base": float(acc_b),
            "mode": "range_spf"}
    with open("models_v2/train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("\n→ 模型已保存 models_v2/")


if __name__ == "__main__":
    main()
