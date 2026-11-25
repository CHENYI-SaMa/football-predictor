# -*- coding: utf-8 -*-
"""半全场预测模型：4 输出 LightGBM 泊松
=====================================
输出 4 个 λ：λ_h1(上半场主队) / λ_a1(上半场客队) / λ_h2(下半场主队) / λ_a2(下半场客队)
- 标签：htg/htag（真实半场，仅五大联赛 3234 场可用）；下半场 = fthg-htg, ftag-htag
- 训练仅在 htg 有效的样本上进行（避免用均值填充的假半场标签）
- 时序切分与全场模型一致：2025-06-30 前训练 / 后验证 / 最近1年测试
- 半全场 9 结果概率：枚举上/下半场比分矩阵卷积导出
用法: python train_hf.py [--trials 0]
"""
import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_HALF = 5   # 半场比分网格上限
MAX_FULL = 10  # 全场比分网格上限

# 半全场 9 结果：半场(胜/平/负) + 全场(胜/平/负)
HTFT_KEYS = ["胜胜", "胜平", "胜负", "平胜", "平平", "平负", "负胜", "负平", "负负"]


def build_half_grid(lh, la, n=MAX_HALF):
    """独立泊松半场比分矩阵。"""
    ph = poisson.pmf(np.arange(n + 1), max(lh, 1e-6))
    pa = poisson.pmf(np.arange(n + 1), max(la, 1e-6))
    g = np.outer(ph, pa)
    s = g.sum()
    return g / s if s > 0 else g


def htft_probs(lam_h1, lam_a1, lam_h2, lam_a2):
    """从 4λ 导出半全场 9 结果概率 + 全场比分矩阵。
    原理：全场比分 = 上半场比分 + 下半场比分（假设上下半场条件独立）。
    """
    g1 = build_half_grid(lam_h1, lam_a1)   # 上半场 (n+1)x(n+1)
    g2 = build_half_grid(lam_h2, lam_a2)   # 下半场
    n1 = g1.shape[0]
    # 全场比分矩阵（卷积）
    full = np.zeros((MAX_FULL + 1, MAX_FULL + 1))
    htft = {k: 0.0 for k in HTFT_KEYS}
    for i in range(n1):
        for j in range(n1):
            p_ht = g1[i, j]
            if p_ht <= 0:
                continue
            ht_res = 0 if i > j else (1 if i == j else 2)  # 半场胜/平/负
            for k in range(n1):
                for l in range(n1):
                    p = p_ht * g2[k, l]
                    if p <= 0:
                        continue
                    fi, fj = i + k, j + l
                    if fi <= MAX_FULL and fj <= MAX_FULL:
                        full[fi, fj] += p
                    ft_res = 0 if fi > fj else (1 if fi == fj else 2)
                    htft[HTFT_KEYS[ht_res * 3 + ft_res]] += p
    s = full.sum()
    if s > 0:
        full /= s
    # 归一化 htft
    s2 = sum(htft.values())
    if s2 > 0:
        htft = {k: v / s2 for k, v in htft.items()}
    return htft, full


def train_one(X_tr, y_tr, X_va, y_va, params, n_rounds=500):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=0)
    args = ap.parse_args()

    # ---- 数据 ----
    df = pd.read_csv("data/features.csv", parse_dates=["date"])
    with open("data/feature_state.pkl", "rb") as f:
        state = pickle.load(f)
    cols = state["feature_cols"]
    miss_cols = [c + "_miss" for c in cols]
    X_all = df[cols + miss_cols].values

    # 仅使用真实半场标签样本
    valid = df["htg"].notna() & df["htag"].notna()
    print(f"总样本 {len(df)}，真实半场标签 {valid.sum()} 场")

    # 下半场标签
    h2 = df["fthg"] - df["htg"]
    a2 = df["ftag"] - df["htag"]
    y = {
        "h1": df["htg"].values.astype(float),
        "a1": df["htag"].values.astype(float),
        "h2": h2.values.astype(float),
        "a2": a2.values.astype(float),
    }

    # ---- 时序切分 ----
    # 注意：半全场标签仅五大联赛有，而五大联赛 6-8 月是休赛期，
    # 验证窗口须避开休赛期 → 验证=测试前一整段赛季末（2025-04-01~08-07）
    test_start = df["date"].max() - pd.Timedelta(days=365)
    val_start = pd.Timestamp("2025-04-01")
    tr = (df["date"] < val_start).values & valid.values
    va = ((df["date"] >= val_start) & (df["date"] <= test_start)).values & valid.values
    te = (df["date"] > test_start).values & valid.values
    print(f"划分(仅真实半场): 训练 {tr.sum()} / 验证 {va.sum()} / 测试 {te.sum()} "
          f"(测试 {df.loc[te,'date'].min().date()} ~ {df.loc[te,'date'].max().date()})")

    params = {"objective": "poisson", "num_leaves": 50, "learning_rate": 0.036,
              "min_child_samples": 88, "feature_fraction": 0.735, "lambda_l2": 1.96,
              "verbosity": -1}
    if args.trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = {"objective": "poisson",
                 "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                 "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                 "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                 "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                 "verbosity": -1}
            nll = 0.0
            for k in ("h1", "a1", "h2", "a2"):
                m = train_one(X_all[tr], y[k][tr], X_all[va], y[k][va], p)
                pred = np.clip(m.predict(X_all[va]), 1e-6, None)
                nll += float(np.mean(pred - y[k][va] * np.log(pred)))
            return nll

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.trials)
        params = {"objective": "poisson", "verbosity": -1, **study.best_params}
        print(f"[Optuna] 最优 NLL={study.best_value:.4f}")

    # ---- 训练 4 个模型 ----
    models = {}
    cals = {}
    for k in ("h1", "a1", "h2", "a2"):
        models[k] = train_one(X_all[tr], y[k][tr], X_all[va], y[k][va], params)
        cals[k] = IsotonicRegression(out_of_bounds="clip")
        cals[k].fit(np.clip(models[k].predict(X_all[tr]), 1e-6, None), y[k][tr])

    def predict_lams(X):
        out = {}
        for k in ("h1", "a1", "h2", "a2"):
            raw = np.clip(models[k].predict(X), 1e-6, None)
            out[k] = np.maximum(cals[k].predict(raw), 1e-6)
        return out

    # ---- 测试集评估 ----
    lams = predict_lams(X_all[te])
    n_te = te.sum()
    htft_ll = 0.0
    full_pm = np.zeros((n_te, 3))   # 全场胜平负（由卷积导出）
    yoh_full = np.zeros((n_te, 3))
    yoh_htft = np.zeros((n_te, 9))
    yh_t, ya_t = df["fthg"].values[te], df["ftag"].values[te]
    htg_t, htag_t = df["htg"].values[te], df["htag"].values[te]

    for i in range(n_te):
        htft, full = htft_probs(lams["h1"][i], lams["a1"][i],
                                lams["h2"][i], lams["a2"][i])
        # 半全场真值索引
        ht_res = 0 if htg_t[i] > htag_t[i] else (1 if htg_t[i] == htag_t[i] else 2)
        ft_res = 0 if yh_t[i] > ya_t[i] else (1 if yh_t[i] == ya_t[i] else 2)
        idx = ht_res * 3 + ft_res
        yoh_htft[i, idx] = 1
        # 全场胜平负（从 full 矩阵）
        n = full.shape[0]
        ih, ja = np.indices((n, n))
        full_pm[i] = [full[ih > ja].sum(), full[ih == ja].sum(), full[ih < ja].sum()]
        yoh_full[i, 0 if yh_t[i] > ya_t[i] else (1 if yh_t[i] == ya_t[i] else 2)] = 1
        # 半全场 LL
        htft_arr = np.array([htft[k] for k in HTFT_KEYS])
        htft_ll += -np.log(max(htft_arr[idx], 1e-12))
    htft_ll /= n_te

    # 全场指标（卷积导出 vs 原全场模型？此处仅报告卷积版）
    ll_full = float(-np.mean(np.sum(yoh_full * np.log(np.clip(full_pm, 1e-12, 1)), axis=1)))
    acc_full = float(np.mean(np.argmax(full_pm, 1) == np.argmax(yoh_full, 1)))
    acc_htft = float(np.mean(np.argmax(yoh_htft, 1) == np.argmax(
        np.stack([np.array([htft[k] for k in HTFT_KEYS]) for _ in range(1)], axis=0)[0]
        if False else yoh_htft, 1))) if False else None

    print("\n" + "=" * 62)
    print(f"半全场模型评估（测试集 {n_te} 场，真实半场标签）")
    print("=" * 62)
    print(f"半全场 9分类 Log Loss: {htft_ll:.4f}")
    print(f"全场胜平负(卷积导出) Log Loss: {ll_full:.4f} | Acc: {acc_full:.3f}")
    # 各半全场结果分布
    print("\n半全场结果实际分布:")
    for j, k in enumerate(HTFT_KEYS):
        print(f"  {k}: {yoh_htft[:, j].mean():.1%}", end="  ")
    print()

    # ---- 保存 ----
    os.makedirs("models_hf", exist_ok=True)
    for k in ("h1", "a1", "h2", "a2"):
        models[k].save_model(f"models_hf/lgb_{k}.txt")
    with open("models_hf/calibrators.pkl", "wb") as f:
        pickle.dump(cals, f)
    with open("models_hf/feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols + miss_cols))
    with open("models_hf/train_meta.json", "w", encoding="utf-8") as f:
        json.dump({"n_train": int(tr.sum()), "n_val": int(va.sum()),
                   "n_test": int(n_te), "htft_ll": float(htft_ll),
                   "full_ll": float(ll_full), "full_acc": float(acc_full)},
                  f, ensure_ascii=False, indent=2)
    print("\n→ 模型已保存 models_hf/ (lgb_h1/a1/h2/a2.txt + calibrators + feature_columns + train_meta)")


if __name__ == "__main__":
    main()
