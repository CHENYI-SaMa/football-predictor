# -*- coding: utf-8 -*-
"""阶段C+E: LightGBM 双输出泊松训练与评估
- 模型1: 两个 LightGBM（poisson objective = softplus 参数化 λ）分别预测 λ_h / λ_a
- 时序划分: 2024-08-01 前训练 / 之后验证 / 最近1年测试（严禁随机切分）
- Optuna 调优 num_leaves/learning_rate/min_child_samples/feature_fraction
- isotonic 校准 λ_h、λ_a
- 阶段D: Dixon-Coles 比分网格 + 胜平负/总进球分布
- 评估: Log Loss / Brier vs SP 赔率基线；特征重要性
- 保存 model.pkl + feature_columns.txt + 校准器
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from scipy.stats import poisson

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

MAX_GOALS = 10

# ---------------- 数据准备 ----------------
def load_split(features_path="data/features.csv"):
    df = pd.read_csv(features_path, parse_dates=["date"])
    with open("data/feature_state.pkl", "rb") as f:
        state = pickle.load(f)
    cols = state["feature_cols"]
    # 缺失列
    miss_cols = [c + "_miss" for c in cols]
    X = df[cols + miss_cols].values
    yh = df["fthg"].values.astype(float)
    ya = df["ftag"].values.astype(float)
    # 时序划分
    test_start = df["date"].max() - pd.Timedelta(days=365)   # 最近1年测试
    val_start = pd.Timestamp("2024-08-01")                   # 训练/验证界
    tr = (df["date"] < val_start).values
    va = ((df["date"] >= val_start) & (df["date"] <= test_start)).values
    te = (df["date"] > test_start).values
    print(f"划分: 训练 {tr.sum()} / 验证 {va.sum()} / 测试 {te.sum()} "
          f"(测试区间 {df.loc[te,'date'].min().date()} ~ {df.loc[te,'date'].max().date()})")
    return (df, cols, X, yh, ya, tr, va, te)


# ---------------- 模型训练 ----------------
def train_lambda(X_tr, y_tr, X_va, y_va, params, num_boost_round=500):
    """训练单输出 poisson 模型，early stopping 在验证集。"""
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=num_boost_round,
                      valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def train_pair(X_tr, yh_tr, ya_tr, X_va, yh_va, ya_va, params):
    m_h = train_lambda(X_tr, yh_tr, X_va, yh_va, params)
    m_a = train_lambda(X_tr, ya_tr, X_va, ya_va, params)
    return m_h, m_a


# ---------------- 比分网格 ----------------
def build_grid(lam_h, lam_a, rho=0.0):
    """独立泊松矩阵 + Dixon-Coles 修正，归一化。返回 (11,11) 概率矩阵。"""
    maxg = MAX_GOALS
    grid = np.zeros((maxg + 1, maxg + 1))
    ph = poisson.pmf(np.arange(maxg + 1), lam_h)
    pa = poisson.pmf(np.arange(maxg + 1), lam_a)
    grid = np.outer(ph, pa)
    # Dixon-Coles 修正（低比分区域）
    if rho != 0:
        for i in range(2):
            for j in range(2):
                lam_h_c = lam_h
                lam_a_c = lam_a
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
    if s > 0:
        grid /= s
    return grid


def grid_to_probs(grid):
    """网格 → 胜平负概率 + 总进球分布。"""
    n = grid.shape[0]
    idx = np.arange(n)
    ih, ja = np.meshgrid(idx, idx, indexing="ij")
    hw = grid[ih > ja].sum()
    dr = grid[ih == ja].sum()
    aw = grid[ih < ja].sum()
    tg = np.zeros(2 * n)
    for i in range(n):
        for j in range(n):
            tg[i + j] += grid[i, j]
    return hw, dr, aw, tg


# ---------------- 评估 ----------------
def log_loss_spf(yh, ya, pm):
    """胜平负 Log Loss。pm: (n,3) 主胜/平/客胜。"""
    yoh = np.zeros((len(yh), 3))
    yoh[yh > ya, 0] = 1
    yoh[yh == ya, 1] = 1
    yoh[yh < ya, 2] = 1
    return float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))


def brier_spf(yh, ya, pm):
    yoh = np.zeros((len(yh), 3))
    yoh[yh > ya, 0] = 1
    yoh[yh == ya, 1] = 1
    yoh[yh < ya, 2] = 1
    return float(np.mean(np.sum((pm - yoh) ** 2, axis=1)))


def odds_baseline(df_sub):
    """SP 赔率去抽水 → 基线概率。"""
    inv = 1.0 / df_sub[["b365h", "b365d", "b365a"]].values
    return inv / inv.sum(axis=1, keepdims=True)


# ---------------- 主流程 ----------------
def main(optuna_trials=30):
    df, cols, X, yh, ya, tr, va, te = load_split()
    miss_cols = [c + "_miss" for c in cols]

    # ---- Optuna 调优（验证集 Poisson NLL） ----
    best_params = None
    best_nll = 1e9
    if optuna_trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "objective": "poisson",
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                "verbosity": -1,
            }
            m_h = train_lambda(X[tr], yh[tr], X[va], yh[va], params)
            m_a = train_lambda(X[tr], ya[tr], X[va], ya[va], params)
            pred_h = np.clip(m_h.predict(X[va]), 1e-6, None)
            pred_a = np.clip(m_a.predict(X[va]), 1e-6, None)
            nll = float(np.mean(pred_h - yh[va] * np.log(pred_h) +
                                pred_a - ya[va] * np.log(pred_a)))
            return nll

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        best_params = study.best_params
        best_nll = study.best_value
        print(f"\n[Optuna] 最优 NLL={best_nll:.4f} 参数: {best_params}")
    else:
        best_params = {"num_leaves": 31, "learning_rate": 0.05,
                       "min_child_samples": 30, "feature_fraction": 0.8,
                       "lambda_l2": 1.0}

    params = {"objective": "poisson", "verbosity": -1, **best_params}
    m_h, m_a = train_pair(X[tr], yh[tr], ya[tr], X[va], yh[va], ya[va], params)

    # ---- isotonic 校准 λ ----
    cal_h = IsotonicRegression(out_of_bounds="clip")
    cal_h.fit(np.clip(m_h.predict(X[tr]), 1e-6, None), yh[tr])
    cal_a = IsotonicRegression(out_of_bounds="clip")
    cal_a.fit(np.clip(m_a.predict(X[tr]), 1e-6, None), ya[tr])

    def predict_lam(Xx):
        lh = cal_h.predict(np.clip(m_h.predict(Xx), 1e-6, None))
        la = cal_a.predict(np.clip(m_a.predict(Xx), 1e-6, None))
        return np.maximum(lh, 1e-6), np.maximum(la, 1e-6)

    # ---- rho 优化（验证集网格搜索） ----
    best_rho = 0.0
    best_rho_ll = 1e9
    lam_h_va, lam_a_va = predict_lam(X[va])
    yh_va = yh[va]
    ya_va = ya[va]
    n_va = va.sum()
    for rho in [0.0, 0.01, -0.01, 0.03, -0.03, 0.05, -0.05]:
        ll_list = []
        for i in range(n_va):
            g = build_grid(lam_h_va[i], lam_a_va[i], rho)
            hw, dr, aw, _ = grid_to_probs(g)
            ll_list.append(-np.log(max(hw, 1e-12)) if yh_va[i] > ya_va[i]
                            else (-np.log(max(dr, 1e-12)) if yh_va[i] == ya_va[i]
                                  else -np.log(max(aw, 1e-12))))
        ll = float(np.mean(ll_list))
        if ll < best_rho_ll:
            best_rho_ll, best_rho = ll, rho
    print(f"[rho] 验证集最优 ρ={best_rho} (LL={best_rho_ll:.4f})")

    # ---- 测试集评估 ----
    lam_h_te, lam_a_te = predict_lam(X[te])
    yh_te = yh[te]
    ya_te = ya[te]
    n_te = te.sum()
    pm_te = np.zeros((n_te, 3))
    score_ll = 0.0
    for i in range(n_te):
        g = build_grid(lam_h_te[i], lam_a_te[i], best_rho)
        hw, dr, aw, tg = grid_to_probs(g)
        pm_te[i] = [hw, dr, aw]
        # 比分平均对数概率
        actual = (int(yh_te[i]), int(ya_te[i]))
        if actual[0] <= MAX_GOALS and actual[1] <= MAX_GOALS:
            score_ll += np.log(max(g[actual[0], actual[1]], 1e-12))
    score_ll /= n_te

    # 基线
    df_te = df[te]
    base = odds_baseline(df_te)
    yoh_te = np.zeros((n_te, 3))
    yoh_te[yh_te > ya_te, 0] = 1
    yoh_te[yh_te == ya_te, 1] = 1
    yoh_te[yh_te < ya_te, 2] = 1
    ll_model = log_loss_spf(yh_te, ya_te, pm_te)
    ll_base = log_loss_spf(yh_te, ya_te, base)
    br_model = brier_spf(yh_te, ya_te, pm_te)
    br_base = brier_spf(yh_te, ya_te, base)
    acc_model = float(np.mean(np.argmax(pm_te, 1) == np.argmax(yoh_te, 1)))
    acc_base = float(np.mean(np.argmax(base, 1) == np.argmax(yoh_te, 1)))

    print("\n" + "=" * 60)
    print("测试集评估（最近1年）")
    print("=" * 60)
    print(f"{'指标':<14}{'模型':<12}{'SP基线':<12}{'提升':<10}")
    print(f"{'Log Loss':<14}{ll_model:<12.4f}{ll_base:<12.4f}{ll_base-ll_model:<+10.4f}")
    print(f"{'Brier':<14}{br_model:<12.4f}{br_base:<12.4f}{br_base-br_model:<+10.4f}")
    print(f"{'命中率':<14}{acc_model:<12.3f}{acc_base:<12.3f}{acc_model-acc_base:<+10.3f}")
    print(f"{'比分MLS':<14}{score_ll:<12.4f}{'':<12}")

    # 特征重要性
    imp = pd.DataFrame({"feature": cols + miss_cols,
                        "importance": m_h.feature_importance("gain")})
    imp = imp.sort_values("importance", ascending=False).head(20)
    print("\n特征重要性 Top10:")
    print(imp.head(10).to_string(index=False))
    imp.to_csv("outputs/feature_importance.csv", index=False, encoding="utf-8-sig")

    # ---- 保存 ----
    model_dir = "models_v2"
    import os
    os.makedirs(model_dir, exist_ok=True)
    m_h.save_model(f"{model_dir}/lgb_h.txt")
    m_a.save_model(f"{model_dir}/lgb_a.txt")
    with open(f"{model_dir}/calibrators.pkl", "wb") as f:
        pickle.dump({"cal_h": cal_h, "cal_a": cal_a, "rho": best_rho}, f)
    with open(f"{model_dir}/feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols + miss_cols))
    meta = {
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "test_start": str(df.loc[te, "date"].min().date()),
        "test_end": str(df.loc[te, "date"].max().date()),
        "rho": best_rho, "params": best_params,
        "ll_model": ll_model, "ll_base": ll_base,
        "brier_model": br_model, "brier_base": br_base,
        "acc_model": acc_model, "acc_base": acc_base,
        "score_mean_log_prob": score_ll,
    }
    with open(f"{model_dir}/train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n→ 模型已保存到 {model_dir}/ (lgb_h.txt / lgb_a.txt / calibrators.pkl / feature_columns.txt / train_meta.json)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()
    main(optuna_trials=args.trials)
