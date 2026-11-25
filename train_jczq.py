# -*- coding: utf-8 -*-
"""
train_jczq.py —— 500 竞彩专属模型训练
=====================================
用 500 网历史开奖（日职/荷甲/德乙/欧罗巴/韩职/挪超等竞彩常见赛事）
训练独立泊松模型，预测 500 网比赛时优先使用。

数据：data/jczq_history.csv（src/jczq_history.py 抓取）
特征：复用 src/features.py（滚动统计/ELO/赔率——赔率用竞彩让球SP）
模型：复用 src/model.py PoissonLightGBM（Poisson NLL + isotonic + ρ + 温度）

用法：
    python train_jczq.py                  # 默认参数训练
    python train_jczq.py --trials 10      # 带 Optuna 超参搜索
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import LEAGUE_NAMES
from src.features import build_features
from src.model import PoissonLightGBM
from src.poisson_utils import MAX_GOALS
from src.evaluation import evaluate_all, odds_baseline_probs

BASE_DIR = Path(__file__).resolve().parent
HIST_CSV = BASE_DIR / "data" / "jczq_history.csv"
OUT_DIR = BASE_DIR / "models_jczq"


def main():
    parser = argparse.ArgumentParser(description="500 竞彩模型训练")
    parser.add_argument("--trials", type=int, default=0,
                        help="Optuna trials（0=跳过搜索用默认参数）")
    parser.add_argument("--val-months", type=int, default=2,
                        help="验证集用最近 N 个月，测试集用最近 1 个月")
    args = parser.parse_args()

    if not HIST_CSV.exists():
        print(f"缺少 {HIST_CSV}，请先运行: python src/jczq_history.py --days 365")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(HIST_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    print(f"[数据] {len(raw):,} 场, {raw['date'].min().date()} ~ {raw['date'].max().date()}")
    print(f"[数据] 联赛: {raw['league'].value_counts().head(12).to_dict()}")

    t0 = time.time()
    feat, state = build_features(raw)
    print(f"[特征] {feat.shape[0]:,} 行 × {feat.shape[1]} 列, 耗时 {time.time()-t0:.1f}s")
    # 让球数并入特征表（让球胜平负评估用；CSV 中让球存于 ahh 列）
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)

    # ---- 时序分割：最近 1 个月测试，前 1 个月验证，其余训练 ----
    max_date = feat["date"].max()
    test_start = max_date - pd.Timedelta(days=30)
    val_start = max_date - pd.Timedelta(days=30 + 30 * args.val_months)
    tr = feat[feat["date"] <= val_start].copy()
    va = feat[(feat["date"] > val_start) & (feat["date"] <= test_start)].copy()
    te = feat[feat["date"] > test_start].copy()
    print(f"[分割] 训练 {len(tr):,} | 验证 {len(va):,} | 测试 {len(te):,} "
          f"(测试 ≥ {test_start.date()})")

    cols = [c for c in feat.columns if c not in
            {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}]
    X_tr = tr[cols].values
    y_h_tr, y_a_tr = tr["fthg"].values.astype(float), tr["ftag"].values.astype(float)
    X_va = va[cols].values
    y_h_va, y_a_va = va["fthg"].values.astype(float), va["ftag"].values.astype(float)
    X_te = te[cols].values
    y_h_te, y_a_te = te["fthg"].values.astype(float), te["ftag"].values.astype(float)

    # ---- Optuna（可选） ----
    best_params = {}
    if args.trials > 0:
        import optuna
        from train import objective
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda t: objective(t, X_tr, y_h_tr, y_a_tr, X_va, y_h_va, y_a_va),
            n_trials=args.trials, show_progress_bar=True)
        best_params = study.best_params
        print(f"[Optuna] 最佳验证 NLL={study.best_value:.4f} {best_params}")

    # ---- 训练 ----
    model = PoissonLightGBM(base_params=best_params)
    t0 = time.time()
    model.fit(X_tr, y_h_tr, y_a_tr, X_va, y_h_va, y_a_va,
              num_boost_round=2000, early_stopping=100)
    print(f"[训练] {time.time()-t0:.1f}s, ρ={model.rho:.4f}, T={model.temperature:.3f}")

    # ---- 评估（赛果 + 让球胜平负 vs 让球SP基线） ----
    import numpy as np

    def eval_set(name, X, y_h, y_a, odds_df):
        grids = model.predict_grids(X)
        res = evaluate_all(y_h, y_a, grids,
                           model.predict_lambda_h(X), model.predict_lambda_a(X))
        # ---- 让球胜平负（竞彩主玩法）：主队让球后进球 = fthg + rang ----
        rang = odds_df["rang"].values
        yh_r = y_h + rang
        yoh_r = np.zeros((len(y_h), 3))
        yoh_r[yh_r > y_a, 0] = 1.0
        yoh_r[yh_r == y_a, 1] = 1.0
        yoh_r[yh_r < y_a, 2] = 1.0
        ih, ja = np.indices((grids.shape[1], grids.shape[2]))
        shifted = ih[None, :, :] + rang[:, None, None]
        p_r = np.clip(np.stack([
            np.sum(grids * (shifted > ja), axis=(1, 2)),
            np.sum(grids * (shifted == ja), axis=(1, 2)),
            np.sum(grids * (shifted < ja), axis=(1, 2)),
        ], axis=1), 1e-12, 1.0)
        res["range_log_loss"] = float(-np.mean(np.sum(yoh_r * np.log(p_r), axis=1)))
        res["range_acc"] = float(np.mean(np.argmax(p_r, axis=1) == np.argmax(yoh_r, axis=1)))
        # 基线：让球SP 去抽水（特征表 odds_h/d/a 即让球SP）
        valid = odds_df["odds_h"].notna().values & \
                odds_df["odds_d"].notna().values & \
                odds_df["odds_a"].notna().values
        if valid.any():
            o = odds_baseline_probs(odds_df["odds_h"].values[valid],
                                    odds_df["odds_d"].values[valid],
                                    odds_df["odds_a"].values[valid])
            res["range_odds_ll"] = float(-np.mean(np.sum(
                yoh_r[valid] * np.log(np.clip(o, 1e-12, 1.0)), axis=1)))
            res["range_odds_acc"] = float(np.mean(np.argmax(o, axis=1) ==
                                                  np.argmax(yoh_r[valid], axis=1)))
            res["odds_n"] = int(valid.sum())
        else:
            res["range_odds_ll"] = res["range_odds_acc"] = float("nan")
            res["odds_n"] = 0
        print(f"[{name}] 赛果LL={res['log_loss']:.4f} | "
              f"让球LL={res['range_log_loss']:.4f}(基线{res['range_odds_ll']:.4f}) "
              f"让球Acc={res['range_acc']:.3f}(基线{res['range_odds_acc']:.3f},"
              f"n={res['odds_n']}) Brier={res['brier']:.4f} "
              f"MLS={res['mean_log_score']:.3f} MAE_h={res['goals_h_mae']:.3f} "
              f"cov_h={res['interval_cov_h']:.3f}")
        return res

    res_te = eval_set("测试集", X_te, y_h_te, y_a_te, te)
    res_va = eval_set("验证集", X_va, y_h_va, y_a_va, va)

    # ---- 保存 ----
    model.save(OUT_DIR / "model.pkl")
    import pickle
    with open(OUT_DIR / "feature_state.pkl", "wb") as f:
        pickle.dump(state, f)
    meta = {
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "best_params": best_params, "rho": model.rho,
        "temperature": model.temperature, "feature_cols": cols,
        "metrics_test": res_te, "metrics_val": res_va,
        "data_range": [str(raw["date"].min().date()), str(raw["date"].max().date())],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_DIR / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[保存] 模型 -> {OUT_DIR/'model.pkl'}")


if __name__ == "__main__":
    main()
