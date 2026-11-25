# -*- coding: utf-8 -*-
"""train_v3.py —— 500 竞彩模型 V3 训练（追加 xG/主客场/趋势/情境特征）
======================================================================
数据: data/jczq_history_xg.csv（merge_xg.py 产出，含 Understat 真实 xG）
特征: src/features_v3.FeatureBuilderV3（v1 全部 + xG族/主客场分窗/趋势斜率/
      周中周末/ahh×elo 交互）
训练: PoissonLightGBM（双输出泊松 + isotonic + ρ + 温度），时序分割
评估: 赛果/让球胜平负 vs 让球SP基线 + 与 v1 模型测试集对比
保存: models_v3/（lgb_h/lgb_a/calibrators/feature_columns/train_meta）
用法: python train_v3.py [--trials 0] [--val-months 2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import LEAGUE_NAMES                 # noqa: E402
from src.features_v3 import FeatureBuilderV3             # noqa: E402
from src.model import PoissonLightGBM                    # noqa: E402
from src.poisson_utils import MAX_GOALS                  # noqa: E402
from src.evaluation import evaluate_all, odds_baseline_probs  # noqa: E402

HIST_CSV = BASE_DIR / "data" / "jczq_history_xg.csv"
OUT_DIR = BASE_DIR / "models_v3"


def build_features_v3(data: pd.DataFrame):
    fb = FeatureBuilderV3(data)
    return fb.build()


def main():
    parser = argparse.ArgumentParser(description="500 竞彩模型 V3 训练")
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--val-months", type=int, default=2)
    args = parser.parse_args()

    if not HIST_CSV.exists():
        print(f"缺少 {HIST_CSV}，请先运行: python merge_xg.py")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(HIST_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    print(f"[数据] {len(raw):,} 场, {raw['date'].min().date()} ~ {raw['date'].max().date()}")
    print(f"[数据] xG 覆盖: {raw['xg_h'].notna().mean():.1%} "
          f"({int(raw['xg_h'].notna().sum())} 场)")

    t0 = time.time()
    feat, state = build_features_v3(raw)
    print(f"[特征] {feat.shape[0]:,} 行 × {feat.shape[1]} 列, 耗时 {time.time()-t0:.1f}s")
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)
    # xG 缺失率展示
    xg_cols = [c for c in feat.columns if c.startswith(("home_xg", "away_xg")) and "_miss" not in c]
    if xg_cols:
        print(f"[特征] xG 特征缺失率: {feat[xg_cols[0]].isna().mean():.1%}（非五大联赛无xG，模型自动忽略）")

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
    # 排除 form 特征（A/B 验证负贡献：含form 57.8% vs 去form 60.2%；
    # 与 v1 近3/5/8场积分/进球特征冗余，模块 src/feature_derived.py 保留备用）
    cols = [c for c in cols if "form" not in c]
    print(f"[特征] 参与训练特征数: {len(cols)}（已排除 form 冗余特征）")
    X_tr, y_h_tr, y_a_tr = tr[cols].values, tr["fthg"].values.astype(float), tr["ftag"].values.astype(float)
    X_va, y_h_va, y_a_va = va[cols].values, va["fthg"].values.astype(float), va["ftag"].values.astype(float)
    X_te, y_h_te, y_a_te = te[cols].values, te["fthg"].values.astype(float), te["ftag"].values.astype(float)

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

    model = PoissonLightGBM(base_params=best_params)
    t0 = time.time()
    model.fit(X_tr, y_h_tr, y_a_tr, X_va, y_h_va, y_a_va,
              num_boost_round=2000, early_stopping=100)
    print(f"[训练] {time.time()-t0:.1f}s, ρ={model.rho:.4f}, T={model.temperature:.3f}")

    def eval_set(name, X, y_h, y_a, odds_df):
        grids = model.predict_grids(X)
        res = evaluate_all(y_h, y_a, grids,
                           model.predict_lambda_h(X), model.predict_lambda_a(X))
        rang = odds_df["rang"].values
        yh_r = y_h + rang
        yoh_r = np.zeros((len(y_h), 3))
        yoh_r[yh_r > y_a, 0] = 1.0
        yoh_r[yh_r == y_a, 1] = 1.0
        yoh_r[yh_r < y_a, 2] = 1.0
        n = grids.shape[1]
        ih, ja = np.indices((n, n))
        shifted = ih[None, :, :] + rang[:, None, None]
        p_r = np.clip(np.stack([
            np.sum(grids * (shifted > ja), axis=(1, 2)),
            np.sum(grids * (shifted == ja), axis=(1, 2)),
            np.sum(grids * (shifted < ja), axis=(1, 2)),
        ], axis=1), 1e-12, 1.0)
        p_r /= p_r.sum(axis=1, keepdims=True)
        res["range_log_loss"] = float(-np.mean(np.sum(yoh_r * np.log(p_r), axis=1)))
        res["range_acc"] = float(np.mean(np.argmax(p_r, axis=1) == np.argmax(yoh_r, axis=1)))
        o = odds_baseline_probs(odds_df["odds_h"].values,
                                odds_df["odds_d"].values,
                                odds_df["odds_a"].values)
        o = np.clip(o, 1e-12, 1.0)
        yoh = np.zeros((len(y_h), 3))
        yoh[y_h > y_a, 0] = 1.0
        yoh[y_h == y_a, 1] = 1.0
        yoh[y_h < y_a, 2] = 1.0
        res["odds_ll"] = float(-np.mean(np.sum(yoh * np.log(o), axis=1)))
        res["odds_acc"] = float(np.mean(np.argmax(o, axis=1) == np.argmax(yoh, axis=1)))
        res["range_odds_ll"] = float(-np.mean(np.sum(yoh_r * np.log(o), axis=1)))
        res["range_odds_acc"] = float(np.mean(np.argmax(o, axis=1) == np.argmax(yoh_r, axis=1)))
        print(f"[{name}] 赛果LL={res['log_loss']:.4f}(基线{res['odds_ll']:.4f}) "
              f"Acc={res['accuracy']:.3f} | 让球LL={res['range_log_loss']:.4f}"
              f"(基线{res['range_odds_ll']:.4f}) 让球Acc={res['range_acc']:.3f}"
              f"(基线{res['range_odds_acc']:.3f})")
        return res

    res_te = eval_set("测试集", X_te, y_h_te, y_a_te, te)
    res_va = eval_set("验证集", X_va, y_h_va, y_a_va, va)

    # ---- 保存 ----
    model.save(OUT_DIR / "model.pkl")
    with open(OUT_DIR / "feature_columns.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cols))
    import pickle
    with open(OUT_DIR / "feature_state.pkl", "wb") as f:
        pickle.dump(state, f)   # 单场预测（build_single）需要增量状态
    print(f"[保存] feature_state.pkl（球队历史状态 {len(state['team_hist'])} 队）")
    meta = {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "data_range": [str(feat["date"].min().date()), str(feat["date"].max().date())],
        "xg_cov": float(raw["xg_h"].notna().mean()),
        "best_params": best_params, "rho": float(model.rho),
        "temperature": float(model.temperature),
        "metrics_test": res_te, "metrics_val": res_va,
        "feature_cols": cols,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_DIR / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n→ 模型已保存 {OUT_DIR}/（V3：v1特征 + xG/主客场/趋势/情境）")


if __name__ == "__main__":
    main()
