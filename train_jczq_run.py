# -*- coding: utf-8 -*-
"""分阶段训练 500 竞彩模型（调试用，正常入口为 train_jczq.py）"""
import sys
import time

sys.path.insert(0, ".")
t0 = time.time()
print("[阶段1] 加载数据", flush=True)
import pandas as pd
raw = pd.read_csv("data/jczq_history.csv", parse_dates=["date"])
raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
print(f"  {len(raw)} 场, 耗时 {time.time()-t0:.1f}s", flush=True)

print("[阶段2] 特征构建", flush=True)
from src.features import build_features
t0 = time.time()
feat, state = build_features(raw)
print(f"  {feat.shape}, 耗时 {time.time()-t0:.1f}s", flush=True)

print("[阶段3] 时序分割", flush=True)
# 把让球数并入特征表（评估让球胜平负用；jczq_history.csv 中让球存于 ahh 列）
raw = raw.sort_values("date").reset_index(drop=True)
feat["rang"] = raw["ahh"].values.astype(float)
max_date = feat["date"].max()
test_start = max_date - pd.Timedelta(days=30)
val_start = max_date - pd.Timedelta(days=90)
tr = feat[feat["date"] <= val_start].copy()
va = feat[(feat["date"] > val_start) & (feat["date"] <= test_start)].copy()
te = feat[feat["date"] > test_start].copy()
print(f"  train={len(tr)} val={len(va)} test={len(te)}", flush=True)

cols = [c for c in feat.columns if c not in {"fthg", "ftag", "date", "league", "home_team", "away_team"}]
X_tr = tr[cols].values
y_h_tr = tr["fthg"].values.astype(float)
y_a_tr = tr["ftag"].values.astype(float)
X_va = va[cols].values
y_h_va = va["fthg"].values.astype(float)
y_a_va = va["ftag"].values.astype(float)
X_te = te[cols].values
y_h_te = te["fthg"].values.astype(float)
y_a_te = te["ftag"].values.astype(float)

print("[阶段4] LightGBM 训练", flush=True)
from src.model import PoissonLightGBM
t0 = time.time()
model = PoissonLightGBM()
model.fit(X_tr, y_h_tr, y_a_tr, X_va, y_h_va, y_a_va,
          num_boost_round=2000, early_stopping=100)
print(f"  完成 {time.time()-t0:.1f}s rho={model.rho:.4f} T={model.temperature:.3f}", flush=True)

print("[阶段5] 评估", flush=True)
import numpy as np
from src.evaluation import evaluate_all, odds_baseline_probs


def ev(name, X, yh, ya, odf):
    grids = model.predict_grids(X)
    res = evaluate_all(yh, ya, grids, model.predict_lambda_h(X), model.predict_lambda_a(X))
    # 让球胜平负（竞彩主玩法）：主队让球后进球 = fthg + rang
    rang = odf["rang"].values
    yh_r = yh + rang
    yoh_r = np.zeros((len(yh), 3))
    yoh_r[yh_r > ya, 0] = 1
    yoh_r[yh_r == ya, 1] = 1
    yoh_r[yh_r < ya, 2] = 1
    # 模型让球概率（网格平移）
    ih, ja = np.indices((grids.shape[1], grids.shape[2]))
    shifted = ih[None, :, :] + rang[:, None, None]
    ph_r = np.sum(grids * (shifted > ja), axis=(1, 2))
    pd_r = np.sum(grids * (shifted == ja), axis=(1, 2))
    pa_r = np.sum(grids * (shifted < ja), axis=(1, 2))
    p_r = np.clip(np.stack([ph_r, pd_r, pa_r], axis=1), 1e-12, 1)
    res["range_log_loss"] = float(-np.mean(np.sum(yoh_r * np.log(p_r), axis=1)))
    res["range_acc"] = float(np.mean(np.argmax(p_r, axis=1) == np.argmax(yoh_r, axis=1)))
    # 基线：让球SP 去抽水（特征表 odds_h/d/a 即让球SP）
    valid = odf["odds_h"].notna().values & odf["odds_d"].notna().values & \
            odf["odds_a"].notna().values
    if valid.any():
        o = odds_baseline_probs(odf["odds_h"].values[valid],
                                odf["odds_d"].values[valid],
                                odf["odds_a"].values[valid])
        res["range_odds_ll"] = float(-np.mean(np.sum(
            yoh_r[valid] * np.log(np.clip(o, 1e-12, 1)), axis=1)))
        res["range_odds_acc"] = float(np.mean(np.argmax(o, axis=1) ==
                                              np.argmax(yoh_r[valid], axis=1)))
        res["odds_n"] = int(valid.sum())
    else:
        res["range_odds_ll"] = res["range_odds_acc"] = float("nan")
        res["odds_n"] = 0
    print(f"  [{name}] 赛果LL={res['log_loss']:.4f} | 让球LL={res['range_log_loss']:.4f}"
          f"(基线{res['range_odds_ll']:.4f}) 让球Acc={res['range_acc']:.3f}"
          f"(基线{res['range_odds_acc']:.3f},n={res['odds_n']}) "
          f"Brier={res['brier']:.4f} MLS={res['mean_log_score']:.3f} "
          f"MAE_h={res['goals_h_mae']:.3f} cov_h={res['interval_cov_h']:.3f}", flush=True)
    return res


res_te = ev("测试集", X_te, y_h_te, y_a_te, te)
res_va = ev("验证集", X_va, y_h_va, y_a_va, va)

print("[阶段6] 保存", flush=True)
import json
import pickle
model.save("models_jczq/model.pkl")
with open("models_jczq/feature_state.pkl", "wb") as f:
    pickle.dump(state, f)
meta = {"n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "rho": model.rho, "temperature": model.temperature,
        "feature_cols": cols, "metrics_test": res_te, "metrics_val": res_va}
with open("models_jczq/train_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
print("全部完成", flush=True)
