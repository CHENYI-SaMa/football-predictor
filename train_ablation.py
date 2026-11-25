# -*- coding: utf-8 -*-
"""
train_ablation.py —— 消融验证：半场比分/射门特征是否提升胜平负预测
训练两个模型（近5年五大联赛 8908 场）：
  A: 全特征（含半场/射门/欧赔）
  B: 剔除半场/射门特征
时序分割评估：胜平负 Acc / LogLoss / 比分 / 大2.5
"""
import glob
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.features import FeatureBuilder, build_features
from src.model import PoissonLightGBM
from src.evaluation import evaluate_all
from src.poisson_utils import derive_probs

# 半场/射门相关特征（剔除用）
SHOT_HALF_FEATS = {"home_hg_365d", "home_ag_365d", "home_hg_180d", "home_ag_180d",
                   "home_hg_90d", "home_ag_90d", "away_hg_365d", "away_ag_365d",
                   "away_hg_180d", "away_ag_180d", "away_hg_90d", "away_ag_90d",
                   "home_hs_365d", "home_as_365d", "home_hst_365d", "home_ast_365d",
                   "away_hs_365d", "away_as_365d", "away_hst_365d", "away_ast_365d",
                   "home_hs_180d", "home_as_180d", "away_hs_180d", "away_as_180d",
                   "h2h_home_wins", "h2h_draws", "h2h_away_wins",
                   "h2h_home_goals", "h2h_away_goals", "h2h_n"}


def load_all():
    files = sorted(glob.glob("data/football_history/*_2[1-6]*.csv"))
    rows = []
    for f in files:
        d = pd.read_csv(f, encoding="utf-8-sig")
        d = d.rename(columns={"FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr",
                              "HTHG": "htg", "HTAG": "htag", "HTR": "htr",
                              "HS": "hs", "AS": "as", "HST": "hst", "AST": "ast",
                              "HC": "hc", "AC": "ac", "HY": "hy", "AY": "ay",
                              "HR": "hr", "AR": "ar",
                              "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
                              "Date": "date", "Div": "div",
                              "HomeTeam": "home_team", "AwayTeam": "away_team"})
        rows.append(d)
    raw = pd.concat(rows, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["date", "fthg", "ftag"]).sort_values("date").reset_index(drop=True)
    raw["league"] = raw["div"]
    raw["ahh"] = np.nan
    # 大小球赔率（football-data 无此列，置 NaN；_odds_feats 需要浮点）
    raw["ou_over"] = np.nan
    raw["ou_under"] = np.nan
    for c in ("b365h", "b365d", "b365a", "htg", "htag", "hs", "as", "hst", "ast",
              "hc", "ac", "hy", "ay", "hr", "ar"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def evaluate(yh, ya, grids, lam_h, lam_a):
    res = evaluate_all(yh, ya, grids, lam_h, lam_a)
    # 胜平负概率从比分网格计算
    n = len(yh)
    ph = np.zeros(n); pd_ = np.zeros(n); pa = np.zeros(n)
    for i in range(n):
        g = grids[i]
        ih, ja = np.indices(g.shape)
        ph[i] = g[ih > ja].sum()
        pd_[i] = g[ih == ja].sum()
        pa[i] = g[ih < ja].sum()
    yoh = np.zeros((n, 3))
    yoh[yh > ya, 0] = 1; yoh[yh == ya, 1] = 1; yoh[yh < ya, 2] = 1
    pm = np.stack([ph, pd_, pa], axis=1)
    acc = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
    ll = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
    return acc, ll, res


def run(cols_keep, tag, feat, tr, va, te, meta=None):
    t0 = time.time()
    model = PoissonLightGBM()
    model.fit(tr[cols_keep].values, tr["fthg"].values.astype(float),
              tr["ftag"].values.astype(float),
              va[cols_keep].values, va["fthg"].values.astype(float),
              va["ftag"].values.astype(float),
              num_boost_round=2000, early_stopping=100)
    print(f"\n===== 模型[{tag}] 训练 {time.time()-t0:.1f}s, 特征 {len(cols_keep)} =====")
    for name, df in (("验证", va), ("测试", te)):
        grids = model.predict_grids(df[cols_keep].values)
        yh = df["fthg"].values.astype(float); ya = df["ftag"].values.astype(float)
        lam_h = model.predict_lambda_h(df[cols_keep].values)
        lam_a = model.predict_lambda_a(df[cols_keep].values)
        acc, ll, res = evaluate(yh, ya, grids, lam_h, lam_a)
        tgd = derive_probs(grids[0])["total_goals_distribution"]
        over = np.mean([sum(p for g, p in derive_probs(grids[i])["total_goals_distribution"].items() if g >= 3) > 0.5 for i in range(len(yh))])
        print(f"[{name}] 胜平负Acc={acc:.3f} LL={ll:.4f} | 比分LL={res['log_loss']:.4f} "
              f"MLS={res['mean_log_score']:.3f} 大2.5≈{over:.1%}")


def main():
    t0 = time.time()
    raw = load_all()
    print(f"数据: {len(raw)} 场, {raw['date'].min().date()} ~ {raw['date'].max().date()}", flush=True)
    feat, state = build_features(raw)
    print(f"特征: {feat.shape}, 耗时 {time.time()-t0:.1f}s", flush=True)
    maxd = feat["date"].max()
    ts = maxd - pd.Timedelta(days=30)
    vs = maxd - pd.Timedelta(days=90)
    tr = feat[feat["date"] <= vs].copy()
    va = feat[(feat["date"] > vs) & (feat["date"] <= ts)].copy()
    te = feat[feat["date"] > ts].copy()
    print(f"分割: 训练{len(tr)} 验证{len(va)} 测试{len(te)}", flush=True)

    drop_cols = {"fthg", "ftag", "ftr", "date", "league", "home_team", "away_team", "htg", "htag", "htr",
                 "hs", "as", "hst", "ast", "hc", "ac", "hy", "ay", "hr", "ar", "div", "ahh"}
    cols_all = [c for c in feat.columns if c not in drop_cols]
    cols_no_half = [c for c in cols_all if c not in SHOT_HALF_FEATS]
    print(f"全特征 {len(cols_all)} | 剔除后 {len(cols_no_half)}")

    run(cols_all, "A-全特征(含半场/射门)", feat, tr, va, te)
    run(cols_no_half, "B-剔除半场/射门", feat, tr, va, te)


if __name__ == "__main__":
    main()
