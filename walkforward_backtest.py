# -*- coding: utf-8 -*-
"""Walk-forward 时序回放回测（无未来泄漏）
=============================================
用 jczq_history.csv（2022-08 ~ 2026-08，含让球SP）逐场回放：
  每场比赛只用『该场之前』的历史构建特征（build_single + 实时状态更新），
  预测后用真实赛果更新状态 —— 完全模拟每天跑预测的真实效果。

产出场景命中率诊断：
  - 总体让球胜平负命中率（vs 市场让球SP去抽水 argmax）
  - 按让球方向 × 让球数
  - 按联赛（日职/挪超/芬超/五大联赛/欧战…）
  - 按模型置信度段（主推让球概率）
  - 按模型-市场概率差（价值差）
  目标：找出能稳定 ≥60% 命中的场景组合，支撑「选择性推荐」策略。

用法: python walkforward_backtest.py [--start 2024-01-01] [--save]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from src.features import FeatureBuilder          # noqa: E402
from src.jczq_fetcher import to_model_vector     # noqa: E402
from src.model import PoissonLightGBM            # noqa: E402


def implied(oh, od, oa):
    try:
        arr = np.array([oh, od, oa], dtype=float)
    except (TypeError, ValueError):
        return None
    if np.any(~np.isfinite(arr)) or np.any(arr <= 1):
        return None
    inv = 1.0 / arr
    return tuple(inv / inv.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-06-01",
                    help="统计起始日期（之前的比赛只喂状态不计入），默认 2023-06-01")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    # ---- 加载模型 ----
    model = PoissonLightGBM.load(BASE_DIR / "models_jczq" / "model.pkl")
    meta = json.loads((BASE_DIR / "models_jczq" / "train_meta.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]

    # ---- 加载历史数据 ----
    df = pd.read_csv(BASE_DIR / "data" / "jczq_history.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # 用 fthg/ftag 重算赛果（CSV 中 ftr 存在脏数据）
    fthg, ftag = df["fthg"], df["ftag"]
    df["ftr2"] = np.where(fthg > ftag, "H", np.where(fthg == ftag, "D", "A"))
    print(f"总场次: {len(df)}（{df['date'].min().date()} ~ {df['date'].max().date()}）")

    # ---- 逐场回放 ----
    state = {"team_hist": {}, "h2h_hist": {}}
    fb = FeatureBuilder.__new__(FeatureBuilder)   # 复用 _process_row 更新状态
    fb._feature_names = FeatureBuilder._feature_names  # 兼容
    records = []
    t0 = pd.Timestamp.now()
    for i, row in df.iterrows():
        m = row.to_dict()
        # 预测特征（不更新状态）
        row_p = pd.Series({
            "date": row["date"], "league": m["league"],
            "home_team": m["home_team"], "away_team": m["away_team"],
            "fthg": -1, "ftag": -1, "ftr": "X",
            "htg": np.nan, "htag": np.nan, "hs": np.nan, "as": np.nan,
            "hst": np.nan, "ast": np.nan,
            "b365h": m["b365h"], "b365d": m["b365d"], "b365a": m["b365a"],
            "ou_over": np.nan, "ou_under": np.nan, "ahh": float(m["ahh"]),
        })
        try:
            fdict = FeatureBuilder.build_single(state, row_p)
            X = to_model_vector(fdict, feature_cols).reshape(1, -1)
            grid = model.predict_grids(X)[0]
            lam_h = float(model.predict_lambda_h(X)[0])
            lam_a = float(model.predict_lambda_a(X)[0])
        except Exception as e:
            if i % 2000 == 0:
                print(f"  [预测失败@{row['date'].date()}] {e}")
            lam_h = lam_a = np.nan
            grid = None

        # 更新状态（用真实赛果）
        try:
            fb._process_row(state["team_hist"], state["h2h_hist"], row_p.replace(
                {"fthg": m["fthg"], "ftag": m["ftag"], "ftr": m["ftr2"]}), update_state=True)
        except Exception:
            pass

        if grid is None or row["date"] < pd.Timestamp(args.start):
            continue

        rang = int(m["ahh"]) if pd.notna(m["ahh"]) else 0
        n = grid.shape[0]
        ih, ja = np.indices((n, n))
        shifted = ih + rang
        p_rs = float(grid[shifted > ja].sum())
        p_rp = float(grid[shifted == ja].sum())
        p_rf = float(grid[shifted < ja].sum())
        pred_r = ["RS", "RP", "RF"][int(np.argmax([p_rs, p_rp, p_rf]))]
        conf = max(p_rs, p_rp, p_rf)

        yh_r = int(m["fthg"]) + rang
        act_r = "RS" if yh_r > int(m["ftag"]) else ("RP" if yh_r == int(m["ftag"]) else "RF")

        mkt = implied(m["b365h"], m["b365d"], m["b365a"])
        mkt_r = None
        mkt_conf = 0.0
        if mkt:
            mkt_r = ["RS", "RP", "RF"][int(np.argmax(mkt))]
            mkt_conf = max(mkt)

        records.append({
            "date": row["date"], "league": m["league"],
            "home": m["home_team"], "away": m["away_team"],
            "rang": rang, "score": f"{int(m['fthg'])}:{int(m['ftag'])}",
            "pred_r": pred_r, "act_r": act_r, "conf": conf,
            "p_rs": p_rs, "p_rp": p_rp, "p_rf": p_rf,
            "mkt_r": mkt_r, "mkt_conf": mkt_conf,
            "lambda": (lam_h, lam_a),
        })
        if (i + 1) % 2000 == 0:
            print(f"  已回放 {i+1}/{len(df)} 场，耗时 {(pd.Timestamp.now()-t0).total_seconds():.0f}s")

    rdf = pd.DataFrame(records)
    print(f"\n统计区间: {args.start} ~ {rdf['date'].max().date()}，共 {len(rdf)} 场")
    print(f"回放总耗时 {(pd.Timestamp.now()-t0).total_seconds():.0f}s")

    # ---- 汇总 ----
    n = len(rdf)
    hit = (rdf["pred_r"] == rdf["act_r"]).sum()
    mkt_n = rdf["mkt_r"].notna().sum()
    mkt_hit = (rdf["mkt_r"] == rdf["act_r"]).sum()

    lines = []
    lines.append(f"Walk-forward 回测报告（{args.start} ~ {rdf['date'].max().date()}，{n} 场）")
    lines.append("=" * 66)
    lines.append(f"让球胜平负命中: {hit}/{n} = {hit/n:.1%}")
    if mkt_n:
        lines.append(f"市场让球SP基线:  {mkt_hit}/{mkt_n} = {mkt_hit/mkt_n:.1%}")
        lines.append(f"模型-市场差:      {hit/n - mkt_hit/mkt_n:+.1%}")

    def seg(title, mask, label=None):
        sub = rdf[mask]
        if len(sub) == 0:
            lines.append(f"  {title}: 0 场")
            return
        h = (sub["pred_r"] == sub["act_r"]).sum()
        tag = f"（{label}）" if label else ""
        lines.append(f"  {title}{tag}: {len(sub)} 场, 命中 {h/len(sub):.1%}")

    lines.append("\n[按让球方向]")
    seg("主让", rdf["rang"] < 0)
    seg("客让", rdf["rang"] > 0)
    seg("平手", rdf["rang"] == 0)
    lines.append("\n[方向 × 让球数]")
    for rg in (-2, -1, 1, 2, 0):
        sub = rdf[rdf["rang"] == rg]
        if len(sub):
            h = (sub["pred_r"] == sub["act_r"]).sum()
            name = "主让2" if rg == -2 else "主让1" if rg == -1 else \
                   "客让1" if rg == 1 else "客让2" if rg == 2 else "平手"
            lines.append(f"  {name}: {len(sub):>3} 场, 命中 {h/len(sub):.1%}")

    lines.append("\n[按联赛（样本≥20场）]")
    lgb = rdf.groupby("league").agg(n=("pred_r", "size"),
                                    hit=("pred_r", lambda s: (s == rdf.loc[s.index, "act_r"]).sum()))
    lgb = lgb[lgb["n"] >= 20].sort_values("hit", ascending=False)
    for lg, r_ in lgb.iterrows():
        lines.append(f"  {lg}: {int(r_['n']):>3} 场, 命中 {r_['hit']/r_['n']:.1%}")

    lines.append("\n[按置信度段 · 让球主推概率]")
    for lo, hi in ((0.7, 1.01), (0.6, 0.7), (0.55, 0.6), (0.5, 0.55), (0.0, 0.5)):
        sub = rdf[(rdf["conf"] >= lo) & (rdf["conf"] < hi)]
        if len(sub):
            h = (sub["pred_r"] == sub["act_r"]).sum()
            lines.append(f"  [{lo:.2f},{hi:.2f}): {len(sub):>3} 场, 命中 {h/len(sub):.1%}")

    lines.append("\n[按模型-市场概率差（价值差 = 模型主推 - 市场主推）]")
    rdf2 = rdf.copy()
    rdf2["val"] = rdf2["conf"] - rdf2["mkt_conf"]
    for lo, hi in ((0.10, 1.0), (0.05, 0.10), (0.0, 0.05), (-1.0, 0.0)):
        sub = rdf2[(rdf2["val"] >= lo) & (rdf2["val"] < hi)]
        if len(sub):
            h = (sub["pred_r"] == sub["act_r"]).sum()
            lines.append(f"  [{lo:+.2f},{hi:+.2f}): {len(sub):>3} 场, 命中 {h/len(sub):.1%}")

    lines.append("\n[按年份]")
    rdf2["year"] = rdf2["date"].dt.year
    for y, sub in rdf2.groupby("year"):
        h = (sub["pred_r"] == sub["act_r"]).sum()
        lines.append(f"  {y}: {len(sub)} 场, 命中 {h/len(sub):.1%}")

    report = "\n".join(lines)
    print("\n" + report)

    if args.save:
        out = BASE_DIR / "outputs"
        out.mkdir(exist_ok=True)
        p = out / "walkforward_回测报告.txt"
        p.write_text(report, encoding="utf-8")
        rdf.to_csv(out / "walkforward_逐场.csv", index=False, encoding="utf-8-sig")
        print(f"\n→ 已保存: {p} + walkforward_逐场.csv")


if __name__ == "__main__":
    main()
