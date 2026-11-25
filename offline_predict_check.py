# -*- coding: utf-8 -*-
"""离线预测 + 赛果核对工具（针对历史日期）
=============================================
用法:
  python offline_predict_check.py                    # 默认核对昨天
  python offline_predict_check.py --date 2026-08-12  # 指定日期
  python offline_predict_check.py --date 2026-08-12 --save

流程:
  1. 抓取 500 网指定日期历史开奖（含让球SP/赛果）
  2. 用 models_jczq（500 专属模型）离线预测每场（让球胜平负 + 胜平负）
  3. 与真实赛果对比，输出命中统计 + 置信度分层分析
  4. --save 时写入 outputs/赛果核对_YYYYMMDD.txt

注意:
  - 不依赖当天赛程接口（历史日期已开赛，500 网只保留开奖页数据）
  - 赔率特征用开奖页的『让球后 SP』，与 500 模型训练一致
  - 球队战绩特征走 FeatureBuilder 增量状态（训练时保存），非五大联赛队
    部分特征缺失时自动用 -1 + missing 指示，置信度会标注 low
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

from src.features import FeatureBuilder          # noqa: E402
from src.jczq_fetcher import to_model_vector     # noqa: E402
from src.jczq_history import fetch_day           # noqa: E402
from src.model import PoissonLightGBM            # noqa: E402
from src.poisson_utils import derive_probs       # noqa: E402

MAX_GOALS = 10


def implied(oh, od, oa):
    """赔率去抽水 → 概率三元组（任一缺失返回 None）。"""
    try:
        arr = np.array([oh, od, oa], dtype=float)
    except (TypeError, ValueError):
        return None
    if np.any(~np.isfinite(arr)) or np.any(arr <= 1):
        return None
    inv = 1.0 / arr
    return tuple(inv / inv.sum())


def load_jczq_model():
    """加载 500 专属模型 + 特征状态。"""
    model = PoissonLightGBM.load(BASE_DIR / "models_jczq" / "model.pkl")
    meta = json.loads((BASE_DIR / "models_jczq" / "train_meta.json").read_text(encoding="utf-8"))
    with open(BASE_DIR / "models_jczq" / "feature_state.pkl", "rb") as f:
        state = pickle.load(f)
    return model, meta, state


def predict_one(model, meta, state, m: dict) -> dict:
    """单场预测（500 模型路径，与 predict_jczq.predict_jczq 路径2 一致）。"""
    rang = int(m.get("rang") or 0)
    odds = (m.get("rsp_h"), m.get("rsp_d"), m.get("rsp_a"))
    if not all(x is not None for x in odds):
        odds = (np.nan, np.nan, np.nan)
    row = pd.Series({
        "date": pd.Timestamp(m["date"]),
        "league": m["league"],
        "home_team": m["home_team"], "away_team": m["away_team"],
        "fthg": -1, "ftag": -1, "ftr": "X",
        "htg": np.nan, "htag": np.nan,
        "hs": np.nan, "as": np.nan, "hst": np.nan, "ast": np.nan,
        "b365h": odds[0], "b365d": odds[1], "b365a": odds[2],
        "ou_over": np.nan, "ou_under": np.nan,
        "ahh": float(rang),
    })
    fdict = FeatureBuilder.build_single(state, pd.Series(row))
    X = to_model_vector(fdict, meta["feature_cols"]).reshape(1, -1)
    lam_h, lam_a = model.predict_lambdas(X)
    grid = model.predict_grids(X)[0]
    probs = derive_probs(grid)

    # 让球胜平负：比分网格按让球数平移
    n = grid.shape[0]
    ih, ja = np.indices((n, n))
    shifted = ih + rang
    p_rs = float(grid[shifted > ja].sum())
    p_rp = float(grid[shifted == ja].sum())
    p_rf = float(grid[shifted < ja].sum())
    return {
        "entry": m,
        "rang": rang,
        "p_h": probs["home_win_prob"], "p_d": probs["draw_prob"], "p_a": probs["away_win_prob"],
        "p_rs": p_rs, "p_rp": p_rp, "p_rf": p_rf,
        "lambda_h": float(lam_h[0]), "lambda_a": float(lam_a[0]),
        "most_likely": probs["most_likely_score"],
    }


def check(matches: list[dict], model, meta, state) -> list[dict]:
    rows = []
    for m in matches:
        r = predict_one(model, meta, state, m)
        fthg, ftag = int(m["fthg"]), int(m["ftag"])
        act_spf = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
        yh_r = fthg + r["rang"]
        act_r = "RS" if yh_r > ftag else ("RP" if yh_r == ftag else "RF")

        pred_spf = ["H", "D", "A"][int(np.argmax([r["p_h"], r["p_d"], r["p_a"]]))]
        pred_r = ["RS", "RP", "RF"][int(np.argmax([r["p_rs"], r["p_rp"], r["p_rf"]]))]

        # 市场基线：让球 SP 去抽水 argmax（评估模型是否优于市场）
        mkt = implied(m.get("rsp_h"), m.get("rsp_d"), m.get("rsp_a"))
        mkt_r = None
        if mkt:
            mkt_r = ["RS", "RP", "RF"][int(np.argmax(mkt))]

        rows.append({
            "no": m.get("no", "-"), "league": m["league"],
            "home": m["home_team"], "away": m["away_team"],
            "rang": r["rang"], "score": f"{fthg}:{ftag}",
            "act_r": act_r, "act_spf": act_spf,
            "pred_r": pred_r, "pred_spf": pred_spf,
            "p_rs": r["p_rs"], "p_rp": r["p_rp"], "p_rf": r["p_rf"],
            "p_h": r["p_h"], "p_d": r["p_d"], "p_a": r["p_a"],
            "conf_r": max(r["p_rs"], r["p_rp"], r["p_rf"]),
            "mkt_r": mkt_r,
            "lambda": (r["lambda_h"], r["lambda_a"]),
            "ml": r["most_likely"],
        })
    return rows


def fmt_prob(x):
    return f"{x:.0%}" if x is not None else "-"


def build_report(rows: list[dict], date: str, dates: list[str] | None = None) -> str:
    n = len(rows)
    hit_r = sum(1 for r in rows if r["pred_r"] == r["act_r"])
    hit_spf = sum(1 for r in rows if r["pred_spf"] == r["act_spf"])
    mkt_n = sum(1 for r in rows if r["mkt_r"] is not None)
    mkt_hit = sum(1 for r in rows if r["mkt_r"] == r["act_r"])

    lines = []
    tag = f" · 区间 {dates[0]} ~ {dates[-1]}" if dates and len(dates) > 1 else f" · {date}"
    lines.append(f"赛果核对报告{tag}（共 {n} 场）")
    lines.append("=" * 62)
    lines.append(f"让球胜平负命中: {hit_r}/{n} = {hit_r / n:.1%}" if n else "无数据")
    if mkt_n:
        lines.append(f"让球SP市场基线:  {mkt_hit}/{mkt_n} = {mkt_hit / mkt_n:.1%}")
        lines.append(f"模型 - 市场基线: {hit_r / n - mkt_hit / mkt_n:+.1%}")
    lines.append(f"普通胜平负命中: {hit_spf}/{n} = {hit_spf / n:.1%}")

    # 置信度分层（模型让球主推概率）
    lines.append("\n[置信度分层 · 让球主推概率]")
    for th in (0.7, 0.6, 0.5):
        sub = [r for r in rows if r["conf_r"] >= th]
        if sub:
            h = sum(1 for r in sub if r["pred_r"] == r["act_r"])
            lines.append(f"  ≥{th:.0%}: {len(sub):>2} 场, 命中 {h} = {h / len(sub):.1%}")
        else:
            lines.append(f"  ≥{th:.0%}: 0 场")

    # 按让球方向分层
    lines.append("\n[按让球方向]")
    for label, cond in (("主让", lambda r: r["rang"] < 0),
                        ("客让", lambda r: r["rang"] > 0),
                        ("平手", lambda r: r["rang"] == 0)):
        sub = [r for r in rows if cond(r)]
        if sub:
            h = sum(1 for r in sub if r["pred_r"] == r["act_r"])
            lines.append(f"  {label}: {len(sub)} 场, 命中 {h} = {h / len(sub):.1%}")

    lines.append("\n[逐场明细]")
    lines.append(f"{'编号':<8}{'赛事':<6}{'主队vs客队':<22}{'让球':<6}{'比分':<8}"
                 f"{'模型(胜/平/负)':<16}{'预测':<4}{'实际':<4}{'市场'}")
    lines.append("-" * 62)
    for r in rows:
        team = f"{r['home']}vs{r['away']}"[:21]
        rng = ("主让" + str(-r["rang"]) if r["rang"] < 0
               else "客让" + str(r["rang"]) if r["rang"] > 0 else "平手")
        pr = f"{r['p_rs']:.2f}/{r['p_rp']:.2f}/{r['p_rf']:.2f}"
        mark = "✓" if r["pred_r"] == r["act_r"] else "✗"
        mkt = r["mkt_r"] or "-"
        lines.append(f"{r['no']:<8}{r['league']:<6}{team:<22}{rng:<6}{r['score']:<8}"
                     f"{pr:<16}{mark}{r['pred_r']:<3}{r['act_r']:<3}{mkt}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="比赛日期 YYYY-MM-DD，默认昨天")
    ap.add_argument("--days", type=int, default=1, help="核对最近 N 天（从昨天往前），默认 1")
    ap.add_argument("--save", action="store_true", help="写入 outputs/赛果核对_日期.txt")
    args = ap.parse_args()

    if args.date:
        dates = [args.date]
    else:
        today = pd.Timestamp.now().normalize()
        dates = [(today - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                 for i in range(1, args.days + 1)]
        dates = dates[::-1]  # 时间正序
    print(f"抓取 {dates} 竞彩历史开奖...")
    all_rows = []
    for date in dates:
        matches = fetch_day(date)
        if not matches:
            print(f"  {date}: 无比赛")
            continue
        print(f"  {date}: {len(matches)} 场")
        model, meta, state = load_jczq_model()
        for r in check(matches, model, meta, state):
            r["date"] = date
            all_rows.append(r)
    if not all_rows:
        print("未抓到任何比赛（可能无开赛或网络异常）")
        return

    report = build_report(all_rows, dates[-1], dates)
    print("\n" + report)

    if args.save:
        out = BASE_DIR / "outputs"
        out.mkdir(exist_ok=True)
        tag = args.date.replace("-", "") if args.date else f"{dates[0].replace('-', '')}_{dates[-1].replace('-', '')}"
        p = out / f"赛果核对_{tag}.txt"
        p.write_text(report, encoding="utf-8")
        print(f"\n→ 已保存: {p}")
        with open(out / f"赛果核对数据_{tag}.json", "w", encoding="utf-8") as f:
            json.dump({"dates": dates, "results": all_rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
