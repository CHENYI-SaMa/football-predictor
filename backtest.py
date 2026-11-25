# -*- coding: utf-8 -*-
"""
backtest.py —— 历史回测：模型 vs 真实赛果
=========================================
用已训练 500 模型对「验证集+测试集」（最近约3个月，模型未见过的场次）逐场预测
让球胜平负，对照实际赛果，输出：
  - 总体命中率 / Log Loss（对比让球SP基线）
  - 按模型置信度分层命中率（高置信是否更准）
  - 预测让平的场次统计
  - 逐场对照表 + 结果存档
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.evaluation import odds_baseline_probs
from src.features import build_features
from src.model import PoissonLightGBM

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    raw = pd.read_csv("data/jczq_history.csv", parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    feat, state = build_features(raw)
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)
    print(f"[特征] {feat.shape[0]} 行, 耗时 {time.time()-t0:.1f}s", flush=True)

    max_date = feat["date"].max()
    ts = max_date - pd.Timedelta(days=30)
    vs = max_date - pd.Timedelta(days=90)
    bt = feat[(feat["date"] > vs)].copy()  # 验证集 + 测试集（模型未见过的最近3个月）
    cols = [c for c in feat.columns if c not in
            {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}]

    model = PoissonLightGBM.load("models_jczq/model.pkl")
    X = bt[cols].values
    grids = model.predict_grids(X)
    yh, ya = bt["fthg"].values.astype(float), bt["ftag"].values.astype(float)
    rang = bt["rang"].values
    yh_r = yh + rang
    yoh = np.zeros((len(yh), 3))
    yoh[yh_r > ya, 0] = 1
    yoh[yh_r == ya, 1] = 1
    yoh[yh_r < ya, 2] = 1
    ih, ja = np.indices((grids.shape[1], grids.shape[2]))
    sh = ih[None, :, :] + rang[:, None, None]
    pm = np.stack([np.sum(grids * (sh > ja), axis=(1, 2)),
                   np.sum(grids * (sh == ja), axis=(1, 2)),
                   np.sum(grids * (sh < ja), axis=(1, 2))], axis=1)
    p_mkt = odds_baseline_probs(bt["odds_h"].values, bt["odds_d"].values,
                                bt["odds_a"].values)

    acc = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
    ll = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
    acc_m = float(np.mean(np.argmax(p_mkt, 1) == np.argmax(yoh, 1)))
    ll_m = float(-np.mean(np.sum(yoh * np.log(np.clip(p_mkt, 1e-12, 1)), axis=1)))
    print(f"\n===== 历史回测：{len(bt)} 场（{bt['date'].min().date()} ~ {bt['date'].max().date()}）=====")
    print(f"模型   让球Acc={acc:.1%}  LL={ll:.4f}")
    print(f"市场基线 让球Acc={acc_m:.1%}  LL={ll_m:.4f}")
    print(f"模型领先 Acc +{acc-acc_m:.1%}pp")

    # 按模型置信度分层
    p_win = pm[np.arange(len(pm)), np.argmax(yoh, 1)]
    conf = pm.max(axis=1)
    pred_pick = np.argmax(pm, 1)
    print(f"\n---- 按模型主推置信度分层 ----")
    print(f"{'置信区间':<12}{'场次':<7}{'命中率':<10}{'其中让平':<10}")
    for lo, hi in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        hit = np.mean(pred_pick[m] == yoh[m].argmax(1))
        n_draw_pred = np.sum(pred_pick[m] == 1)
        print(f"{lo:.0%}-{hi:.0%}     {m.sum():<7}{hit:<10.1%}{n_draw_pred:<10}")

    # 预测让平 vs 实际让平
    n_pred_draw = int(np.sum(pred_pick == 1))
    n_actual_draw = int(np.sum(yoh[:, 1] == 1))
    print(f"\n---- 让平 ----")
    print(f"模型主推让平: {n_pred_draw} 场（其中命中 {np.sum((pred_pick==1)&(yoh[:,1]==1))}）")
    print(f"实际让平: {n_actual_draw} 场 ({n_actual_draw/len(bt):.1%})")

    # 保存逐场对照
    rows = []
    for i in range(len(bt)):
        rows.append({
            "日期": bt["date"].iloc[i].date(), "赛事": bt["league"].iloc[i],
            "主队": bt["home_team"].iloc[i], "客队": bt["away_team"].iloc[i],
            "让球": int(rang[i]),
            "真实比分": f"{int(yh[i])}:{int(ya[i])}",
            "模型主推": ["RS 让胜", "RP 让平", "RF 让负"][pred_pick[i]],
            "模型概率": round(float(pm[i][pred_pick[i]]), 3),
            "实际": ["RS 让胜", "RP 让平", "RF 让负"][yoh[i].argmax()],
            "命中": bool(pred_pick[i] == yoh[i].argmax()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "回测对照表.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ 逐场对照表已保存: outputs/回测对照表.csv")
    print("\n---- 最近 15 场对照 ----")
    print(f"{'日期':<12}{'主队':<10}{'客队':<10}{'让':<4}{'模型主推':<9}{'实际比分':<8}{'实际':<9}{'结果'}")
    for r in df.tail(15).itertuples():
        print(f"{str(r.日期):<12}{r.主队:<10}{r.客队:<10}{r.让球:<4}{r.模型主推:<9}"
              f"{r.真实比分:<8}{r.实际:<9}{'✅' if r.命中 else '❌'}")

    # 命中率明细存档
    report = (f"回测范围: {bt['date'].min().date()} ~ {bt['date'].max().date()} ({len(bt)} 场)\n"
              f"模型让球Acc={acc:.1%} LL={ll:.4f} | 市场基线Acc={acc_m:.1%} LL={ll_m:.4f}\n")
    (OUT / "回测报告.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
