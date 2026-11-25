# -*- coding: utf-8 -*-
"""error_analysis.py —— 错误类型分析（任务8）
=====================================================
输入：walkforward 逐场回测 CSV（outputs/walkforward_逐场.csv，含
      pred_r/act_r/conf/rang/league/date）或 offline 核对 JSON。
分析错误类型：
  - 爆冷：模型主推让胜/主胜，实际让负/客胜（低估弱队）
  - 高估强队让球：预测让胜，实际让平/让负
  - 方向误判：预测让平，实际让胜/让负（或反之）
  - 高置信失败：模型让球主推概率 ≥0.70 但错（校准漂移信号）
  - 平局误判：预测分胜负，实际平
输出：错误分布 / 场景画像（联赛、让球方向、置信段、价值差）/ 近4周趋势。

用法: python error_analysis.py [--csv outputs/walkforward_逐场.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def classify(row) -> str:
    """返回错误类型（命中返回 None）。"""
    pred, act = row["pred_r"], row["act_r"]
    if pred == act:
        return None
    conf = row.get("conf", 0.0)
    if conf >= 0.70:
        return "高置信失败"
    if pred == "RS" and act == "RF":
        return "低估弱队爆冷"
    if pred == "RS" and act == "RP":
        return "高估强队让球"
    if pred == "RP" and act in ("RS", "RF"):
        return "让球方向误判"
    if pred == "RF" and act == "RS":
        return "低估强队逆转"
    if (pred in ("RS", "RF")) and act == "RP":
        return "漏判平局"
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE_DIR / "outputs" / "walkforward_逐场.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df[df["act_r"].notna()].copy()
    df["err"] = df.apply(classify, axis=1)
    n = len(df)
    n_err = df["err"].notna().sum()
    print(f"总场次 {n}，错误 {n_err}（错误率 {n_err / n:.1%}）\n")

    # ---- 1. 错误类型分布 ----
    print("【错误类型分布】")
    dist = df[df["err"].notna()]["err"].value_counts()
    for k, v in dist.items():
        print(f"  {k:<10}{v:>5} 场  {v / n_err:.1%}")
    print(f"  {'命中':<10}{n - n_err:>5} 场  {(n - n_err) / n:.1%}")

    # ---- 2. 错误画像：按场景 ----
    err = df[df["err"].notna()]
    print("\n【按让球方向】错误率")
    for label, m in (("主让", df["rang"] < 0), ("客让", df["rang"] > 0),
                     ("平手", df["rang"] == 0)):
        sub = df[m]
        if len(sub):
            print(f"  {label}: {len(sub)} 场, 错误率 {(sub['err'].notna()).mean():.1%}")

    print("\n【按置信度段】错误率（校准漂移信号）")
    for lo, hi in ((0.70, 1.01), (0.60, 0.70), (0.50, 0.60), (0.0, 0.50)):
        sub = df[(df["conf"] >= lo) & (df["conf"] < hi)]
        if len(sub):
            print(f"  [{lo:.2f},{hi:.2f}): {len(sub)} 场, 错误率 {(sub['err'].notna()).mean():.1%}")

    print("\n【按联赛】错误率（样本≥50场，top10 最差）")
    lg = df.groupby("league").agg(n=("err", "size"),
                                  err_rate=("err", lambda s: s.notna().mean()))
    for lg_name, r in lg[lg["n"] >= 50].sort_values("err_rate", ascending=False).head(10).iterrows():
        print(f"  {lg_name}: {int(r['n'])} 场, 错误率 {r['err_rate']:.1%}")

    # ---- 3. 近4周趋势 ----
    print("\n【近4周错误率趋势】")
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    last = sorted(df["week"].unique())[-4:]
    for w in last:
        sub = df[df["week"] == w]
        if len(sub):
            print(f"  第{w}周: {len(sub)} 场, 错误率 {(sub['err'].notna()).mean():.1%}, "
                  f"高置信失败 {(sub['err'] == '高置信失败').sum()} 场")

    # ---- 4. 高置信失败明细（重点监控） ----
    hc = df[df["err"] == "高置信失败"]
    if len(hc):
        print(f"\n【高置信失败明细】{len(hc)} 场（模型≥70%但错）")
        for _, r in hc.sort_values("conf", ascending=False).head(8).iterrows():
            print(f"  {r['date'].date()} {r['home']}vs{r['away']} "
                  f"让球{r['rang']} 预测{r['pred_r']}{r['conf']:.0%} 实际{r['act_r']} "
                  f"比分{r['score']}")

    # ---- 存档 ----
    out = BASE_DIR / "outputs" / "error_analysis.txt"
    report = [f"错误分析报告（{n} 场，错误率 {n_err / n:.1%}）"]
    report.append("错误分布: " + " | ".join(f"{k} {v}" for k, v in dist.items()))
    report.append("高置信失败率: " +
                  f"{(df['err'] == '高置信失败').sum()}/{n} "
                  f"{(df['err'] == '高置信失败').mean():.1%}")
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\n→ 已保存 {out}")

    # ---- E1: 结构化 JSON（供 retrain_weekly / 外部消费） ----
    # 收集画像统计
    stats = {
        "n": int(n), "n_err": int(n_err),
        "err_rate": float(n_err / n),
        "dist": {k: int(v) for k, v in dist.items()},
        "by_rang": {},
        "by_conf": {},
        "by_league": {},
        "weekly": {},
        "hc_fail": int((df["err"] == "高置信失败").sum()),
        "hc_fail_rate": float((df["err"] == "高置信失败").mean()),
    }
    for label, m in (("主让", df["rang"] < 0), ("客让", df["rang"] > 0),
                     ("平手", df["rang"] == 0)):
        sub = df[m]
        if len(sub):
            stats["by_rang"][label] = {"n": int(len(sub)),
                                       "err_rate": float(sub["err"].notna().mean())}
    for lo, hi in ((0.70, 1.01), (0.60, 0.70), (0.50, 0.60), (0.0, 0.50)):
        sub = df[(df["conf"] >= lo) & (df["conf"] < hi)]
        if len(sub):
            stats["by_conf"][f"{lo:.2f}-{hi:.2f}"] = {
                "n": int(len(sub)), "err_rate": float(sub["err"].notna().mean())}
    lg = df.groupby("league").agg(n=("err", "size"),
                                  err_rate=("err", lambda s: s.notna().mean()))
    for lg_name, r in lg[lg["n"] >= 50].sort_values("err_rate", ascending=False).head(10).iterrows():
        stats["by_league"][lg_name] = {"n": int(r["n"]), "err_rate": float(r["err_rate"])}
    for w in last:
        sub = df[df["week"] == w]
        if len(sub):
            stats["weekly"][f"第{w}周"] = {
                "n": int(len(sub)), "err_rate": float(sub["err"].notna().mean()),
                "hc_fail": int((sub["err"] == "高置信失败").sum())}

    # ---- E2: 触发规则 → 自动调整建议 ----
    advice = _gen_advice(stats)
    stats["advice"] = advice
    (BASE_DIR / "outputs" / "error_analysis.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 结构化报告已保存 outputs/error_analysis.json（含 {len(advice)} 条调整建议）")
    if advice:
        print("\n【自动调整建议】")
        for a in advice:
            print(f"  [{a['severity']}] {a['msg']}")


def _gen_advice(stats: dict) -> list[dict]:
    """E2：基于错误画像生成自动调整建议（供周重训闭环消费）。"""
    advice: list[dict] = []

    # 规则1：高置信段错误率异常 → 校准漂移
    hc_key = "0.70-1.01"
    if hc_key in stats["by_conf"]:
        hc_err = stats["by_conf"][hc_key]["err_rate"]
        if hc_err >= 0.25:
            advice.append({"type": "calibration", "severity": "high",
                           "msg": f"高置信段(≥70%)错误率 {hc_err:.0%}，模型过度自信，建议温度重校准(T)"})
        elif hc_err >= 0.20:
            advice.append({"type": "calibration", "severity": "medium",
                           "msg": f"高置信段错误率 {hc_err:.0%} 偏高，观察是否需要温度校准"})

    # 规则2：主让方向系统性偏差 → 分桶校准
    if "主让" in stats["by_rang"]:
        hm_err = stats["by_rang"]["主让"]["err_rate"]
        if hm_err >= 0.60:
            advice.append({"type": "calibration", "severity": "high",
                           "msg": f"主让场景错误率 {hm_err:.0%}，建议按让球方向分桶校准"})

    # 规则3：联赛画像异常（错误率 > 全局 +10pp）
    global_err = stats["err_rate"]
    for lg, r in stats["by_league"].items():
        if r["err_rate"] >= global_err + 0.10 and r["n"] >= 50:
            advice.append({"type": "feature", "severity": "medium",
                           "msg": f"联赛[{lg}] 错误率 {r['err_rate']:.0%}（全局 {global_err:.0%}），"
                                  f"检查该联赛特征是否缺失/失真"})

    # 规则4：近4周错误率上升趋势 → 数据/分布漂移
    weeks = list(stats["weekly"].items())
    if len(weeks) >= 2:
        first, last_w = weeks[0][1]["err_rate"], weeks[-1][1]["err_rate"]
        if last_w > first + 0.05:
            advice.append({"type": "retrain", "severity": "medium",
                           "msg": f"近4周错误率上升（{first:.0%}→{last_w:.0%}），"
                                  f"建议重训并检查近期分布漂移"})

    # 规则5：高置信失败占比过高 → 价值投注置信度不可信
    if stats["n"] >= 100 and stats["hc_fail_rate"] >= 0.05:
        advice.append({"type": "strategy", "severity": "low",
                       "msg": f"高置信失败占比 {stats['hc_fail_rate']:.1%}，"
                              f"价值投注的置信过滤阈值可考虑上调"})
    return advice


if __name__ == "__main__":
    main()
