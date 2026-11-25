# -*- coding: utf-8 -*-
"""check_v3_results.py —— V3 预测赛果复盘（txt，严格格式）
================================================================
对照模型【让球选项】逐条标注 ✅/❌，统计命中率 + ★高置信单独统计。
用法: python check_v3_results.py --date 2026-08-14
输出: outputs/赛果复盘V3_{date}.txt（并同步桌面 竞彩预测/{date}/）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.jczq_history import fetch_day          # noqa: E402


def result_r(fthg: float, ftag: float, rang: float) -> str:
    """让球结果：fthg+rang vs ftag → RS/RP/RF。"""
    yh = fthg + rang
    return "RS" if yh > ftag else ("RP" if yh == ftag else "RF")


def spf_label(fthg: float, ftag: float) -> str:
    return "胜" if fthg > ftag else ("平" if fthg == ftag else "负")


R_LABEL = {"RS": "让胜", "RP": "让平", "RF": "让负"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="比赛日 YYYY-MM-DD")
    args = ap.parse_args()
    date = args.date

    # 1. 预测
    pred_path = BASE_DIR / "outputs" / f"预测_v3_{date.replace('-', '')}.json"
    if not pred_path.exists():
        print(f"未找到预测 {pred_path}")
        return
    preds = {r["no"]: r for r in
             json.loads(pred_path.read_text(encoding="utf-8"))["results"]}
    print(f"预测场次: {len(preds)}")

    # 2. 赛果（强制刷新，避免缓存空结果）
    ms = fetch_day(date, use_cache=False)
    if not ms:
        print(f"{date} 无赛果（可能未开赛或未开奖）")
        return
    print(f"赛果场次: {len(ms)}")

    # 3. 逐场比对
    rows = []
    for m in ms:
        no = m.get("no")
        p = preds.get(no)
        if not p:
            continue
        fthg, ftag = int(m["fthg"]), int(m["ftag"])
        rang = int(m.get("rang", 0) or 0)
        act_r = result_r(fthg, ftag, rang)
        # 预测 pred_range 为中文（让胜/让平/让负），转 RS/RP/RF 比对
        pred_r = {"让胜": "RS", "让平": "RP", "让负": "RF"}.get(p["pred_range"], "?")
        hit = pred_r == act_r
        star = p.get("conf", 0) >= 0.80
        rows.append({
            "no": no, "score": f"{fthg}:{ftag}",
            "spf": spf_label(fthg, ftag),
            "act_r": act_r, "pred_r": pred_r,
            "hit": hit, "star": star, "conf": p.get("conf", 0),
            "home": m["home_team"], "away": m["away_team"],
        })

    if not rows:
        print("预测与赛果无交集（可能预测日与比赛日错位）")
        return

    # 按编号升序排列（500 页面赛果顺序不固定，必须重排）
    rows.sort(key=lambda x: x["no"])

    # 4. 生成 txt（严格格式）
    lines = []
    for r in rows:
        star = "★" if r["star"] else ""
        # 行格式：场次号：比分，基础、让球  ✅/❌（✅/❌ 只按模型让球选项判定）
        lines.append(f"{r['no']}：{r['score']}，{r['spf']}、{R_LABEL[r['act_r']]}"
                     f"  {'✅' if r['hit'] else '❌'}{star}")
    lines.append("")
    lines.append("统计：")
    hits = [r for r in rows if r["hit"]]
    blacks = [r for r in rows if not r["hit"]]
    lines.append("✅让球命中：" + "、".join(r["no"] for r in hits)
                 if hits else "✅让球命中：无")
    lines.append("❌黑：" + "、".join(r["no"] for r in blacks)
                 if blacks else "❌黑：无")
    lines.append(f"总命中 {len(hits)}/{len(rows)}，命中率 {len(hits)/len(rows):.0%}")
    lines.append("")
    stars = [r for r in rows if r["star"]]
    if stars:
        s_hit = [r for r in stars if r["hit"]]
        s_blk = [r for r in stars if not r["hit"]]
        lines.append(f"★共{len(stars)}中{len(s_hit)}：")
        lines.append("  命中：" + ("、".join(r["no"] for r in s_hit) if s_hit else "无"))
        lines.append("  黑：" + ("、".join(r["no"] for r in s_blk) if s_blk else "无"))
    lines.append("")
    # 现象总结（1-2 句，从命中/黑单中找规律）
    if hits and blacks:
        lines.append("现象总结：" + ("命中集中；" if len(hits) >= len(blacks) else "黑单集中；")
                     + (f"让负类命中{sum(1 for r in hits if r['pred_r']=='RF')}场，"
                        f"让胜类命中{sum(1 for r in hits if r['pred_r']=='RS')}场。"))
    elif not blacks:
        lines.append("现象总结：全部命中。")
    else:
        lines.append("现象总结：全部黑单。")
    lines.append("温馨提示：彩票有风险，理性看待。")
    txt = "\n".join(lines)

    out = BASE_DIR / "outputs" / f"赛果复盘V3_{date.replace('-', '')}.txt"
    out.write_text(txt, encoding="utf-8")
    print(f"\n→ 已保存 {out}")
    print(txt)

    # 同步桌面
    try:
        desk = Path("C:/Users/CHENYI/Desktop/竞彩预测") / date.replace("-", "")
        desk.mkdir(parents=True, exist_ok=True)
        (desk / f"赛果复盘V3_{date.replace('-', '')}.txt").write_text(
            txt, encoding="utf-8")
        print(f"→ 已同步桌面 {desk}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
