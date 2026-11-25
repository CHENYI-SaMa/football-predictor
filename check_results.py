# -*- coding: utf-8 -*-
"""
check_results.py —— 预测结果回填核对（闭环）
============================================
读取 outputs/ 下已保存的预测 JSON，从 500 网历史开奖页自动回填真实赛果，
对比预测的「让球胜平负方向」与实际结果，统计真实命中率。

用法：
    python check_results.py                    # 核对最近一次批量预测
    python check_results.py --file outputs/竞彩预测数据_20260807.json
    python check_results.py --days 30          # 核对近 30 天所有预测记录
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from path_helper import get_base_dir, get_outputs_dir  # noqa: E402
BASE_DIR = get_base_dir()
sys.path.insert(0, str(BASE_DIR))

from src.jczq_history import fetch_day


def load_predictions(path: Path | None = None) -> list[dict]:
    """加载预测 JSON（results 列表）。"""
    if path is None:
        cands = sorted(BASE_DIR.glob("outputs/竞彩预测数据_*.json"))
        if not cands:
            raise FileNotFoundError("outputs/ 下没有预测数据文件")
        path = cands[-1]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    # 补充比赛日期：从 match 字段或 match_time 提取
    for r in results:
        if "日期" not in r:
            mt = r.get("match", {})
            if isinstance(mt, dict):
                t = mt.get("match_time", "")
            else:
                t = ""
            mm = re.search(r"(\d{2})-(\d{2})", str(t))
            r["日期"] = (f"2026-{mm.group(1)}-{mm.group(2)}" if mm else "")
        if "编号" not in r:
            mt = r.get("match", {})
            r["编号"] = mt.get("no", "") if isinstance(mt, dict) else ""
    return results


def backfill(predictions: list[dict], verbose: bool = True) -> dict:
    """按比赛日期抓开奖并回填赛果，统计命中率。"""
    # 按日期分组
    by_date: dict[str, list[dict]] = {}
    for p in predictions:
        d = p.get("日期", "")
        if d:
            by_date.setdefault(d, []).append(p)

    n_pred = 0
    n_hit = 0
    per_day = []
    for date, plist in sorted(by_date.items()):
        try:
            # 开奖数据必须实时抓取（本地缓存可能是旧/空数据）
            day_matches = {m["no"]: m for m in fetch_day(date, use_cache=False)}
        except Exception as e:
            if verbose:
                print(f"  [跳过] {date}: {e}")
            continue
        hit = miss = no_result = 0
        for p in plist:
            no = p.get("编号", "")
            m = day_matches.get(no)
            if m is None:
                no_result += 1
                continue
            # 实际让球赛果：主队进球 + 让球数 vs 客队
            rang = m["rang"]
            actual = 0 if m["fthg"] + rang > m["ftag"] else (
                1 if m["fthg"] + rang == m["ftag"] else 2)
            pred = p.get("预测让球胜平负") or p.get("推荐让球胜平负") or ""
            pred_code = {"RS": 0, "RP": 1, "RF": 2}.get(str(pred)[:2])
            if pred_code is None:
                no_result += 1
                continue
            n_pred += 1
            ok = (pred_code == actual)
            n_hit += ok
            if ok:
                hit += 1
            else:
                miss += 1
            p["实际让球结果"] = ["RS 让胜", "RP 让平", "RF 让负"][actual]
            p["是否命中"] = ok
        per_day.append({"日期": date, "预测": hit + miss, "命中": hit,
                        "未中": miss, "无结果": no_result})
        if verbose:
            r = f"{hit/(hit+miss):.0%}" if hit + miss else "-"
            print(f"  {date}: 命中 {hit}/{hit+miss} ({r})")

    total = f"{n_hit/n_pred:.1%}" if n_pred else "-"
    summary = {
        "总预测": n_pred, "总命中": n_hit,
        "总命中率": round(n_hit / n_pred, 4) if n_pred else None,
        "按日": per_day,
    }
    if verbose:
        print(f"\n==== 汇总：命中 {n_hit}/{n_pred} = {total} ====")
    return summary


def main():
    parser = argparse.ArgumentParser(description="预测结果回填核对")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--save", action="store_true",
                        help="把回填结果写回预测 JSON")
    args = parser.parse_args()

    preds = load_predictions(Path(args.file) if args.file else None)
    print(f"加载预测 {len(preds)} 条")
    summary = backfill(preds)

    if args.save:
        path = Path(args.file) if args.file else \
            sorted(BASE_DIR.glob("outputs/竞彩预测数据_*.json"))[-1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "results": preds, "回填核对": summary},
                      f, ensure_ascii=False, indent=2)
        print(f"→ 已写回 {path.name}")
    # 保存核对报告
    out = get_outputs_dir() / "结果核对报告.txt"
    lines = [f"预测结果核对报告（{time.strftime('%Y-%m-%d %H:%M:%S')}）",
             f"总预测 {summary['总预测']} 场，命中 {summary['总命中']} 场，"
             f"命中率 {summary['总命中率']:.1%}" if summary["总命中率"] else "无有效对比",
             "-" * 50]
    for d in summary["按日"]:
        lines.append(f"{d['日期']}: 命中 {d['命中']}/{d['预测']}（无结果 {d['无结果']}）")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"→ 报告 {out.name}")


if __name__ == "__main__":
    main()
