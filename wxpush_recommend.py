# -*- coding: utf-8 -*-
"""wxpush_recommend.py —— 用 WxPusher 把 V3 推荐推送到微信
========================================================
读取 outputs/预测_v3_{date}.json → 格式化为文本 → 推送到配置的微信用户。
用法: python wxpush_recommend.py [--date 2026-08-14] [--top 全部]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
# WxPusher 配置（复用 Delta Force 推送的 app_token 与接收人）
CFG = Path("C:/Users/CHENYI/WorkBuddy/2026-07-23-01-15-09/delta-force-cloud/wxpusher_config.json")


def rng_label(r: int) -> str:
    return ("主让" + str(-r) if r < 0
            else "客让" + str(r) if r > 0 else "平手")


def build_content(results, date: str, only_day: str | None = None) -> str:
    picks = [r for r in results if r["pick"]]
    if only_day:
        picks = [r for r in picks if r["no"].startswith(only_day)]
    # 按比赛日分组（周五/周六/周日），组内按编号排序
    def day_key(x):
        prefix = x["no"][:2]
        return {"周五": 0, "周六": 1, "周日": 2}.get(prefix, 9)
    picks.sort(key=lambda x: (day_key(x), x["no"][2:]))
    lines = [f"{date} 竞彩让球推荐（V3模型 60.2%）",
             f"共{len(picks)}场 | 置信≥60% 价值≥10% | ★=置信≥80%"]
    cur_day = None
    for x in picks:
        day = x["no"][:2]
        if day != cur_day:
            lines.append("")
            lines.append(f"━━━ {day} ━━━")
            cur_day = day
        star = "★" if x["conf"] >= 0.80 else " "
        main_pick = x["pred_range"] + ("/让平" if x.get("rp_second") else "")
        lines.append(
            f"{x['no']} {x['home']}vs{x['away']} [{rng_label(x['rang'])}]")
        lines.append(f"  {main_pick} {x['conf']:.0%}{star}")
        lines.append("")
    return "\n".join(lines).rstrip()


def push(content: str, uids: list[str] | None = None) -> tuple[bool, str]:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    r = requests.post(
        "https://wxpusher.zjiecode.com/api/send/message",
        json={
            "appToken": cfg["app_token"],
            "content": content,
            "contentType": 1,          # 1=text
            "uids": uids or cfg["uids"],
        }, timeout=30)
    d = r.json()
    ok = d.get("code") == 1000
    return ok, d.get("msg", str(d))[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--day", default=None, help="只推某比赛日，如 周五/周六/周日")
    ap.add_argument("--uid", default=None,
                    help="只推给指定 UID，逗号分隔可多个（默认推配置里全部）")
    args = ap.parse_args()

    src = BASE_DIR / "outputs" / f"预测_v3_{args.date.replace('-', '')}.json"
    if not src.exists():
        print(f"未找到 {src}")
        return
    d = json.loads(src.read_text(encoding="utf-8"))
    content = build_content(d["results"], args.date, args.day)
    print(f"内容长度: {len(content)} 字符")
    uids = [u.strip() for u in args.uid.split(",") if u.strip()] if args.uid else None
    ok, msg = push(content, uids)
    print(f"推送{'✅ 成功' if ok else '❌ ' + msg}"
          + (f"（仅 {len(uids)} 个 UID）" if uids else ""))
    if ok:
        tag = f"_{args.day}" if args.day else ""
        (BASE_DIR / "outputs" / f"wxpush{tag}_{args.date}.log").write_text(
            content, encoding="utf-8")


if __name__ == "__main__":
    main()
