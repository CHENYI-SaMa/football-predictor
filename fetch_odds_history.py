# -*- coding: utf-8 -*-
"""fetch_odds_history.py —— 历史欧赔初终盘回溯（任务11 C1+C2 数据部分）
======================================================================
对历史 N 天：
  1. 抓 trade.500.com/jczq/?date=YYYY-MM-DD → 提取 ouzhi-{id}（历史比赛id）
  2. 对每场抓 odds.500.com/fenxi/ouzhi-{id}.shtml → 初盘/终盘/隐含概率变化
  3. 与 jczq_history.csv 按（日期+中文队名）关联，输出可训练数据
输出: data/odds_change/history_{date}.json（每日）+ history_all.csv（合并）
用法: python fetch_odds_history.py --days 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "data" / "odds_change"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0.0.0 Safari/537.36"),
           "Referer": "https://trade.500.com/jczq/",
           "Accept-Language": "zh-CN,zh;q=0.9"}


def get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = "gb2312"
    return r.text


def get_day_ids(date: str) -> list[dict]:
    """抓某天竞彩赛程页 → [id]（校验：返回45=疑似失效则重试）。"""
    for attempt in range(3):
        html = get(f"https://trade.500.com/jczq/?date={date}")
        ids = list(dict.fromkeys(re.findall(r"fenxi/ouzhi-(\d+)", html)))
        # 45 且日期较旧 → 疑似被限流返回默认页，重试
        if len(ids) == 45 and attempt < 2:
            time.sleep(3 + attempt * 3)
            continue
        return [{"id": i} for i in ids]
    return []


def parse_company(tr_html: str) -> dict | None:
    m = re.search(r'id="(\d+)" ttl="zy"', tr_html)
    if not m:
        return None
    cid = int(m.group(1))
    name_m = re.search(r'title="([^"]*)"', tr_html)
    name = name_m.group(1) if name_m else f"c{cid}"
    tm = re.search(r'data-time="([^"]*)"', tr_html)
    inner = re.search(r'<table[^>]*class="pl_table_data".*?</table>', tr_html, re.S)
    if not inner:
        return None
    tbl = inner.group(0)
    open_tr = re.search(r'<tr[^>]*class="[^"]*tr_bdb[^"]*"[^>]*>(.*?)</tr>', tbl, re.S)
    close_tr = None
    if open_tr:
        after = tbl[open_tr.end():]
        cm = re.search(r'<tr[^>]*>(.*?)</tr>', after, re.S)
        if cm:
            close_tr = cm

    def odds(tr):
        if not tr:
            return None
        nums = [float(x) for x in re.findall(r'>\s*(\d+\.\d+)\s*</td>', tr)]
        return nums[:3] if len(nums) >= 3 else None

    open_odds = odds(open_tr.group(1) if open_tr else None)
    close_odds = odds(close_tr.group(1) if close_tr else None)
    if not open_odds or not close_odds:
        return None
    return {"id": cid, "name": name, "time": tm.group(1) if tm else "",
            "open": open_odds, "close": close_odds}


def parse_page(html: str) -> list[dict]:
    companies = []
    parts = re.split(r'(?=<tr[^>]*id="\d+"[^>]*ttl="zy"[^>]*>)', html)
    for part in parts:
        c = parse_company(part)
        if c:
            companies.append(c)
    return companies


def fetch_match_odds(mid: str) -> dict | None:
    try:
        html = get(f"https://odds.500.com/fenxi/ouzhi-{mid}.shtml")
    except Exception:
        return None
    companies = parse_page(html)
    if not companies:
        return None
    op = np.mean([c["open"] for c in companies], axis=0)
    cl = np.mean([c["close"] for c in companies], axis=0)
    # 标题队名（关联用）
    t = re.search(r"<title>([^<]*)</title>", html)
    title = t.group(1) if t else ""
    teams = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]+)VS([\u4e00-\u9fa5A-Za-z0-9]+)", title)
    inv = lambda o: (1.0 / np.array(o))
    p_op = inv(op); p_op /= p_op.sum()
    p_cl = inv(cl); p_cl /= p_cl.sum()
    return {
        "match_id": mid, "n_company": len(companies),
        "title": title[:60],
        "home": teams.group(1) if teams else "",
        "away": teams.group(2) if teams else "",
        "avg_open": [round(x, 2) for x in op],
        "avg_close": [round(x, 2) for x in cl],
        "implied_open": [round(x, 3) for x in p_op],
        "implied_close": [round(x, 3) for x in p_cl],
        "delta_implied": [round(x, 3) for x in (p_cl - p_op)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    today = pd.Timestamp.now().normalize()
    dates = [(today - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, args.days + 1)]
    dates = dates[::-1]

    all_rows = []
    for date in dates:
        try:
            day_ids = get_day_ids(date)
        except Exception as e:
            print(f"  {date}: 赛程页失败 {e}")
            continue
        if not day_ids:
            print(f"  {date}: 无比赛")
            continue
        rows = []
        for m in day_ids:
            o = fetch_match_odds(m["id"])
            if o:
                o["date"] = date
                rows.append(o)
            time.sleep(args.sleep)
        all_rows.extend(rows)
        if rows:
            p = OUT_DIR / f"history_{date}.json"
            p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {date}: {len(rows)} 场（{len(day_ids)} id）")

    if not all_rows:
        print("无数据")
        return
    df = pd.DataFrame(all_rows)
    out = OUT_DIR / "history_all.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n总计 {len(df)} 场 → {out}")
    print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")


if __name__ == "__main__":
    main()
