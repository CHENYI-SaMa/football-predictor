# -*- coding: utf-8 -*-
"""fetch_odds_change.py —— 500 欧赔初盘→终盘变化抓取（任务4）
================================================================
从 500 网欧赔分析页（服务端渲染，GBK）解析每家公司【初盘/终盘】胜平负赔率，
输出每场比赛的平均初盘/平均终盘/变化率（Δ胜/Δ平/Δ负）+ 隐含概率。

流程：
  1. 抓 trade.500.com/jczq/ 当天比赛 → 提取每场 fenxi 链接中的比赛 id
  2. 对每场抓 odds.500.com/fenxi/ouzhi-{id}.shtml（GBK 解码）
  3. 解析：公司行 tr[id][ttl=zy] → 初盘表(tr_bdb) 3赔率 + 终盘行 3赔率
  4. 计算平均初盘/终盘 → 去抽水隐含概率 → 变化量
输出：data/odds_change/{date}.json（每场初终盘）+ 控制台汇总
用法: python fetch_odds_change.py [--date 2026-08-14] [--ids 1460771,1467674]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data" / "odds_change"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0.0.0 Safari/537.36")}


def get(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    r.encoding = "gb2312"
    return r.text


def _extract_kelly(tr_html: str) -> tuple[list, list] | None:
    """从公司 tr 段提取凯利指数：(即时凯利[胜平负], 初盘凯利[胜平负])。

    定位：返还率（xx.xx%）td 之后紧跟的 6 个两位小数 = 凯利 6 值
    （前 3 = 即时凯利，后 3 = 初盘凯利）。
    """
    tds = [re.sub(r"<[^>]+>", "", t).strip()
           for t in re.findall(r"<td[^>]*>(.*?)</td>", tr_html, re.S)]
    tds = [t for t in tds if t]
    for i in range(len(tds) - 1):
        if re.match(r"^\d+\.?\d*%$", tds[i]) and re.match(r"^\d+\.?\d*%$", tds[i + 1]):
            kellys = []
            j = i + 2
            while len(kellys) < 6 and j < len(tds):
                if re.match(r"^\d+\.\d{2}$", tds[j]):
                    kellys.append(float(tds[j]))
                j += 1
            if len(kellys) >= 6:
                return kellys[:3], kellys[3:]
            return None
    return None


def parse_company(tr_html: str) -> dict | None:
    """解析一家公司：初盘/终盘 3 赔率 + 公司名 + 更新时间 + 凯利指数。"""
    m = re.search(r'id="(\d+)" ttl="zy"', tr_html)
    if not m:
        return None
    cid = int(m.group(1))
    name_m = re.search(r'title="([^"]*)"', tr_html)
    name = name_m.group(1) if name_m else f"c{cid}"
    tm = re.search(r'data-time="([^"]*)"', tr_html)
    # 初盘表（tr_bdb td_show_cp 内 3 个 klfc td）与终盘行
    inner = re.search(r'<table[^>]*class="pl_table_data".*?</table>', tr_html, re.S)
    if not inner:
        return None
    tbl = inner.group(0)
    # 初盘：第一个含 klfc 的 tr（tr_bdb）
    open_tr = re.search(r'<tr[^>]*class="[^"]*tr_bdb[^"]*"[^>]*>(.*?)</tr>', tbl, re.S)
    # 终盘：紧随其后不含 tr_bdb 的 tr
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
    kelly = _extract_kelly(tr_html)
    return {"id": cid, "name": name, "time": tm.group(1) if tm else "",
            "open": open_odds, "close": close_odds,
            "kelly_close": kelly[0] if kelly else None,
            "kelly_open": kelly[1] if kelly else None}


def parse_page(html: str) -> list[dict]:
    """解析整页所有公司。"""
    companies = []
    # 按公司 tr 切分（id="\d+" ttl="zy"）
    pattern = re.compile(r'<tr[^>]*id="\d+"[^>]*ttl="zy"[^>]*>.*?</tr>\s*</tr>', re.S)
    # 公司 tr 是嵌套的（内含表格），用栈匹配困难；改用 findall tr 再按 id 切
    parts = re.split(r'(?=<tr[^>]*id="\d+"[^>]*ttl="zy"[^>]*>)', html)
    for part in parts:
        c = parse_company(part)
        if c:
            companies.append(c)
    return companies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--ids", default=None, help="逗号分隔比赛id（跳过当天抓取）")
    ap.add_argument("--sleep", type=float, default=0.8)
    args = ap.parse_args()

    # ---- 1. 获取比赛 id 列表 ----
    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        trade_html = get("https://trade.500.com/jczq/")
        ids = list(dict.fromkeys(
            re.findall(r"fenxi/ouzhi-(\d+)\.shtml", trade_html)))
        print(f"当天比赛 id: {len(ids)} 个")
    if not ids:
        print("未获取到比赛 id")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, mid in enumerate(ids):
        try:
            html = get(f"https://odds.500.com/fenxi/ouzhi-{mid}.shtml")
            companies = parse_page(html)
            if not companies:
                print(f"  [{i+1}/{len(ids)}] {mid}: 无解析数据")
                continue
            # 平均初盘/终盘（所有公司）
            op = np.mean([c["open"] for c in companies], axis=0)
            cl = np.mean([c["close"] for c in companies], axis=0)

            def implied(o):
                inv = 1.0 / np.array(o)
                return inv / inv.sum()

            p_op, p_cl = implied(op), implied(cl)
            delta = cl - op
            # 平均凯利（即时/初盘，取有凯利数据的公司）
            kcs = [c["kelly_close"] for c in companies if c.get("kelly_close")]
            kos = [c["kelly_open"] for c in companies if c.get("kelly_open")]
            k_close = list(np.mean(kcs, axis=0)) if kcs else None
            k_open = list(np.mean(kos, axis=0)) if kos else None
            results.append({
                "match_id": mid, "n_company": len(companies),
                "avg_open": [round(x, 2) for x in op],
                "avg_close": [round(x, 2) for x in cl],
                "implied_open": [round(x, 3) for x in p_op],
                "implied_close": [round(x, 3) for x in p_cl],
                "delta_odds": [round(x, 2) for x in delta],
                "delta_implied": [round(x, 3) for x in (p_cl - p_op)],
                "kelly_open": [round(x, 3) for x in k_open] if k_open else None,
                "kelly_close": [round(x, 3) for x in k_close] if k_close else None,
                "top_companies": [
                    {"name": c["name"], "open": c["open"], "close": c["close"],
                     "kelly_open": c.get("kelly_open"),
                     "kelly_close": c.get("kelly_close"),
                     "time": c["time"]} for c in companies[:6]],
            })
            print(f"  [{i+1}/{len(ids)}] {mid}: {len(companies)}家 "
                  f"初{op[0]:.2f}/{op[1]:.2f}/{op[2]:.2f} → "
                  f"终{cl[0]:.2f}/{cl[1]:.2f}/{cl[2]:.2f}"
                  + (f" 凯利{k_close[0]:.2f}/{k_close[1]:.2f}/{k_close[2]:.2f}"
                     if k_close else ""))
        except Exception as e:
            print(f"  [{i+1}/{len(ids)}] {mid}: 失败 {e}")
        time.sleep(args.sleep)

    out = OUT_DIR / f"{args.date}.json"
    out.write_text(json.dumps({
        "date": args.date, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matches": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 已保存 {out}（{len(results)} 场）")

    # 汇总（赔率变化明显的场次）
    print("\n=== 赔率异动 top（|Δ隐含概率| 最大）===")
    for m in sorted(results, key=lambda x: -max(abs(v) for v in x["delta_implied"]))[:8]:
        d = m["delta_implied"]
        print(f"  {m['match_id']}: 初{m['implied_open']} 终{m['implied_close']} "
              f"Δ主{d[0]:+.0%} Δ平{d[1]:+.0%} Δ客{d[2]:+.0%}")


if __name__ == "__main__":
    main()
