# -*- coding: utf-8 -*-
"""fetch_hist.py —— 下载 football-data 指定年份范围（带重试，抗网络抖动）"""
import sys
import time
from pathlib import Path

import requests

OUT = Path("data/football_history")
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}
LEAGUES = ["E0", "E1", "E2", "E3", "SC0", "D1", "D2", "I1", "SP1", "F1",
           "N1", "B1", "P1", "T1", "G1", "EC", "EL"]


def fetch_one(code: str, season: str) -> bool:
    f = OUT / f"{code}_{season}.csv"
    if f.exists():
        return True
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=H, timeout=20)
            if r.status_code == 200 and len(r.text) > 300:
                f.write_text(r.text, encoding="utf-8", errors="ignore")
                return True
        except Exception:
            pass
        time.sleep(1.5 + attempt)
    return False


start, end = int(sys.argv[1]), int(sys.argv[2])
ok = exist = 0
fails = []
for y in range(start, end + 1):
    season = f"{y:02d}{(y + 1) % 100:02d}"
    for code in LEAGUES:
        if (OUT / f"{code}_{season}.csv").exists():
            exist += 1
            continue
        if fetch_one(code, season):
            ok += 1
        else:
            fails.append(f"{code}_{season}")
        time.sleep(0.2)
# 失败重试一轮
for tag in fails[:]:
    code, season = tag.split("_")
    if fetch_one(code, season):
        fails.remove(tag)
print(f"[{start}-{end}] 新增{ok} 已有{exist} 仍失败{len(fails)}")
if fails:
    print("失败:", ",".join(fails[:20]))
