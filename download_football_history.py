# -*- coding: utf-8 -*-
"""download_football_history.py —— 下载 football-data.co.uk 全历史
（1993-94 ~ 2025-26，15+ 欧洲联赛：英/德/意/西/法/荷/比/葡/土/希/苏 + 英冠英甲英乙 + 欧冠欧联）"""
import sys
import time
from pathlib import Path

import requests

OUT = Path("data/football_history")
OUT.mkdir(parents=True, exist_ok=True)
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}

# 联赛代码（football-data.co.uk）
LEAGUES = {
    "E0": "英格兰超级", "E1": "英格兰冠军", "E2": "英格兰甲级", "E3": "英格兰乙级",
    "SC0": "苏格兰超级", "D1": "德国甲级", "D2": "德国乙级",
    "I1": "意大利甲级", "SP1": "西班牙甲级", "F1": "法国甲级",
    "N1": "荷兰甲级", "B1": "比利时甲级", "P1": "葡萄牙超级",
    "T1": "土耳其超", "G1": "希腊超", "EC": "欧冠", "EL": "欧联",
}
# 赛季（1993-94 ~ 2025-26）
SEASONS = [f"{y:02d}{(y+1) % 100:02d}" for y in range(1993, 2026)]

total = 0
fails = []
t0 = time.time()
for season in SEASONS:
    for code, name in LEAGUES.items():
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        f = OUT / f"{code}_{season}.csv"
        if f.exists():
            total += 1
            continue
        try:
            r = requests.get(url, headers=H, timeout=25)
            if r.status_code == 200 and len(r.text) > 300:
                f.write_text(r.text, encoding="utf-8", errors="ignore")
                total += 1
            elif r.status_code == 200:
                fails.append(f"{code}_{season} (空)")
        except Exception as e:
            fails.append(f"{code}_{season} ({e})")
        time.sleep(0.25)
print(f"共 {total} 个赛季-联赛文件, 耗时 {time.time()-t0:.0f}s, 失败 {len(fails)}")
print("失败示例:", fails[:10])
