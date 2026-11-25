# -*- coding: utf-8 -*-
"""fetch_understat.py —— 下载 understat xG 数据（每联赛每赛季一次请求）
用法: python fetch_understat.py <联赛> <起始赛季> <结束赛季>
联赛: EPL | La liga | Bundesliga | Serie A | Ligue 1
"""
import json
import sys
import time
from pathlib import Path

import requests

OUT = Path("data/understat")
OUT.mkdir(parents=True, exist_ok=True)
H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "Referer": "https://understat.com/league/EPL",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
}

league = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
for season in range(end, start - 1, -1):
    f = OUT / f"{league.replace(' ', '_')}_{season}.json"
    if f.exists():
        continue
    url = f"https://understat.com/getLeagueData/{league}/{season}"
    for att in range(3):
        try:
            r = requests.get(url, headers=H, timeout=25)
            if r.status_code == 200 and len(r.content) > 10000:
                f.write_bytes(r.content)
                print(f"OK {league}/{season} {len(r.content)//1024}KB", flush=True)
                break
        except Exception:
            pass
        time.sleep(2 + att)
    else:
        print(f"FAIL {league}/{season}", flush=True)
    time.sleep(1.5)
