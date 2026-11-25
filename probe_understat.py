# -*- coding: utf-8 -*-
"""解析 understat API（带重试）"""
import time
import requests

H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
    "Referer": "https://understat.com/league/EPL",
    "Accept": "application/json, text/plain, */*",
    "X-Requested-With": "XMLHttpRequest",
}
d = None
for att in range(4):
    r = requests.get("https://understat.com/getLeagueData/EPL/2025", headers=H, timeout=25)
    print("尝试", att, "status:", r.status_code, "长度:", len(r.content))
    if r.status_code == 200:
        try:
            d = r.json()
            break
        except Exception as e:
            print("  JSON解析失败, 内容前80:", r.text[:80].replace("\n", " "))
    time.sleep(3 + att * 2)
if d is None:
    raise SystemExit("获取失败")
teams = d["teams"]
print("球队数:", len(teams))
t = list(teams.values())[0]
print("球队示例:", t["title"], "history 场数:", len(t["history"]))
h0 = t["history"][0]
print("单场字段:", list(h0.keys()))
print("单场:", {k: h0[k] for k in ("h_a", "xG", "xGA", "npxG", "time", "result") if k in h0})
if d.get("dates"):
    print("dates[0]:", d["dates"][0])
