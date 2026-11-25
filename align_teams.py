# -*- coding: utf-8 -*-
"""对齐 understat xG 与 football-data 队名，评估对齐率"""
import json
import glob
import pandas as pd

# 1) understat dates → 比赛记录
records = []
for f in glob.glob("data/understat/*.json"):
    d = json.load(open(f, encoding="utf-8"))
    league_season = f.split("/")[-1].replace(".json", "")
    for dt in d.get("dates", []):
        if not dt.get("isResult") or not dt["xG"]["h"] or not dt["xG"]["a"]:
            continue
        records.append({
            "datetime": dt["datetime"], "league": league_season.split("_")[0],
            "home": dt["h"]["title"], "away": dt["a"]["title"],
            "xG_h": float(dt["xG"]["h"]), "xG_a": float(dt["xG"]["a"]),
            "fh": int(dt["goals"]["h"]), "fa": int(dt["goals"]["a"]),
        })
us = pd.DataFrame(records)
us["date"] = pd.to_datetime(us["datetime"]).dt.date
print(f"understat 比赛: {len(us)} 场")

# 2) football-data 近5年
dfs = []
for f in glob.glob("data/football_history/*_2[1-6]*.csv"):
    dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
fd = pd.concat(dfs, ignore_index=True)
fd["date"] = pd.to_datetime(fd["Date"], dayfirst=True, errors="coerce").dt.date
fd = fd.dropna(subset=["date"])
print(f"football-data 比赛: {len(fd)} 场")

# 3) 队名集合对比
us_teams = set(us["home"]) | set(us["away"])
fd_teams = set(fd["HomeTeam"]) | set(fd["AwayTeam"])
print(f"\nunderstat 队名: {len(us_teams)} | football-data 队名: {len(fd_teams)}")
common = us_teams & fd_teams
print(f"完全同名: {len(common)}")
miss = sorted(us_teams - fd_teams)
print("understat有但fd无(前20):", miss[:20])
miss2 = sorted(fd_teams - us_teams)
print("fd有但understat无(前20):", miss2[:20])
