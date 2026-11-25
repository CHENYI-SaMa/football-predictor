# -*- coding: utf-8 -*-
"""阶段A-2: 数据融合与清洗（双源命名归一化版）
以 jczq_history.csv 为基础，左连接 football_history（半场/射门/欧赔）与
understat（xG），统一英文名后按 (date, 主队, 客队) 连接，
无法连接用联赛均值填充 + 缺失指示列，然后去重。
输出 data/master_table.csv
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# ---------- 0. 队名映射（中文 → 标准英文名） ----------
mapping = pd.read_csv("team_mapping.csv", encoding="utf-8-sig")
CN2EN = dict(zip(mapping["中文名"], mapping["英文名"]))

# ---------- 0.1 双源命名归一化：源内名 → 标准英文名 ----------
FD_NORM = {
    "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atletico Madrid",
    "Celta": "Celta Vigo", "FC Koln": "Cologne", "Ein Frankfurt": "Eintracht Frankfurt",
    "Espanol": "Espanyol", "Hertha": "Hertha Berlin",
    "M'Gladbach": "Monchengladbach", "M'gladbach": "Monchengladbach",
    "Nott'm Forest": "Nottingham Forest", "Paris SG": "PSG",
    "Vallecano": "Rayo Vallecano", "Sociedad": "Real Sociedad",
    "St Etienne": "Saint-Etienne", "St Pauli": "St. Pauli",
}
US_NORM = {
    "AC Milan": "Milan", "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund", "Borussia M.Gladbach": "Monchengladbach",
    "FC Cologne": "Cologne", "FC Heidenheim": "Heidenheim",
    "Hamburger SV": "Hamburg", "Mainz 05": "Mainz",
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Paris Saint Germain": "PSG",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Real Betis": "Betis", "St. Etienne": "Saint-Etienne",
    "St. Pauli": "St. Pauli", "Hertha BSC": "Hertha Berlin",
}

# ---------- 1. 基础表 jczq ----------
raw = pd.read_csv("data/jczq_history.csv")
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw = raw.dropna(subset=["date", "fthg", "ftag"]).reset_index(drop=True)
raw["home_en"] = raw["home_team"].map(CN2EN).fillna(raw["home_team"])
raw["away_en"] = raw["away_team"].map(CN2EN).fillna(raw["away_team"])
# 丢弃 jczq 自带的空过程列（半场/射门/大小球），由侧表补齐
DROP_EMPTY = ["htg", "htag", "hs", "as", "hst", "ast", "ou_over", "ou_under"]
raw = raw.drop(columns=[c for c in DROP_EMPTY if c in raw.columns])
print(f"基础表: {len(raw)} 场, 时间 {raw['date'].min().date()} ~ {raw['date'].max().date()}")

# ---------- 2. football_history 侧表（归一化英文名） ----------
fd_rows = []
for f in glob.glob("data/football_history/*.csv"):
    try:
        x = pd.read_csv(f)
        if "Date" not in x.columns or "HomeTeam" not in x.columns:
            continue
        x["date"] = pd.to_datetime(x["Date"], dayfirst=True, errors="coerce").dt.normalize()
        cols = {"date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HTHG", "HTAG",
                "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A",
                "B365>2.5", "B365<2.5"}
        keep = [c for c in cols if c in x.columns]
        fd_rows.append(x[keep].copy())
    except Exception as e:
        print("跳过", f, e)
fd = pd.concat(fd_rows, ignore_index=True)
fd = fd.dropna(subset=["date", "HomeTeam", "AwayTeam"])
fd["HomeTeam"] = fd["HomeTeam"].map(lambda s: FD_NORM.get(s, s))
fd["AwayTeam"] = fd["AwayTeam"].map(lambda s: FD_NORM.get(s, s))
fd = fd.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"], keep="first")
fd = fd.rename(columns={"FTHG": "fd_fthg", "FTAG": "fd_ftag", "HTHG": "htg",
                        "HTAG": "htag", "HS": "hs", "AS": "as", "HST": "hst",
                        "AST": "ast", "B365H": "fd_b365h", "B365D": "fd_b365d",
                        "B365A": "fd_b365a", "B365>2.5": "ou_over",
                        "B365<2.5": "ou_under"})
print(f"football_history 侧表: {len(fd)} 场")

# ---------- 3. understat xG 侧表（归一化英文名） ----------
us_rows = []
for f in glob.glob("data/understat/*.json"):
    try:
        j = json.load(open(f, encoding="utf-8"))
        for dt in j.get("dates", []):
            if not dt.get("isResult", True):
                continue
            try:
                g, xg = dt["goals"], dt["xG"]
                us_rows.append({
                    "date": pd.to_datetime(dt["datetime"]).normalize(),
                    "HomeTeam": US_NORM.get(dt["h"]["title"], dt["h"]["title"]),
                    "AwayTeam": US_NORM.get(dt["a"]["title"], dt["a"]["title"]),
                    "xG_h": float(xg["h"]) if xg.get("h") else np.nan,
                    "xG_a": float(xg["a"]) if xg.get("a") else np.nan,
                })
            except Exception:
                pass
    except Exception as e:
        print("跳过", f, e)
us = pd.DataFrame(us_rows).drop_duplicates(
    subset=["date", "HomeTeam", "AwayTeam"], keep="first")
print(f"understat 侧表: {len(us)} 场")

# ---------- 4. 左连接（统一键：date + home_en + away_en） ----------
merged = raw.merge(
    fd, how="left",
    left_on=["date", "home_en", "away_en"],
    right_on=["date", "HomeTeam", "AwayTeam"])
merged = merged.drop(columns=["HomeTeam", "AwayTeam"]).merge(
    us, how="left",
    left_on=["date", "home_en", "away_en"],
    right_on=["date", "HomeTeam", "AwayTeam"])
merged = merged.drop(columns=["HomeTeam", "AwayTeam"])

# ---------- 5. 联赛均值填充 + 缺失指示 ----------
PROCESS_COLS = ["htg", "htag", "hs", "as", "hst", "ast",
                "ou_over", "ou_under", "xG_h", "xG_a"]
for c in PROCESS_COLS:
    if c not in merged.columns:
        merged[c] = np.nan
    merged[c + "_miss"] = merged[c].isna().astype(int)
league_means = merged.groupby("league")[PROCESS_COLS].transform("mean")
merged[PROCESS_COLS] = merged[PROCESS_COLS].fillna(league_means)
for c in PROCESS_COLS:
    merged[c] = merged[c].fillna(merged[c].mean())

# ---------- 6. 去重 ----------
before = len(merged)
merged = merged.drop_duplicates(subset=["date", "home_team", "away_team", "league"])
print(f"去重: {before} -> {len(merged)}")

# ---------- 7. 保存 ----------
merged.to_csv("data/master_table.csv", index=False, encoding="utf-8-sig")
print(f"\n主训练表: {merged.shape}")

# 覆盖率
b5 = merged[merged["league"].isin({"英超", "西甲", "德甲", "意甲", "法甲"})]
print("\n五大联赛覆盖率:")
for c in PROCESS_COLS:
    print(f"  {c}: {b5[c].notna().mean():.0%}", end="  ")
print()
# 匹配率验证
print(f"\n半场比分匹配率(五大): {b5['htg'].notna().mean():.0%}")
print(f"xG匹配率(五大): {b5['xG_h'].notna().mean():.0%}")
