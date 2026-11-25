# -*- coding: utf-8 -*-
"""train_xg_ablation.py —— xG 特征消融：Understat xG 是否提升胜平负预测"""
import glob
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from src.features import build_features
from src.model import PoissonLightGBM
from src.evaluation import evaluate_all

# football-data 队名 → understat 队名
NAME_MAP = {
    "Man United": "Manchester United", "Man City": "Manchester City",
    "Newcastle": "Newcastle United", "Nott'm Forest": "Nottm Forest",
    "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atletico Madrid",
    "Betis": "Real Betis", "Celta": "Celta Vigo", "Espanol": "Espanyol",
    "Dortmund": "Borussia Dortmund", "Leverkusen": "Bayer Leverkusen",
    "FC Koln": "FC Cologne", "M'gladbach": "Borussia M.Gladbach",
    "Ein Frankfurt": "Eintracht Frankfurt", "Bielefeld": "Arminia Bielefeld",
    "Heidenheim": "FC Heidenheim", "Mainz": "Mainz 05", "Milan": "AC Milan",
    "Hertha": "Hertha Berlin", "Hamburg": "Hamburger SV",
    "Clermont": "Clermont Foot", "Greuther Furth": "Greuther Fürth",
    "Freiburg": "SC Freiburg", "Monchengladbach": "Borussia M.Gladbach",
    "Stuttgart": "VfB Stuttgart", "Bochum": "VfL Bochum",
    "Frankfurt": "Eintracht Frankfurt", "Koln": "FC Cologne",
    "Wolfsburg": "VfL Wolfsburg", "Bremen": "Werder Bremen",
    "RB Leipzig": "RB Leipzig", "Leverkusen": "Bayer Leverkusen",
    "Atletico": "Atletico Madrid", "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid", "Vallecano": "Rayo Vallecano",
    "Villarreal": "Villarreal", "Sevilla": "Sevilla",
    "Osasuna": "Osasuna", "Real Sociedad": "Real Sociedad",
    "Valencia": "Valencia", "Girona": "Girona", "Alaves": "Alaves",
    "Las Palmas": "Las Palmas", "Getafe": "Getafe",
    "Mallorca": "Mallorca", "Leganes": "Leganes",
    "Napoli": "Napoli", "Roma": "Roma", "Lazio": "Lazio",
    "Inter": "Inter", "Juventus": "Juventus", "Atalanta": "Atalanta",
    "Torino": "Torino", "Fiorentina": "Fiorentina", "Bologna": "Bologna",
    "Udinese": "Udinese", "Sassuolo": "Sassuolo", "Genoa": "Genoa",
    "Verona": "Hellas Verona", "Cagliari": "Cagliari", "Empoli": "Empoli",
    "Parma": "Parma", "Como": "Como", "Lecce": "Lecce", "Venezia": "Venezia",
    "Monza": "Monza", "Spezia": "Spezia", "Cremonese": "Cremonese",
    "Salernitana": "Salernitana", "Sampdoria": "Sampdoria",
    "PSG": "Paris Saint-Germain", "Marseille": "Marseille",
    "Lille": "Lille", "Lyon": "Lyon", "Monaco": "Monaco",
    "Nice": "Nice", "Rennes": "Rennes", "Lens": "Lens",
    "Strasbourg": "Strasbourg", "Nantes": "Nantes", "Reims": "Reims",
    "Montpellier": "Montpellier", "Toulouse": "Toulouse", "Brest": "Brest",
    "Le Havre": "Le Havre", "Metz": "Metz", "Auxerre": "Auxerre",
    "Angers": "Angers", "Lorient": "Lorient", "Troyes": "Troyes",
    "Clermont Foot": "Clermont Foot",
}


def load_fd():
    dfs = []
    for f in glob.glob("data/football_history/*_2[1-6]*.csv"):
        dfs.append(pd.read_csv(f, encoding="utf-8-sig"))
    raw = pd.concat(dfs, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    raw = raw.rename(columns={"FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr",
                              "Div": "div", "HomeTeam": "home_team", "AwayTeam": "away_team",
                              "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
                              "HTHG": "htg", "HTAG": "htag", "HS": "hs", "AS": "as",
                              "HST": "hst", "AST": "ast"})
    raw = raw.dropna(subset=["date", "fthg", "ftag"]).sort_values("date").reset_index(drop=True)
    raw["league"] = raw["div"]
    raw["ahh"] = np.nan
    raw["ou_over"] = np.nan
    raw["ou_under"] = np.nan
    for c in ("b365h", "b365d", "b365a", "htg", "htag", "hs", "as", "hst", "ast"):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def load_xg():
    """understat → 每队每场 xG 表"""
    rows = []
    for f in glob.glob("data/understat/*.json"):
        d = json.load(open(f, encoding="utf-8"))
        for dt in d.get("dates", []):
            if not dt.get("isResult") or not dt["xG"]["h"] or not dt["xG"]["a"]:
                continue
            rows.append({"date": pd.Timestamp(dt["datetime"]).normalize(),
                         "home": dt["h"]["title"], "away": dt["a"]["title"],
                         "xg_h": float(dt["xG"]["h"]), "xg_a": float(dt["xG"]["a"])})
    xg = pd.DataFrame(rows)
    # 长表：每队每场
    long = []
    for _, r in xg.iterrows():
        long.append({"date": r["date"], "team": r["home"], "xg_for": r["xg_h"],
                     "xg_against": r["xg_a"]})
        long.append({"date": r["date"], "team": r["away"], "xg_for": r["xg_a"],
                     "xg_against": r["xg_h"]})
    return pd.DataFrame(long)


def rolling_xg(team_hist, date, n=5):
    """该队 date 之前最近 n 场 xG 均值（for/against）"""
    past = team_hist[team_hist["date"] < date].tail(n)
    if len(past) == 0:
        return np.nan, np.nan
    return past["xg_for"].mean(), past["xg_against"].mean()


def main():
    t0 = time.time()
    fd = load_fd()
    xg = load_xg()
    xg = xg.sort_values("date")
    team_xg = {t: g for t, g in xg.groupby("team")}
    print(f"fd {len(fd)} 场 | xg {len(xg)} 条 | 队 {len(team_xg)}", flush=True)

    # 对齐队名 + 计算滚动 xG 特征
    f1 = f2 = 0
    xg_feats = []
    for i, r in fd.iterrows():
        h = NAME_MAP.get(r["home_team"], r["home_team"])
        a = NAME_MAP.get(r["away_team"], r["away_team"])
        ht = team_xg.get(h)
        at = team_xg.get(a)
        if ht is not None and at is not None:
            hf, hga = rolling_xg(ht, r["date"])
            af, aga = rolling_xg(at, r["date"])
            f1 += 1
        else:
            hf = hga = af = aga = np.nan
            f2 += 1
        xg_feats.append({"home_xg5": hf, "home_xga5": hga,
                         "away_xg5": af, "away_xga5": aga})
    xgf = pd.DataFrame(xg_feats)
    print(f"对齐成功 {f1} 场, 缺失 {f2} 场", flush=True)

    feat, state = build_features(fd)
    feat = pd.concat([feat.reset_index(drop=True), xgf], axis=1)
    print(f"特征 {feat.shape}, 耗时 {time.time()-t0:.1f}s", flush=True)
    maxd = feat["date"].max()
    ts = maxd - pd.Timedelta(days=30)
    vs = maxd - pd.Timedelta(days=90)
    tr = feat[feat["date"] <= vs].copy()
    va = feat[(feat["date"] > vs) & (feat["date"] <= ts)].copy()
    te = feat[feat["date"] > ts].copy()
    print(f"分割: 训练{len(tr)} 验证{len(va)} 测试{len(te)}", flush=True)

    drop_cols = {"fthg", "ftag", "ftr", "date", "league", "home_team", "away_team",
                 "htg", "htag", "htr", "hs", "as", "hst", "ast", "hc", "ac",
                 "hy", "ay", "hr", "ar", "div", "ahh"}
    xg_cols = ["home_xg5", "home_xga5", "away_xg5", "away_xga5"]

    def run(cols, tag):
        m = PoissonLightGBM()
        m.fit(tr[cols].values, tr["fthg"].values.astype(float),
              tr["ftag"].values.astype(float),
              va[cols].values, va["fthg"].values.astype(float),
              va["ftag"].values.astype(float), num_boost_round=2000, early_stopping=100)
        print(f"\n[模型 {tag}] 特征{len(cols)}", flush=True)
        for name, df in (("验证", va), ("测试", te)):
            grids = m.predict_grids(df[cols].values)
            yh = df["fthg"].values.astype(float)
            ya = df["ftag"].values.astype(float)
            n = len(yh)
            ph = np.zeros(n); pd_ = np.zeros(n); pa = np.zeros(n)
            for i in range(n):
                g = grids[i]
                ih, ja = np.indices(g.shape)
                ph[i] = g[ih > ja].sum(); pd_[i] = g[ih == ja].sum(); pa[i] = g[ih < ja].sum()
            yoh = np.zeros((n, 3))
            yoh[yh > ya, 0] = 1; yoh[yh == ya, 1] = 1; yoh[yh < ya, 2] = 1
            pm = np.stack([ph, pd_, pa], axis=1)
            acc = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
            ll = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
            res = evaluate_all(yh, ya, grids, m.predict_lambda_h(df[cols].values),
                               m.predict_lambda_a(df[cols].values))
            print(f"[{name}] 胜平负Acc={acc:.3f} LL={ll:.4f} 比分LL={res['log_loss']:.4f}", flush=True)

    cols_all = [c for c in feat.columns if c not in drop_cols]
    cols_no_xg = [c for c in cols_all if c not in xg_cols]
    run(cols_all, "A-含xG")
    run(cols_no_xg, "B-不含xG")


if __name__ == "__main__":
    main()
