# -*- coding: utf-8 -*-
"""merge_xg.py —— 把 Understat 逐场 xG 合并进 500 竞彩训练数据
=============================================================
输入: data/jczq_history.csv（500 开奖，中文队名 + 让球SP）
      data/understat_xg/*.csv（五大联赛英文队名 + xG）
匹配: 中文队名 → TEAM_NAME_MAP 英文 → norm → (日期±1天 + 两队 + 比分) 匹配
输出: data/jczq_history_xg.csv（追加 xg_h/xg_a 列，非五大联赛为 NaN）
用法: python merge_xg.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.jczq_fetcher import TEAM_NAME_MAP  # noqa: E402


def norm(s: str) -> str:
    s = str(s).replace("_", " ").lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# TEAM_NAME_MAP 英文值 → understat_xg(fd) 队名 别名（norm 后）
EN_ALIAS = {
    "paris sg": "psg", "atletico madrid": "atletico",
    "mgladbach": "monchengladbach", "schalke 04": "schalke",
    "st pauli": "st pauli", "barcelona": "barcelona",
    "real madrid": "real madrid", "bayern munich": "bayern munich",
    "eintracht frankfurt": "frankfurt", "hertha": "hertha",
    "cologne": "cologne", "mainz": "mainz", "hamburg": "hamburg",
    "nottm forest": "nottm forest", "leeds": "leeds",
}


def to_en(zh: str) -> str | None:
    en = TEAM_NAME_MAP.get(zh)
    if not en:
        return None
    e = norm(en)
    return EN_ALIAS.get(e, e)


def main():
    hist = pd.read_csv(BASE / "data" / "jczq_history.csv", parse_dates=["date"])
    print(f"500 历史: {len(hist)} 场（{hist['date'].min().date()} ~ {hist['date'].max().date()}）")

    xg_parts = []
    for p in sorted((BASE / "data" / "understat_xg").glob("*.csv")):
        df = pd.read_csv(p, parse_dates=["date"])
        xg_parts.append(df)
    xg = pd.concat(xg_parts, ignore_index=True)
    xg["date_s"] = xg["date"].dt.strftime("%Y-%m-%d")
    print(f"xG 表: {len(xg)} 场（五大联赛 {xg['date'].dt.year.nunique()} 个赛季）")

    # xG 表索引：norm 队名 -> 行
    xg["nh"] = xg["home_team"].map(norm)
    xg["na"] = xg["away_team"].map(norm)

    # football-data 附加统计（角球/犯规/红黄牌/大小球）：五大联赛全部赛季合并
    fd_parts = []
    for p in sorted((BASE / "data" / "football_history").glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        keep = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        extra = [c for c in ("HC", "AC", "HF", "AF", "HY", "AY") if c in df.columns]
        # 大小球盘口（2.5 线：B365 优先，缺则 Avg/P）
        ou = None
        for cand in ("B365>2.5", "Avg>2.5", "P>2.5"):
            if cand in df.columns:
                ou = cand
                break
        if ou:
            extra += [ou, ou.replace(">", "<")]
        if not extra:
            continue
        df = df.rename(columns={"Date": "date", "HomeTeam": "home",
                                "AwayTeam": "away", "FTHG": "fh", "FTAG": "fa"})
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y",
                                    errors="coerce").dt.strftime("%Y-%m-%d")
        fd_parts.append(df[["date", "home", "away", "fh", "fa"] + extra])
    if fd_parts:
        fd_all = pd.concat(fd_parts, ignore_index=True)
        fd_all = fd_all.dropna(subset=["date", "fh", "fa"])
        fd_all["nh"] = fd_all["home"].map(norm)
        fd_all["na"] = fd_all["away"].map(norm)
        print(f"football-data 附加表: {len(fd_all)} 场")
    else:
        fd_all = pd.DataFrame()

    n_match = n_fail = 0
    fail_examples = []
    rows = []
    for _, r in hist.iterrows():
        en_h = to_en(r["home_team"])
        en_a = to_en(r["away_team"])
        row = r.to_dict()
        row["xg_h"] = np.nan
        row["xg_a"] = np.nan
        for c in ("hc_h", "ac_h", "hf_h", "af_h", "hy_h", "ay_h",
                  "hc_a", "ac_a", "hf_a", "af_a", "hy_a", "ay_a"):
            row[c] = np.nan
        row["ou_over"] = np.nan
        row["ou_under"] = np.nan
        if en_h and en_a:
            d0 = r["date"].strftime("%Y-%m-%d")
            d1 = (r["date"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            dm = (r["date"] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            cand = xg[(xg["date_s"].isin([dm, d0, d1])) &
                      (((xg["nh"] == en_h) & (xg["na"] == en_a)) |
                       ((xg["nh"] == en_a) & (xg["na"] == en_h)))]
            if len(cand):
                m = cand.iloc[0]
                if int(m["fthg"]) == int(r["fthg"]) and int(m["ftag"]) == int(r["ftag"]):
                    if norm(m["home_team"]) == en_h:
                        row["xg_h"], row["xg_a"] = m["xg_h"], m["xg_a"]
                    else:
                        row["xg_h"], row["xg_a"] = m["xg_a"], m["xg_h"]
                    # 附加统计（football-data）
                    if len(fd_all):
                        fc = fd_all[(fd_all["date"].isin([dm, d0, d1])) &
                                    (((fd_all["nh"] == en_h) & (fd_all["na"] == en_a)) |
                                     ((fd_all["nh"] == en_a) & (fd_all["na"] == en_h)))]
                        if len(fc) and int(fc.iloc[0]["fh"]) == int(r["fthg"]) \
                                and int(fc.iloc[0]["fa"]) == int(r["ftag"]):
                            mm = fc.iloc[0]
                            if norm(mm["home"]) == en_h:
                                row["hc_h"], row["ac_h"] = mm.get("HC", np.nan), mm.get("AC", np.nan)
                                row["hf_h"], row["af_h"] = mm.get("HF", np.nan), mm.get("AF", np.nan)
                                row["hy_h"], row["ay_h"] = mm.get("HY", np.nan), mm.get("AY", np.nan)
                                row["hc_a"], row["ac_a"] = mm.get("AC", np.nan), mm.get("HC", np.nan)
                                row["hf_a"], row["af_a"] = mm.get("AF", np.nan), mm.get("HF", np.nan)
                                row["hy_a"], row["ay_a"] = mm.get("AY", np.nan), mm.get("HY", np.nan)
                            else:
                                row["hc_a"], row["ac_a"] = mm.get("HC", np.nan), mm.get("AC", np.nan)
                                row["hf_a"], row["af_a"] = mm.get("HF", np.nan), mm.get("AF", np.nan)
                                row["hy_a"], row["ay_a"] = mm.get("HY", np.nan), mm.get("AY", np.nan)
                                row["hc_h"], row["ac_h"] = mm.get("AC", np.nan), mm.get("HC", np.nan)
                                row["hf_h"], row["af_h"] = mm.get("AF", np.nan), mm.get("HF", np.nan)
                                row["hy_h"], row["ay_h"] = mm.get("AY", np.nan), mm.get("HY", np.nan)
                            # 大小球盘口（不分主客，同一 2.5 线）
                            ou_ov = mm.get("B365>2.5", None)
                            if ou_ov is None:
                                ou_ov = mm.get("Avg>2.5", None)
                            if ou_ov is None:
                                ou_ov = mm.get("P>2.5", None)
                            if ou_ov is not None:
                                row["ou_over"] = ou_ov
                                ou_un = mm.get("B365<2.5", None) or mm.get("Avg<2.5", None) or mm.get("P<2.5", None)
                                row["ou_under"] = ou_un
                    n_match += 1
                else:
                    n_fail += 1
                    if len(fail_examples) < 5:
                        fail_examples.append(
                            f"{d0} {r['home_team']}vs{r['away_team']} "
                            f"比分{int(r['fthg'])}:{int(r['ftag'])} "
                            f"xG表={int(m['fthg'])}:{int(m['ftag'])}")
            else:
                n_fail += 1
                if len(fail_examples) < 5 and r["date"] >= pd.Timestamp("2025-07-01"):
                    fail_examples.append(f"{d0} {r['home_team']}vs{r['away_team']} 无xG候选")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(BASE / "data" / "jczq_history_xg.csv", index=False, encoding="utf-8-sig")
    print(f"\n匹配到 xG: {n_match} 场（覆盖 {n_match / len(hist):.1%}）")
    print(f"未匹配: {n_fail} 场")
    if fail_examples:
        print("失败样例:")
        for f in fail_examples:
            print("  ", f)
    cov = out["xg_h"].notna()
    print(f"\n近12个月 xG 覆盖率: {cov[out['date'] >= '2025-08-01'].mean():.1%}")
    print(f"→ data/jczq_history_xg.csv 已保存（{len(out)} 行）")


if __name__ == "__main__":
    main()
