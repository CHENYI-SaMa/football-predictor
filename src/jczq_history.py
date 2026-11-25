# -*- coding: utf-8 -*-
"""
500 竞彩足球历史开奖数据抓取
=============================
数据来源：https://zx.500.com/jczq/kaijiang.php （竞彩足球开奖，历史赛果）

抓取最近 N 天每天的比赛开奖（编号/赛事/时间/主队/让球/客队/半场/全场比分/
胜平负SP/让球SP/总进球SP/半全场SP），转换为与 football-data 一致的 schema，
输出 data/jczq_history.csv，可直接复用 src/features.py 特征工程训练。

用法：
    python src/jczq_history.py --days 180        # 抓最近 180 天
    python src/jczq_history.py --from 2025-08-01 --to 2026-08-07
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_CSV = BASE_DIR / "data" / "jczq_history.csv"
RAW_DIR = BASE_DIR / "data" / "jczq_history_raw"

HIST_URL = "https://zx.500.com/jczq/kaijiang.php"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Encoding": "identity",
}
SLEEP = 0.4  # 请求间隔（秒），礼貌抓取


def fetch_day(date: str, use_cache: bool = True) -> list[dict]:
    """
    抓取指定日期（YYYY-MM-DD）的竞彩开奖，返回比赛 dict 列表。
    每场固定 19 个 td：编号/赛事/时间/主队/让球/客队/(半场)全场比分/
    胜平负结果/SP/让球结果/SP/总进球/SP/半全场/SP。
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{date}.json"
    if use_cache and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass

    r = requests.get(HIST_URL, params={"playid": 4, "d": date},
                     headers=HEADERS, timeout=20,
                     proxies={"http": None, "https": None})   # 直连，避免本机代理残留干扰
    r.raise_for_status()
    html = r.content.decode("gb2312", errors="ignore")

    matches = []
    # 按 tr 行解析；playid=4（比分玩法）开奖页每场 15 个 td：
    # [0]编号 [1]赛事 [2]时间 [3]主队 [4]让球 [5]客队 [6]'(半场) 全场'比分
    # [7]全场比分(冗余) [9]全场比分SP [11]让球胜SP [12]让球平SP [13]让球负SP
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        tds = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t)).strip()
               for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        tds = [t for t in tds if t]
        if len(tds) < 14 or not re.match(r"^周[一二三四五六日]\d{3}$", tds[0]):
            continue
        # 比分：(半场) 全场，如 "(1:0) 2:0"；无半场时只有全场
        score_m = re.search(r"\((\d+)\s*:\s*(\d+)\)\s*(\d+)\s*:\s*(\d+)", tds[6])
        if score_m:
            htg, htag = int(score_m.group(1)), int(score_m.group(2))
            fthg, ftag = int(score_m.group(3)), int(score_m.group(4))
        else:
            score_m2 = re.search(r"(\d+)\s*:\s*(\d+)", tds[6])
            if not score_m2:
                continue
            fthg, ftag = int(score_m2.group(1)), int(score_m2.group(2))
            htg = htag = None
        ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
        try:
            rang = int(tds[4])
        except ValueError:
            rang = 0
        # 半全场结果（由半/全场比分推导）：胜胜/平胜/负胜/胜平/平平/负平/胜负/平负/负负
        htft = "-"
        if htg is not None:
            hr = "胜" if htg > htag else ("平" if htg == htag else "负")
            fr = "胜" if fthg > ftag else ("平" if fthg == ftag else "负")
            htft = hr + fr
        # 让球胜平负三项 SP（完整赔率）
        def _f(x):
            try:
                return float(x) if x and x not in ("--", "—", "&nbsp;") else None
            except ValueError:
                return None
        matches.append({
            "date": date, "no": tds[0], "league": tds[1], "match_time": tds[2],
            "home_team": tds[3], "away_team": tds[5],
            "rang": rang, "fthg": fthg, "ftag": ftag,
            "htg": htg, "htag": htag, "ftr": ftr, "htft": htft,
            "sp_win": _f(tds[9]),          # 全场比分 SP
            "rsp_h": _f(tds[11]),          # 让球后 胜 SP
            "rsp_d": _f(tds[12]),          # 让球后 平 SP
            "rsp_a": _f(tds[13]),          # 让球后 负 SP
            "result": ftr,
        })
    # 只缓存非空结果：当天中午抓到的空结果（比赛未开奖）不写缓存，
    # 否则会永久阻塞后续回填（fetch_history 读缓存永远为空）
    if use_cache and matches:
        cache.write_text(json.dumps(matches, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return matches


def fetch_history(dates: list[str], verbose: bool = True) -> pd.DataFrame:
    """抓取多个日期的开奖并合并为标准 schema 的 DataFrame。"""
    all_m = []
    for date in dates:
        try:
            day = fetch_day(date)
        except Exception as e:
            if verbose:
                print(f"  [跳过] {date}: {e}")
            continue
        if verbose:
            print(f"  {date}: {len(day)} 场")
        all_m.extend(day)
        time.sleep(SLEEP)

    if not all_m:
        raise RuntimeError("未抓到任何数据")

    df = pd.DataFrame(all_m)
    # 转标准 schema（与 data_loader.load_football_data 一致）
    # 注意：b365h/d/a 用『让球胜平负三项 SP』（完整赔率，对应让球盘）
    std = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "league": df["league"],
        "home_team": df["home_team"],
        "away_team": df["away_team"],
        "fthg": df["fthg"], "ftag": df["ftag"],
        "ftr": df["ftr"],
        "htg": pd.NA, "htag": pd.NA,
        "hs": pd.NA, "as": pd.NA, "hst": pd.NA, "ast": pd.NA,
        "b365h": df["rsp_h"], "b365d": df["rsp_d"], "b365a": df["rsp_a"],
        "ou_over": pd.NA, "ou_under": pd.NA,
        "ahh": df["rang"],
    })
    std = std.sort_values("date").reset_index(drop=True)
    return std


def date_range(start: str, end: str) -> list[str]:
    dr = pd.date_range(start, end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dr]


TRADE_URL = "https://trade.500.com/jczq/"


def fetch_day_trade(date: str) -> list[dict]:
    """备用数据源：500 网竞彩赛事+赛果页（trade.500.com，playid=269 让球玩法）。
    与 fetch_day（zx.500.com 开奖页）互补：开奖页未录入/漏抓时，这里按对战编号
    （data-matchnum，如 周一001）仍可拿到已结束比赛的真实比分（格式 1:0）。
    结构对齐 fetch_day 的输出：date/no/league/match_time/home_team/away_team/
    rang/fthg/ftag/htft/result（htg/htag 该页无半场比分，置 None）。
    只在 fetch_day 抓不全时兜底调用，不影响原有开奖页主路径。
    """
    r = requests.get(TRADE_URL, params={"playid": 269, "g": 2, "date": date},
                     headers=HEADERS, timeout=20,
                     proxies={"http": None, "https": None})
    r.raise_for_status()
    html = r.content.decode("gb2312", errors="ignore")

    matches = []
    for tr in re.findall(r"(<tr[^>]*>[\s\S]*?</tr>)", html):
        m_no = re.search(r'data-matchnum="(周[一二三四五六日]\d{3})"', tr)
        if not m_no or 'data-isend="1"' not in tr:
            continue
        no = m_no.group(1)
        # 联赛
        league_m = re.search(r'td-evt.*?title="([^"]+)"', tr, re.S)
        league = league_m.group(1) if league_m else ""
        # 截止时间（如 09-01 00:00）
        time_m = re.search(r'td-endtime[^>]*title="([^"]+)"', tr)
        match_time = time_m.group(1) if time_m else ""
        # 主/客队 + 比分：td-team 内 team-l / score / team-r
        home_m = re.search(r'class="team-l"[^>]*title="([^"]+)"', tr)
        away_m = re.search(r'class="team-r"[^>]*title="([^"]+)"', tr)
        score_m = re.search(r'class="score"[^>]*>(?:<[^>]+>)*\s*(\d+)\s*:\s*(\d+)', tr)
        if not (home_m and away_m and score_m):
            continue
        home, away = home_m.group(1), away_m.group(1)
        fthg, ftag = int(score_m.group(1)), int(score_m.group(2))
        # 让球数：data-rangqiu="-1"（主让n为负，客让为正，与训练一致）
        rang_m = re.search(r'data-rangqiu="(-?\d+)"', tr)
        rang = int(rang_m.group(1)) if rang_m else 0
        ftr = "H" if fthg > ftag else ("D" if fthg == ftag else "A")
        matches.append({
            "date": date, "no": no, "league": league, "match_time": match_time,
            "home_team": home, "away_team": away,
            "rang": rang, "fthg": fthg, "ftag": ftag,
            "htg": None, "htag": None, "ftr": ftr, "htft": "-",
            "sp_win": None, "rsp_h": None, "rsp_d": None, "rsp_a": None,
            "result": ftr, "src": "trade",
        })
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抓取 500 竞彩历史开奖")
    parser.add_argument("--days", type=int, default=180, help="抓最近 N 天")
    parser.add_argument("--from", dest="from_", type=str, default=None)
    parser.add_argument("--to", type=str, default=None)
    args = parser.parse_args()

    if args.from_ and args.to:
        dates = date_range(args.from_, args.to)
    else:
        to = pd.Timestamp.now().strftime("%Y-%m-%d")
        from_ = (pd.Timestamp.now() - pd.Timedelta(days=args.days - 1)).strftime("%Y-%m-%d")
        dates = date_range(from_, to)

    print(f"抓取 {len(dates)} 天: {dates[0]} ~ {dates[-1]}")
    df = fetch_history(dates)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n共 {len(df)} 场，已保存 -> {OUT_CSV}")
    print("联赛分布:", df["league"].value_counts().head(15).to_dict())
    print("时间范围:", df["date"].min().date(), "~", df["date"].max().date())
    print("场均进球:", round(df["fthg"].mean() + df["ftag"].mean(), 2))
