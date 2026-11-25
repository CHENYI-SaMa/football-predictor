# -*- coding: utf-8 -*-
"""Understat 逐场 xG 抓取器（2026 新版：队页日历视角）
=====================================================
Understat 2026 改版后联赛页只显示当周比赛；逐场 xG 数据在球队页
(https://understat.com/team/{Slug}/{Season}) 的日历表格中：
  每场 4 行文本块 = 日期 / 比分(主:客) / xG(主xG:客xG 粘连) / 对手
  两队页显示同一场比赛的相同主:客块 → 抓全队页后按 (日期,比分,对手) 去重
主客归属：与 football-data 本地赛程按 (日期+比分) join 确定，并校验比分

用法:
  python fetch_understat_xg.py --league EPL --season 2025
  python fetch_understat_xg.py --all   # 五大联赛近3个赛季
输出:
  data/understat_xg/{league}_{season}.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

CHROME_EXE = r"C:\Users\CHENYI\AppData\Local\ms-playwright\chromium-1234\chrome-win64\chrome.exe"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

LEAGUES = {
    "EPL": "https://understat.com/league/EPL",
    "La_liga": "https://understat.com/league/La_liga",
    "Bundesliga": "https://understat.com/league/Bundesliga",
    "Serie_A": "https://understat.com/league/Serie_A",
    "Ligue_1": "https://understat.com/league/Ligue_1",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

DATE_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2}), (\d{4})$")
SCORE_RE = re.compile(r"^(\d)(\d)$")
XG_RE = re.compile(r"^(\d\.\d{2})(\d\.\d{2})$")


def norm(name: str) -> str:
    """队名归一化（用于 slug 与 football-data 匹配）。"""
    s = name.replace("_", " ").lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# slug -> football-data 队名（值已 norm 归一化，便于直接 == 比较）
FD_ALIAS = {
    "manchester city": "man city", "manchester united": "man united",
    "newcastle united": "newcastle", "wolverhampton wanderers": "wolves",
    "nottingham forest": "nottm forest", "leeds united": "leeds",
    "west ham united": "west ham", "tottenham hotspur": "tottenham",
    "brighton hove albion": "brighton",
    # 西甲（fd 用缩写；understat 显示完整名）
    "atletico madrid": "ath madrid", "athletic bilbao": "ath bilbao",
    "athletic club": "ath bilbao", "real sociedad": "sociedad",
    "rayo vallecano": "vallecano", "real oviedo": "oviedo",
    "real betis": "betis", "celta vigo": "celta", "espanyol": "espanol",
    # 德甲（understat 显示完整名）
    "eintracht frankfurt": "ein frankfurt", "koln": "fc koln",
    "borussia dortmund": "dortmund",
    "borussia monchengladbach": "mgladbach",
    "bayer leverkusen": "leverkusen",
    "st pauli": "st pauli",
    "hamburger sv": "hamburg", "borussia m.gladbach": "mgladbach",
    "mainz 05": "mainz", "fc cologne": "fc koln",
    "rasenballsport leipzig": "rb leipzig", "vfb stuttgart": "stuttgart",
    "1 fc heidenheim": "heidenheim", "fc heidenheim": "heidenheim",
    # 意甲/法甲
    "inter milan": "inter", "ssc napoli": "napoli",
    "as roma": "roma", "paris saint germain": "psg",
    "paris saint-germain": "psg", "as monaco": "monaco",
    "olympique lyonnais": "lyon", "olympique marseille": "marseille",
    "losc lille": "lille", "ogc nice": "nice", "stade rennais": "rennes",
    "racing strasbourg": "strasbourg",
}

# football-data 队名 -> understat slug（解决 league 页当周列表不全的问题）
FD_TO_SLUG = {
    # 西甲
    "Alaves": "Alaves", "Ath Bilbao": "Athletic_Bilbao", "Ath Madrid": "Atletico_Madrid",
    "Barcelona": "Barcelona", "Betis": "Real_Betis", "Celta": "Celta_Vigo",
    "Elche": "Elche", "Espanol": "Espanyol", "Getafe": "Getafe", "Girona": "Girona",
    "Levante": "Levante", "Mallorca": "Mallorca", "Osasuna": "Osasuna",
    "Oviedo": "Real_Oviedo", "Real Madrid": "Real_Madrid", "Sevilla": "Sevilla",
    "Sociedad": "Real_Sociedad", "Valencia": "Valencia", "Vallecano": "Rayo_Vallecano",
    "Villarreal": "Villarreal",
    # 德甲
    "Augsburg": "Augsburg", "Bayern Munich": "Bayern_Munich",
    "Dortmund": "Borussia_Dortmund", "Ein Frankfurt": "Eintracht_Frankfurt",
    "Freiburg": "Freiburg", "Hamburg": "Hamburg", "Heidenheim": "FC_Heidenheim",
    "Hoffenheim": "Hoffenheim", "Leverkusen": "Bayer_Leverkusen",
    "Mainz": "Mainz", "RB Leipzig": "RB_Leipzig", "St Pauli": "St._Pauli",
    "Stuttgart": "Stuttgart", "Union Berlin": "Union_Berlin",
    "Werder Bremen": "Werder_Bremen", "Wolfsburg": "Wolfsburg",
    # 意甲/法甲（fd 全名大多直接可用，补充缩写）
    "Ath Madrid": "Atletico_Madrid",
    "M'gladbach": "Borussia_M.Gladbach",
    "PSG": "Paris_Saint_Germain", "Paris SG": "Paris_Saint_Germain",
    "Ein Frankfurt": "Eintracht_Frankfurt", "FC Koln": "FC_Cologne",
}

FD_LEAGUE_FILE = {"EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1",
                  "Serie_A": "I1", "Ligue_1": "F1"}


def parse_team_page(txt: str, team_slug: str) -> list[dict]:
    """解析队页 innerText → 比赛块列表。"""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    matches = []
    i = 0
    while i < len(lines) - 3:
        m = DATE_RE.match(lines[i])
        if m:
            mon, day, yr = m.group(1), int(m.group(2)), int(m.group(3))
            sc = SCORE_RE.match(lines[i + 1])
            xg = XG_RE.match(lines[i + 2])
            if sc and xg:
                matches.append({
                    "date": f"{yr}-{MONTHS[mon]:02d}-{day:02d}",
                    "fh": int(sc.group(1)), "fa": int(sc.group(2)),
                    "xg_h": float(xg.group(1)), "xg_a": float(xg.group(2)),
                    "opp": lines[i + 3], "team": team_slug,
                })
                i += 4
                continue
        i += 1
    return matches


def fetch_team_page(page, slug: str, season: int, retry: int = 2) -> str:
    for attempt in range(retry):
        try:
            page.goto(f"https://understat.com/team/{slug}/{season}",
                      timeout=40000, wait_until="domcontentloaded")
            page.wait_for_timeout(3500 + attempt * 1500)
            return page.evaluate("document.body.innerText")
        except Exception:
            if attempt == retry - 1:
                raise
            time.sleep(2)
    return ""


def get_team_slugs(page, league_url: str, season: int, league: str = "") -> list[str]:
    """优先从 football-data 队名构造完整 slug 列表（league 页当周视图不全）；
    无本地表时回退 league 页抓取。"""
    slugs: list[str] = []
    if league:
        code = FD_LEAGUE_FILE.get(league)
        short = f"{season % 100:02d}{(season + 1) % 100:02d}"
        p = BASE / "data" / "football_history" / f"{code}_{short}.csv"
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p)
            teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
            for t in teams:
                s = FD_TO_SLUG.get(t) or t.replace(" ", "_")
                if s not in slugs:
                    slugs.append(s)
            if slugs:
                return slugs
    try:
        page.goto(f"{league_url}/{season}", timeout=40000, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
        links = page.evaluate(
            'Array.from(document.querySelectorAll("a")).map(a=>a.href)')
        for u in links:
            m = re.search(r"/team/([^/]+)/", u)
            if m and m.group(1) not in slugs:
                slugs.append(m.group(1))
    except Exception:
        pass
    return slugs


def load_fd(league: str, season: int) -> pd.DataFrame:
    """读 football-data 本地赛季表（确定主客 + 校验比分）。"""
    fd_map = {"EPL": "E0", "La_liga": "SP1", "Bundesliga": "D1",
              "Serie_A": "I1", "Ligue_1": "F1"}
    code = fd_map.get(league, "E0")
    short = f"{season % 100:02d}{(season + 1) % 100:02d}"
    p = BASE / "data" / "football_history" / f"{code}_{short}.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df = df.rename(columns={"Date": "date", "HomeTeam": "home",
                            "AwayTeam": "away", "FTHG": "fh", "FTAG": "fa"})
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y").dt.strftime("%Y-%m-%d")
    return df[["date", "home", "away", "fh", "fa"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="EPL", choices=list(LEAGUES))
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--all", action="store_true", help="五大联赛近3个赛季")
    ap.add_argument("--sleep", type=float, default=0.8)
    args = ap.parse_args()

    out_dir = BASE / "data" / "understat_xg"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.all:
        for lg in LEAGUES:
            for s in (2025, 2024, 2023):
                jobs.append((lg, s))
    else:
        jobs.append((args.league, args.season))

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True, executable_path=CHROME_EXE,
            args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(user_agent=UA)
        page = ctx.new_page()

        for league, season in jobs:
            t0 = time.time()
            try:
                slugs = get_team_slugs(page, LEAGUES[league], season, league)
            except Exception as e:
                print(f"[{league}/{season}] 获取球队列表失败: {e}")
                continue
            print(f"[{league}/{season}] 球队 {len(slugs)} 支")

            blocks = []
            for slug in slugs:
                try:
                    txt = fetch_team_page(page, slug, season)
                    ms = parse_team_page(txt, slug)
                    blocks.extend(ms)
                except Exception as e:
                    print(f"  {slug}: 抓取失败 {e}")
                time.sleep(args.sleep)

            if not blocks:
                print(f"[{league}/{season}] 无数据")
                continue
            df = pd.DataFrame(blocks)
            print(f"[{league}/{season}] 原始块 {len(df)}（20队×38场=760）")

            # 去重：同一场比赛在两队页各出现一次（主:客块相同）
            key = df.apply(
                lambda r: (r["date"], r["fh"], r["fa"],
                           tuple(sorted([norm(r["team"]), norm(r["opp"])]))),
                axis=1)
            df = df.loc[key.drop_duplicates().index].reset_index(drop=True)
            print(f"[{league}/{season}] 去重后 {len(df)} 场")

            # 与 football-data join 定主客（队名集合匹配 + 比分严格校验）
            fd = load_fd(league, season)
            if len(fd):
                rows, fail = [], []
                for _, r in df.iterrows():
                    t1 = FD_ALIAS.get(norm(r["team"]), norm(r["team"]))
                    t2 = FD_ALIAS.get(norm(r["opp"]), norm(r["opp"]))
                    # Understat 按访问者时区显示日期，晚场可能跨天 → 容差 ±1 天
                    d0 = pd.Timestamp(r["date"])
                    dates = [d0 - pd.Timedelta(days=1), d0, d0 + pd.Timedelta(days=1)]
                    cand = fd[(fd["date"].isin([d.strftime("%Y-%m-%d") for d in dates])) &
                              (((fd["home"].map(norm) == t1) & (fd["away"].map(norm) == t2)) |
                               ((fd["home"].map(norm) == t2) & (fd["away"].map(norm) == t1)))]
                    if not len(cand):
                        fail.append(f"无候选:{r['date']} {r['team']}vs{r['opp']} {r['fh']}:{r['fa']}")
                        continue
                    m = cand.iloc[0]
                    if int(m["fh"]) != r["fh"] or int(m["fa"]) != r["fa"]:
                        fail.append(f"比分不符:{r['date']} {m['home']}vs{m['away']} "
                                    f"fd={m['fh']}:{m['fa']} us={r['fh']}:{r['fa']}")
                        continue
                    if norm(m["home"]) == t1:
                        home, away = m["home"], m["away"]
                    else:
                        home, away = m["away"], m["home"]
                    rows.append({
                        "date": m["date"], "league": league,
                        "home_team": home, "away_team": away,
                        "fthg": r["fh"], "ftag": r["fa"],
                        "xg_h": r["xg_h"], "xg_a": r["xg_a"],
                    })
                df = pd.DataFrame(rows)
                print(f"[{league}/{season}] join 成功 {len(df)} 场，失败 {len(fail)}")
                if fail:
                    print("  失败样例:")
                    for f in fail[:8]:
                        print("   ", f)
            else:
                print(f"[{league}/{season}] 无 football-data 本地表，仅输出原始块")

            p_out = out_dir / f"{league}_{season}.csv"
            df.to_csv(p_out, index=False, encoding="utf-8-sig")
            print(f"[{league}/{season}] → {p_out} ({len(df)} 场, "
                  f"耗时 {time.time()-t0:.0f}s)")
            if len(df):
                print(f"  xG 范围: {df['xg_h'].min():.2f}~{df['xg_h'].max():.2f} / "
                      f"{df['xg_a'].min():.2f}~{df['xg_a'].max():.2f}")

        browser.close()


if __name__ == "__main__":
    main()
