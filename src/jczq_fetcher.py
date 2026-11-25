# -*- coding: utf-8 -*-
"""
500 彩票网（竞彩足球）数据抓取模块
=================================
数据来源：https://trade.500.com/jczq/ （竞彩足球数据大厅）

功能：
1. fetch_matches()        —— 抓取当天/指定日期竞彩足球比赛列表
   （比赛编号、赛事、开赛时间、主客队中文名、让球数、竞彩 SP 胜平负赔率）
2. fetch_team_form(id)    —— 抓取球队近 10 场战绩（赛事/比分/赛果/盘路/大小），
   并汇总为模型可用特征（近 3/5/8 场积分、进球、失球、赢盘率；主客场分列）
3. build_feature_vector() —— 用 500 数据构建与 src/features.py 对齐的特征向量，
   供 predict_jczq.py 直接喂给已训练模型

说明：
- 页面编码为 GB2312；请求带 UA，失败时自动重试 1 次；
- 抓取结果缓存在本地 data/jczq_cache/，避免频繁请求；
- 数据仅用于个人研究/娱乐预测，请遵守网站使用条款。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import requests

from path_helper import get_base_dir  # noqa: E402
BASE_DIR = get_base_dir()
CACHE_DIR = BASE_DIR / "data" / "jczq_cache"
JCZQ_URL = "https://trade.500.com/jczq/"
TEAM_URL = "https://liansai.500.com/team/{tid}/"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.500.com/",
    "Connection": "keep-alive",
}

# 竞彩赛事中文名 -> 模型训练联赛代码（仅映射五大联赛；其余联赛映射失败=缺失）
# 注意：模型训练覆盖 E0/SP1/D1/I1/F1，500 上的日职/挪超/荷甲等不在其中，
# 该类比赛靠 500 实时战绩特征 + 竞彩赔率特征支撑。
LEAGUE_MAP = {
    "英超": "E0", "英格兰超级联赛": "E0",
    "西甲": "SP1", "西班牙甲级联赛": "SP1",
    "德甲": "D1", "德国甲级联赛": "D1",
    "意甲": "I1", "意大利甲级联赛": "I1",
    "法甲": "F1", "法国甲级联赛": "F1",
}
LEAGUE_IDX = {"E0": 0.0, "SP1": 1.0, "D1": 2.0, "I1": 3.0, "F1": 4.0}

# 500 中文队名 -> football-data.co.uk 英文队名（模型训练命名）
# 覆盖五大联赛常见球队；未列出的球队按"未知球队"处理（走 500 战绩特征路径）
TEAM_NAME_MAP = {
    # 英超
    "曼城": "Man City", "曼彻斯特城": "Man City", "利物浦": "Liverpool",
    "阿森纳": "Arsenal", "切尔西": "Chelsea", "曼联": "Man United",
    "曼彻斯特联": "Man United", "托特纳姆热刺": "Tottenham",
    "纽卡斯尔联": "Newcastle", "纽卡斯尔": "Newcastle",
    "阿斯顿维拉": "Aston Villa", "西汉姆联": "West Ham",
    "布莱顿": "Brighton", "水晶宫": "Crystal Palace",
    "富勒姆": "Fulham", "布伦特福德": "Brentford",
    "埃弗顿": "Everton", "狼队": "Wolves", "狼": "Wolves",
    "莱斯特城": "Leicester", "利兹联": "Leeds", "诺丁汉森林": "Nott'm Forest",
    "南安普顿": "Southampton", "伯恩茅斯": "Bournemouth",
    "米德尔斯堡": "Middlesbrough", "女王公园巡游者": "QPR",
    "斯旺西": "Swansea", "伯明翰": "Birmingham", "朴次茅斯": "Portsmouth",
    "米尔沃尔": "Millwall", "雷克斯汉姆": "Wrexham",
    # 西甲
    "皇家马德里": "Real Madrid", "巴塞罗那": "Barcelona",
    "马德里竞技": "Atletico Madrid", "塞维利亚": "Sevilla",
    "皇家社会": "Real Sociedad", "皇家贝蒂斯": "Betis",
    "比利亚雷亚尔": "Villarreal", "瓦伦西亚": "Valencia",
    "毕尔巴鄂竞技": "Athletic Club", "毕尔巴鄂": "Athletic Club",
    "奥萨苏纳": "Osasuna", "塞尔塔": "Celta", "赫塔菲": "Getafe",
    "赫罗纳": "Girona", "阿拉维斯": "Alaves", "马洛卡": "Mallorca",
    "拉斯帕尔马斯": "Las Palmas", "莱加内斯": "Leganes",
    # 德甲
    "拜仁慕尼黑": "Bayern Munich", "多特蒙德": "Dortmund",
    "RB莱比锡": "RB Leipzig", "勒沃库森": "Leverkusen",
    "法兰克福": "Eintracht Frankfurt", "斯图加特": "Stuttgart",
    "门兴格拉德巴赫": "M'gladbach", "沃尔夫斯堡": "Wolfsburg",
    "弗赖堡": "Freiburg", "霍芬海姆": "Hoffenheim",
    "柏林联合": "Union Berlin", "美因茨": "Mainz", "奥格斯堡": "Augsburg",
    "云达不莱梅": "Werder Bremen", "波鸿": "Bochum",
    "柏林赫塔": "Hertha", "汉堡": "Hamburg", "沙尔克04": "Schalke 04",
    "科隆": "Cologne", "圣保利": "St Pauli",
    # 意甲
    "国际米兰": "Inter", "AC米兰": "Milan", "尤文图斯": "Juventus",
    "那不勒斯": "Napoli", "罗马": "Roma", "拉齐奥": "Lazio",
    "亚特兰大": "Atalanta", "佛罗伦萨": "Fiorentina",
    "都灵": "Torino", "博洛尼亚": "Bologna", "乌迪内斯": "Udinese",
    "热那亚": "Genoa", "维罗纳": "Verona", "卡利亚里": "Cagliari",
    "恩波利": "Empoli", "莱切": "Lecce", "帕尔马": "Parma",
    "蒙扎": "Monza", "科莫": "Como", "威尼斯": "Venezia",
    # 法甲
    "巴黎圣日耳曼": "Paris SG", "马赛": "Marseille", "里昂": "Lyon",
    "摩纳哥": "Monaco", "里尔": "Lille", "尼斯": "Nice",
    "朗斯": "Lens", "雷恩": "Rennes", "斯特拉斯堡": "Strasbourg",
    "南特": "Nantes", "蒙彼利埃": "Montpellier", "图卢兹": "Toulouse",
    "兰斯": "Reims", "布雷斯特": "Brest", "欧塞尔": "Auxerre",
    "勒阿弗尔": "Le Havre", "昂热": "Angers", "圣埃蒂安": "St Etienne",
}


def _get(url: str, timeout: int = 20) -> str:
    """带 UA 与重试的 GET，返回按 GB2312 解码的文本。
    加固：显式禁用代理（本机可能有残留代理端口/安全软件拦截，直连更稳）；
    重试 3 次指数退避；失败给出中文提示。"""
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout,
                             proxies={"http": None, "https": None})
            r.raise_for_status()
            return r.content.decode("gb2312", errors="ignore")
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < 2:
                time.sleep(1.5 ** attempt + 0.5)   # 1.0s / 2.0s 退避
    raise RuntimeError(f"抓取失败 {url}（连试3次失败）: {last}") from last


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


# ---------------------------------------------------------------------------
# 1. 比赛列表
# ---------------------------------------------------------------------------
def fetch_matches(use_cache: bool = True, cache_hours: float = 3.0,
                  date: str | None = None) -> list[dict]:
    """
    抓取竞彩足球比赛列表（默认当天开售场次；指定 date 时抓取该日期）。

    Returns
    -------
    list[dict]
        [{no, league, match_time, home, away, home_id, away_id,
          rang, sp_h, sp_d, sp_a, range_sp_h, range_sp_d, range_sp_a}]
        其中 rang 为竞彩让球数（如 +1 表示客队让 1 球）；
        sp_* 为不让球胜平负 SP；range_sp_* 为让球后胜平负 SP。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "matches.json"
    if use_cache and cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < cache_hours * 3600:
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass

    # 500 网站默认页(不带 date)今天起不再展示当天比赛(显示"暂无")，
    # 必须显式带 playid=269(胜平负/让球) + date=当日 参数抓取。
    if date is None:
        date = time.strftime("%Y-%m-%d")
    html = _get(f"{JCZQ_URL}?playid=269&g=2&date={date}")
    matches = []
    # 比赛行：以编号"周五001"等开头的 tr
    pattern = re.compile(
        r'<tr[^>]*id="(?:match_\w+)"?[^>]*>(.*?)</tr>', re.S)
    # 用更通用的方式：找所有含 class="td-no" 的 tr
    tr_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
    for tr in tr_pattern.findall(html):
        if "td-no" not in tr:
            continue
        no_m = re.search(r"td-no[^>]*>\s*<a[^>]*>([^<]+)", tr)
        if not no_m:
            continue
        match_no = _strip_tags(no_m.group(1))
        if not re.match(r"周[一二三四五六日]\d{3}", match_no):
            continue

        def grab(pat, default=""):
            m = re.search(pat, tr)
            return _strip_tags(m.group(1)) if m else default

        # 联赛（含链接中的联赛 id；无 zuqiu 链接时用标题 title 兜底，如日联赛杯）
        lg_m = re.search(r'zuqiu-(\d+)/[^>]*>([^<]+)<', tr)
        if lg_m:
            league = lg_m.group(2)
            league_id = lg_m.group(1)
        else:
            ttl_m = re.search(r'td-evt[^>]*>[^<]*<a[^>]*title="([^"]+)"', tr) or \
                    re.search(r'td-evt[^>]*title="([^"]+)"[^>]*>', tr)
            league = ttl_m.group(1) if ttl_m else grab(r"td-evt[^>]*>(.*?)</td>")
            league_id = ""
        # 时间
        time_m = re.search(r"td-endtime[^>]*title=\"([^\"]+)\"", tr)
        match_time = time_m.group(1) if time_m else grab(r"td-endtime[^>]*>(.*?)</td>")
        # 主客队（liansai team id + 名称；无 liansai 链接时按 team-l/team-r 的 title 兜底）
        tm = re.findall(r'liansai\.500\.com/team/(\d+)/"[^>]*title="([^"]+)"', tr)
        if len(tm) >= 2:
            home_id, away_id, home, away = tm[0][0], tm[1][0], tm[0][1], tm[1][1]
        else:
            home_m = re.search(r'<a[^>]*class="team-l"[^>]*title="([^"]+)"', tr) or \
                     re.search(r'class="team-l"[^>]*title="([^"]+)"', tr)
            away_m = re.search(r'<a[^>]*class="team-r"[^>]*title="([^"]+)"', tr) or \
                     re.search(r'class="team-r"[^>]*title="([^"]+)"', tr)
            home = home_m.group(1) if home_m else ""
            away = away_m.group(1) if away_m else ""
            home_id = away_id = ""
        # 让球（竞彩官方让球数：itm-rangA2 为实际让球；
        #  正数=客队让球（主队受让+），负数=主队让球，空/0=无让球）
        rang_m = re.findall(r'itm-rangA(\d)"[^>]*>\s*([+-]?\d*)\s*<', tr)
        rang = "0"
        if rang_m:
            by_no = {no: v.strip() for no, v in rang_m}
            # 优先 A2（实际让球），回退 A1
            for no in ("2", "1"):
                if by_no.get(no):
                    rang = by_no[no]
                    break
        # SP 赔率（不让球胜平负 3 项）
        sps = re.findall(r'data-type="nspf" data-value="\d+" data-sp="([\d.]+)"', tr)
        sp_h, sp_d, sp_a = (float(x) for x in sps[:3]) if len(sps) >= 3 else (None, None, None)
        # 让球后 SP（让球胜平负 3 项）
        rsp = re.findall(r'data-type="spf" data-value="\d+" data-sp="([\d.]+)"', tr)
        rsp_h, rsp_d, rsp_a = (float(x) for x in rsp[:3]) if len(rsp) >= 3 else (None, None, None)
        # 比赛 id（欧赔分析页用，如 fenxi/ouzhi-1460771.shtml）
        oid = re.search(r"fenxi/ouzhi-(\d+)", tr)
        match_id = oid.group(1) if oid else ""

        matches.append({
            "no": match_no, "league": league, "league_id": league_id,
            "match_time": match_time, "home": home, "away": away,
            "home_id": home_id, "away_id": away_id, "rang": rang,
            "sp_h": sp_h, "sp_d": sp_d, "sp_a": sp_a,
            "range_sp_h": rsp_h, "range_sp_d": rsp_d, "range_sp_a": rsp_a,
            "match_id": match_id,
        })

    if use_cache and matches:
        cache.write_text(json.dumps(matches, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return matches


# ---------------------------------------------------------------------------
# 2. 球队近 10 场战绩
# ---------------------------------------------------------------------------
def fetch_team_form(team_id: str, use_cache: bool = True) -> dict:
    """
    抓取球队近 10 场比赛（含比分/赛果/盘路/大小），汇总成特征。

    Returns
    -------
    dict
        {
          "team": 球队中文名, "league": 联赛,
          "matches": [ {date, comp, home, away, fthg, ftag, result, cover} ],
          "form": { "gf_3":..,"ga_3":..,"pts_3":..,"gf_5":..,...,"cover_8":.. },
          "home_gf_3":.., "away_gf_3":..,   # 主/客场分列（近 3 场）
        }
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"team_{team_id}.json"
    if use_cache and cache.exists():
        age = time.time() - cache.stat().st_mtime
        if age < 24 * 3600:
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except Exception:
                pass

    html = _get(TEAM_URL.format(tid=team_id))
    # 球队名
    name_m = re.search(r"<title>([^<_]+).*?</title>", html) or \
             re.search(r'class="team-name"[^>]*>([^<]+)<', html)
    team_name = _strip_tags(name_m.group(1)) if name_m else team_id

    matches = []
    # 战绩行格式: 赛事 日期 主队 比分(半场) 客队 赛果 盘路 大小
    row_re = re.compile(
        r"<tr[^>]*>(.*?)</tr>", re.S)
    for tr in row_re.findall(html):
        tds = [_strip_tags(x) for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        tds = [t for t in tds if t]
        if len(tds) < 7:
            continue
        # 识别: [赛事, 日期, 主队, 比分, 客队, 赛果, 盘路, 大小, 分析]
        score_m = re.search(r"(\d+)\s*[:：]\s*(\d+)", tds[3])
        if not score_m:
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", tds[1]):
            continue
        try:
            comp, date, hm, away = tds[0], tds[1], tds[2], tds[4]
            fthg, ftag = int(score_m.group(1)), int(score_m.group(2))
            result = tds[5]  # 胜/平/负（主队视角）
            cover = tds[6] if len(tds) > 6 else "-"  # 赢/输/走盘（主队视角）
        except (ValueError, IndexError):
            continue
        matches.append({
            "date": date, "comp": comp, "home": hm, "away": away,
            "fthg": fthg, "ftag": ftag, "result": result, "cover": cover,
        })
    # 只保留最近 10 场（页面默认按时间倒序）
    matches = matches[:10]

    # ---- 汇总特征（主队视角：本队进球=主场时fthg, 客场时ftag） ----
    def team_stats(ms):
        recs = []
        for m in ms:
            is_home = (m["home"] == team_name)
            gf = m["fthg"] if is_home else m["ftag"]
            ga = m["ftag"] if is_home else m["fthg"]
            res = m["result"]
            pts = 3 if res == "胜" else (1 if res == "平" else 0)
            win = 1.0 if res == "胜" else 0.0
            cov = 1.0 if m["cover"] == "赢" else (0.0 if m["cover"] in ("输", "输半") else np.nan)
            recs.append({"gf": gf, "ga": ga, "pts": pts, "win": win, "cover": cov,
                         "is_home": is_home})
        return recs

    recs = team_stats(matches)
    form = {}
    for k in (3, 5, 8):
        sub = recs[:k]
        if not sub:
            form[f"gf_{k}"] = form[f"ga_{k}"] = form[f"pts_{k}"] = form[f"cover_{k}"] = np.nan
            continue
        w = np.array([0.8 ** (k - 1 - i) for i in range(k)])
        form[f"gf_{k}"] = float(np.average([r["gf"] for r in sub], weights=w))
        form[f"ga_{k}"] = float(np.average([r["ga"] for r in sub], weights=w))
        form[f"pts_{k}"] = float(np.average([r["pts"] for r in sub], weights=w))
        covs = np.array([r["cover"] for r in sub])
        valid = np.isfinite(covs)
        form[f"cover_{k}"] = (float(np.average(covs[valid], weights=w[valid]))
                              if valid.any() else np.nan)
    # 主客场分列（近 3 场）
    home_recs = [r for r in recs[:5] if r["is_home"]]
    away_recs = [r for r in recs[:5] if not r["is_home"]]
    form["home_gf_5"] = (float(np.mean([r["gf"] for r in home_recs])) if home_recs else np.nan)
    form["home_ga_5"] = (float(np.mean([r["ga"] for r in home_recs])) if home_recs else np.nan)
    form["away_gf_5"] = (float(np.mean([r["gf"] for r in away_recs])) if away_recs else np.nan)
    form["away_ga_5"] = (float(np.mean([r["ga"] for r in away_recs])) if away_recs else np.nan)
    # 近 10 场总量
    form["gf_10"] = float(np.mean([r["gf"] for r in recs])) if recs else np.nan
    form["ga_10"] = float(np.mean([r["ga"] for r in recs])) if recs else np.nan

    out = {"team": team_name, "matches": matches, "form": form}
    if use_cache:
        cache.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 3. 构建模型特征向量
# ---------------------------------------------------------------------------
def build_feature_vector(entry: dict, form_home: dict | None,
                         form_away: dict | None) -> dict:
    """
    用 500 数据构建特征 dict（与模型 feature_names 对齐，供预测）。

    规则：
    - 赔率特征：直接用竞彩 SP（去抽水）
    - 近期状态：用 500 近 10 场战绩（近 3/5/8 场指数加权）
    - 基础实力：近 10 场均值近似 90 天窗口；主客场分列补充
    - 无法获得的特征（ELO/交锋/真实历史窗口）：填中性/缺失值
    """
    f = {}
    # ---- 赔率特征（D） ----
    if all(v is not None for v in (entry["sp_h"], entry["sp_d"], entry["sp_a"])):
        sp = np.array([entry["sp_h"], entry["sp_d"], entry["sp_a"]], dtype=float)
        inv = 1.0 / sp
        p = inv / inv.sum()
        f["odds_h"], f["odds_d"], f["odds_a"] = entry["sp_h"], entry["sp_d"], entry["sp_a"]
        f["prob_h"], f["prob_d"], f["prob_a"] = float(p[0]), float(p[1]), float(p[2])
        f["log_odds_ratio"] = float(np.log(p[0] / p[2]))
    # 让球盘口（亚盘语义：负=主队让球，正=主队受让；与竞彩 rang 符号一致）
    try:
        f["ahh"] = float(entry["rang"])
    except (TypeError, ValueError):
        pass

    # ---- 近期状态（B）与基础实力（A） ----
    for side, form in (("home", form_home), ("away", form_away)):
        if form:
            fm = form["form"]
            f[f"{side}_gf_365d"] = fm.get("gf_10")
            f[f"{side}_ga_365d"] = fm.get("ga_10")
            f[f"{side}_sot_365d"] = np.nan
            f[f"{side}_gf_180d"] = fm.get("gf_8")
            f[f"{side}_ga_180d"] = fm.get("ga_8")
            f[f"{side}_sot_180d"] = np.nan
            f[f"{side}_gf_90d"] = fm.get("gf_5")
            f[f"{side}_ga_90d"] = fm.get("ga_5")
            f[f"{side}_sot_90d"] = np.nan
            for k in (3, 5, 8):
                f[f"{side}_pts_{k}"] = fm.get(f"pts_{k}")
                f[f"{side}_gf_{k}"] = fm.get(f"gf_{k}")
                f[f"{side}_ga_{k}"] = fm.get(f"ga_{k}")
                f[f"{side}_cover_{k}"] = fm.get(f"cover_{k}")
        else:
            for k in (3, 5, 8):
                f[f"{side}_pts_{k}"] = f[f"{side}_gf_{k}"] = \
                    f[f"{side}_ga_{k}"] = f[f"{side}_cover_{k}"] = np.nan
            for w in ("365d", "180d", "90d"):
                f[f"{side}_gf_{w}"] = f[f"{side}_ga_{w}"] = \
                    f[f"{side}_sot_{w}"] = np.nan
        # 主客场分列近 5 场
        if form:
            fm = form["form"]
            f[f"{side}_home_gf_5"] = fm.get("home_gf_5")
            f[f"{side}_home_ga_5"] = fm.get("home_ga_5")
            f[f"{side}_away_gf_5"] = fm.get("away_gf_5")
            f[f"{side}_away_ga_5"] = fm.get("away_ga_5")
        else:
            for col in ("home_gf_5", "home_ga_5", "away_gf_5", "away_ga_5"):
                f[f"{side}_{col}"] = np.nan

    # ---- 联赛（D2） ----
    league_code = LEAGUE_MAP.get(entry["league"])
    f["league_idx"] = LEAGUE_IDX.get(league_code, np.nan) if league_code else np.nan
    # 净攻防强度
    if np.isfinite(f.get("home_gf_365d", np.nan)) and np.isfinite(f.get("away_ga_365d", np.nan)):
        f["net_att_h"] = f["home_gf_365d"] - f["away_ga_365d"]
    else:
        f["net_att_h"] = np.nan
    if np.isfinite(f.get("away_gf_365d", np.nan)) and np.isfinite(f.get("home_ga_365d", np.nan)):
        f["net_att_a"] = f["away_gf_365d"] - f["home_ga_365d"]
    else:
        f["net_att_a"] = np.nan

    # ---- 缺失特征（ELO / 交锋 / 赛程 / 附加）----
    f["home_elo"] = 1500.0   # 中性 ELO（未知球队）
    f["away_elo"] = 1500.0
    f["elo_diff"] = 0.0
    f["h2h_home_win"] = f["h2h_draw"] = f["h2h_goals"] = f["h2h_home_goals"] = np.nan
    f["home_rest_days"] = f["away_rest_days"] = np.nan
    f["away_consec_away"] = 0.0
    f["home_week_double"] = np.nan
    return f


def to_model_vector(fdict: dict, feature_cols: list[str]) -> np.ndarray:
    """
    将 500 构建的特征 dict 对齐到模型特征列（NaN->-1，_missing->1）。
    """
    vec = []
    for col in feature_cols:
        v = fdict.get(col, np.nan)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            vec.append(1.0 if col.endswith("_missing") else -1.0)
        else:
            vec.append(float(v))
    return np.asarray(vec, dtype=float)


if __name__ == "__main__":
    # 自检：列出今天的比赛
    ms = fetch_matches()
    print(f"今天共 {len(ms)} 场竞彩比赛：")
    for m in ms[:15]:
        print(f"  {m['no']}  {m['league']}  {m['match_time']}  "
              f"{m['home']} vs {m['away']}  "
              f"(让{m['rang']}  SP {m['sp_h']}/{m['sp_d']}/{m['sp_a']})")
