# -*- coding: utf-8 -*-
"""阶段A-1: 队名映射构建
从 jczq_history.csv 提取中文队名（五大联赛），
与 football_history + understat 的英文队名做模糊映射，
输出 team_mapping.csv 与未匹配列表（暂停等人工确认）。
"""
import sys, json, glob
import pandas as pd
from fuzzywuzzy import process

sys.path.insert(0, ".")
from src.jczq_fetcher import TEAM_NAME_MAP

BIG5 = {"英超", "西甲", "德甲", "意甲", "法甲"}

# ---- 1. 中文队名（五大联赛） ----
d = pd.read_csv("data/jczq_history.csv")
big5_mask = d["league"].isin(BIG5)
cn_teams = sorted(set(d.loc[big5_mask, "home_team"]) | set(d.loc[big5_mask, "away_team"]))
print(f"需映射中文队名: {len(cn_teams)}")

# ---- 2. 英文队名全集 ----
en_teams = set()
for f in glob.glob("data/football_history/*.csv"):
    try:
        x = pd.read_csv(f)
        en_teams |= set(x["HomeTeam"].dropna()) | set(x["AwayTeam"].dropna())
    except Exception:
        pass
for f in glob.glob("data/understat/*.json"):
    try:
        j = json.load(open(f, encoding="utf-8"))
        for t in j.get("teams", {}).values():
            if isinstance(t, dict) and t.get("title"):
                en_teams.add(t["title"])
        for dt in j.get("dates", []):
            en_teams.add(dt["h"]["title"]); en_teams.add(dt["a"]["title"])
    except Exception:
        pass
en_teams = sorted(en_teams)
print(f"英文队名全集: {len(en_teams)}")

# ---- 3. 合并映射 ----
mapping = dict(TEAM_NAME_MAP)
ALIAS = {
    "国米": "Inter", "国际米兰": "Inter", "尤文": "Juventus", "尤文图斯": "Juventus",
    "拜仁": "Bayern Munich", "巴萨": "Barcelona", "皇马": "Real Madrid",
    "马竞": "Atletico Madrid", "热刺": "Tottenham", "勒沃库森": "Leverkusen",
    "多特": "Dortmund", "门兴": "Monchengladbach", "那不勒斯": "Napoli",
    "莱斯特城": "Leicester", "狼队": "Wolves", "纽卡斯尔联": "Newcastle",
    "西汉姆联": "West Ham", "布莱顿": "Brighton", "水晶宫": "Crystal Palace",
    "伯恩茅斯": "Bournemouth", "富勒姆": "Fulham", "埃弗顿": "Everton",
    "阿斯顿维拉": "Aston Villa", "诺丁汉森林": "Nottingham Forest",
    "南安普顿": "Southampton", "布伦特福德": "Brentford", "伯恩利": "Burnley",
    "利兹联": "Leeds", "谢菲联": "Sheffield United", "伊普斯维奇": "Ipswich",
    "比利亚雷亚尔": "Villarreal", "皇家社会": "Real Sociedad", "贝蒂斯": "Betis",
    "毕尔巴鄂": "Athletic Club", "塞维利亚": "Sevilla", "瓦伦西亚": "Valencia",
    "马略卡": "Mallorca", "赫罗纳": "Girona", "奥萨苏纳": "Osasuna",
    "塞尔塔": "Celta Vigo", "赫塔菲": "Getafe", "西班牙人": "Espanyol",
    "拉斯帕尔马斯": "Las Palmas", "阿拉维斯": "Alaves", "莱加内斯": "Leganes",
    "巴列卡诺": "Rayo Vallecano", "加的斯": "Cadiz", "埃尔切": "Elche",
    "巴拉多利德": "Valladolid", "阿尔梅里亚": "Almeria", "皇家奥维耶多": "Oviedo",
    "莱万特": "Levante", "格拉纳达": "Granada",
    "马赛": "Marseille", "里昂": "Lyon", "摩纳哥": "Monaco", "里尔": "Lille",
    "尼斯": "Nice", "雷恩": "Rennes", "朗斯": "Lens", "斯特拉斯堡": "Strasbourg",
    "蒙彼利埃": "Montpellier", "南特": "Nantes", "图卢兹": "Toulouse",
    "布雷斯特": "Brest", "昂热": "Angers", "兰斯": "Reims", "克莱蒙": "Clermont",
    "勒阿弗尔": "Le Havre", "欧塞尔": "Auxerre", "洛里昂": "Lorient",
    "圣埃蒂安": "Saint-Etienne", "梅斯": "Metz", "特鲁瓦": "Troyes",
    "阿雅克肖": "Ajaccio", "圣保利": "St. Pauli", "汉堡": "Hamburg",
    "法兰克福": "Eintracht Frankfurt", "斯图加特": "Stuttgart", "柏林联合": "Union Berlin",
    "沃尔夫斯堡": "Wolfsburg", "霍芬海姆": "Hoffenheim", "美因茨": "Mainz",
    "弗赖堡": "Freiburg", "奥格斯堡": "Augsburg", "科隆": "Cologne",
    "沙尔克04": "Schalke 04", "波鸿": "Bochum", "云达不莱梅": "Werder Bremen",
    "海登海姆": "Heidenheim", "柏林赫塔": "Hertha Berlin",
    "科莫": "Como", "威尼斯": "Venezia", "帕尔马": "Parma", "卡利亚里": "Cagliari",
    "恩波利": "Empoli", "莱切": "Lecce", "萨索洛": "Sassuolo", "斯佩齐亚": "Spezia",
    "桑普多利亚": "Sampdoria", "热那亚": "Genoa", "维罗纳": "Verona",
    "都灵": "Torino", "乌迪内斯": "Udinese", "萨勒尼塔纳": "Salernitana",
    "佛罗伦萨": "Fiorentina", "博洛尼亚": "Bologna", "罗马": "Roma",
    "拉齐奥": "Lazio", "亚特兰大": "Atalanta", "克雷莫纳": "Cremonese",
    "蒙扎": "Monza", "巴黎圣日耳曼": "PSG", "巴黎FC": "Paris FC",
    "桑德兰": "Sunderland", "比萨": "Pisa",
    "荷尔斯泰因": "Holstein Kiel", "艾禾斯堡": "Elversberg",
}
mapping.update(ALIAS)

# ---- 4. 分类 ----
direct, fuzzy_ok, unmatched = [], [], []
for cn in cn_teams:
    if cn in mapping:
        direct.append((cn, mapping[cn], "现有映射"))
        continue
    best = process.extractOne(cn, en_teams, score_cutoff=85)
    if best:
        fuzzy_ok.append((cn, best[0], f"模糊匹配{best[1]}"))
    else:
        unmatched.append(cn)

print(f"直接命中: {len(direct)}  模糊匹配: {len(fuzzy_ok)}  未匹配: {len(unmatched)}")
print("\n=== 模糊匹配结果 ===")
for x in fuzzy_ok:
    print(" ", x)
print("\n=== 未匹配（需人工确认）===")
for x in unmatched:
    print(" ", x)

# ---- 5. 输出映射表 ----
rows = [(cn, en, src) for cn, en, src in direct] + \
       [(cn, en, src) for cn, en, src in fuzzy_ok]
pdf = pd.DataFrame(rows, columns=["中文名", "英文名", "来源"])
pdf.to_csv("team_mapping.csv", index=False, encoding="utf-8-sig")
print(f"\n→ team_mapping.csv 已写入 {len(pdf)} 条")
