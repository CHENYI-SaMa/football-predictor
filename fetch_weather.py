# -*- coding: utf-8 -*-
"""fetch_weather.py —— 比赛天气抓取（open-meteo，免费无key）
================================================================
链路：500 比赛主队（中文）→ TEAM_CITY 映射 → 城市 geocoding 坐标
      → open-meteo hourly 天气（温度/降水概率/风速/天气码）→ 存 data/weather/

开赛时间：500 的 match_time 是投注截止时间，取截止-2h 近似开赛（误差对天气可忽略）。
城市坐标缓存：data/weather/city_coords.json（geocoding 结果复用，不重复请求）。
用法: python fetch_weather.py [--date 2026-08-14]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data" / "weather"
COORD_CACHE = OUT_DIR / "city_coords.json"

HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/126.0.0.0 Safari/537.36")}

# 球队（中文主队名）→ 城市（英文，geocoding 用）。逐步扩充，抓不到的城市会标记。
TEAM_CITY = {
    # 荷甲/荷乙
    "SBV精英": "Rotterdam", "乌德勒支": "Utrecht", "特温特": "Enschede",
    "费耶诺德": "Rotterdam", "阿贾克斯": "Amsterdam", "PSV埃因霍温": "Eindhoven",
    "威廉二世": "Tilburg", "海牙": "The Hague", "瓦尔韦克": "Waalwijk",
    "福图纳锡塔德": "Sittard", "赫拉克勒斯": "Almelo", "特尔斯达": "Velsen",
    # 德甲/德乙
    "不伦瑞克": "Braunschweig", "基尔": "Kiel", "柏林赫塔": "Berlin",
    "拜仁慕尼黑": "Munich", "多特蒙德": "Dortmund", "勒沃库森": "Leverkusen",
    "圣保利": "Hamburg", "海登海姆": "Heidenheim", "奥格斯堡": "Augsburg",
    "斯图加特": "Stuttgart", "弗赖堡": "Freiburg", "沃尔夫斯堡": "Wolfsburg",
    "门兴格拉德巴赫": "Moenchengladbach", "美因茨": "Mainz", "法兰克福": "Frankfurt",
    # 英超
    "阿森纳": "London", "曼城": "Manchester", "曼联": "Manchester",
    "利物浦": "Liverpool", "切尔西": "London", "热刺": "London",
    "纽卡斯尔": "Newcastle", "维拉": "Birmingham", "狼队": "Wolverhampton",
    "伍尔弗汉普顿": "Wolverhampton", "西汉姆联": "London", "布莱顿": "Brighton",
    # 西甲
    "塞维利亚": "Seville", "西班牙人": "Barcelona", "阿拉维斯": "Vitoria-Gasteiz",
    "桑坦德竞技": "Santander", "皇家马德里": "Madrid", "巴塞罗那": "Barcelona",
    "马德里竞技": "Madrid", "皇家社会": "San Sebastian", "毕尔巴鄂竞技": "Bilbao",
    "瓦伦西亚": "Valencia", "比利亚雷亚尔": "Villarreal", "贝蒂斯": "Seville",
    # 法甲/法乙
    "兰斯": "Reims", "圣埃蒂安": "Saint-Etienne", "巴黎圣日耳曼": "Paris",
    "马赛": "Marseille", "里昂": "Lyon", "摩纳哥": "Monaco",
    "里尔": "Lille", "尼斯": "Nice", "阿纳西": "Annecy",
    # 意甲
    "国际米兰": "Milan", "AC米兰": "Milan", "尤文图斯": "Turin",
    "罗马": "Rome", "那不勒斯": "Naples", "亚特兰大": "Bergamo",
    # 日职/日乙
    "东京绿茵": "Tokyo", "浦和红钻": "Saitama", "神户胜利船": "Kobe",
    "鹿岛鹿角": "Kashima", "横滨水手": "Yokohama", "川崎前锋": "Kawasaki",
    "大阪钢巴": "Osaka", "广岛三箭": "Hiroshima", "名古屋鲸八": "Nagoya",
    "札幌冈萨多": "Sapporo", "鸟栖砂岩": "Tosu", "湘南海洋": "Hiratsuka",
    "德岛漩涡": "Tokushima", "秋田蓝色闪电": "Akita", "新泻天鹅": "Niigata",
    # 韩K
    "仁川联": "Incheon", "光州FC": "Gwangju", "首尔FC": "Seoul",
    "富川FC": "Bucheon", "全北现代": "Jeonju", "蔚山现代": "Ulsan",
    "浦项制铁": "Pohang", "水原三星": "Suwon", "大邱FC": "Daegu",
    # 北欧
    "罗森博格": "Trondheim", "埃尔夫斯堡": "Boras", "瓦萨": "Vaasa",
    "马尔默": "Malmo", "哥德堡": "Gothenburg", "赫根": "Gothenburg",
    "佐加顿斯": "Stockholm", "AIK索尔纳": "Stockholm", "布兰": "Bergen",
    "莫尔德": "Molde", "博多格林特": "Bodo", "维京": "Stavanger",
    # 葡超/葡甲
    "里斯本竞技": "Lisbon", "葡萄牙国民": "Funchal", "维塞乌": "Viseu",
    "里奥阿维": "Vila do Conde", "阿尔维卡": "Alverca", "阿罗卡": "Arouca",
    "波尔图": "Porto", "本菲卡": "Lisbon", "布拉加": "Braga",
    # 沙特
    "利雅得新月": "Riyadh", "达曼协定": "Dammam", "利雅得胜利": "Riyadh",
    "吉达联合": "Jeddah", "吉达国民": "Jeddah",
    "新未来SC": "Neom", "迈季迈阿宽广": "Al Majma'ah", "费哈": "Al Majma'ah",
}


def get(url: str, timeout: int = 20) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _load_coords() -> dict:
    if COORD_CACHE.exists():
        return json.loads(COORD_CACHE.read_text(encoding="utf-8"))
    return {}


def _save_coords(coords: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COORD_CACHE.write_text(json.dumps(coords, ensure_ascii=False, indent=1),
                           encoding="utf-8")


def city_coords(city: str, coords: dict) -> tuple | None:
    """城市 → (lat, lon, elevation_m)，带缓存。"""
    if city in coords:
        c = coords[city]
        if len(c) >= 2:
            if len(c) == 2:      # 旧缓存无海拔，补 0 并回写
                c = (c[0], c[1], 0.0)
                coords[city] = list(c)
            return tuple(c)
        return None
    try:
        d = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "format": "json"},
            headers=HEADERS, timeout=15).json()
        r = d.get("results") or []
        if r:
            c = (r[0]["latitude"], r[0]["longitude"], r[0].get("elevation", 0.0))
            coords[city] = list(c)
            return c
    except Exception:
        pass
    return None


def fetch_weather(lat: float, lon: float, hour_local: str) -> dict | None:
    """按坐标查 hourly 天气，返回指定本地小时的 温度/降水概率/风速/天气码。"""
    try:
        d = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={
                             "latitude": lat, "longitude": lon,
                             "hourly": "temperature_2m,precipitation_probability,"
                                       "wind_speed_10m,weather_code",
                             "timezone": "auto", "forecast_days": 3,
                         }, headers=HEADERS, timeout=20).json()
        h = d.get("hourly") or {}
        times = h.get("time") or []
        for i, t in enumerate(times):
            if t.startswith(hour_local):
                return {
                    "time": t,
                    "temp_c": h["temperature_2m"][i],
                    "precip_prob": h["precipitation_probability"][i],
                    "wind_kmh": h["wind_speed_10m"][i],
                    "weather_code": h["weather_code"][i],
                }
    except Exception:
        pass
    return None


# 场馆档案：球队 → 场馆信息（待核实填充，不编造）。
#   venue_type: "ac"=空调/封闭（天气特征无效）| "open"=全开放 | None=未知(默认按开放处理)
#   stadium: 球场名 | pitch: "natural"=天然草|"artificial"=人工草
TEAM_VENUE: dict = {}


def parse_kickoff(match_time: str) -> str:
    """'08-14 18:00截止' → 开赛近似 '2026-08-14T16'（截止-2h）。"""
    m = re.match(r"(\d{2})-(\d{2}) (\d{2}):(\d{2})", match_time or "")
    if not m:
        return ""
    mm, dd, hh, mi = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    hour = hh - 2
    day = dd
    if hour < 0:
        hour += 24
        day -= 1
    year = time.strftime("%Y")
    return f"{year}-{mm:02d}-{day:02d}T{hour:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    args = ap.parse_args()

    sys.path.insert(0, str(BASE_DIR))
    from src.jczq_fetcher import fetch_matches

    matches = fetch_matches(use_cache=False)
    if not matches:
        print("未获取到比赛列表")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coords = _load_coords()
    results, missing = [], []
    for m in matches:
        home = m.get("home") or ""
        city = TEAM_CITY.get(home)
        if not city:
            missing.append(f"{home}(无城市映射)")
            continue
        c = city_coords(city, coords)
        if not c:
            missing.append(f"{home}({city} 坐标解析失败)")
            continue
        hour = parse_kickoff(m.get("match_time"))
        if not hour:
            missing.append(f"{home}(开赛时间解析失败:{m.get('match_time')})")
            continue
        w = fetch_weather(c[0], c[1], hour)
        v = TEAM_VENUE.get(home) or {}
        results.append({
            "match_id": m.get("match_id"), "no": m.get("no"),
            "home": home, "away": m.get("away"), "city": city,
            "lat": c[0], "lon": c[1], "elevation_m": c[2],
            "venue_type": v.get("venue_type"),   # ac/open/None(未知)
            "stadium": v.get("stadium"), "pitch": v.get("pitch"),
            "kickoff_approx": hour, "weather": w,
        })
        print(f"  {m.get('no')} {home} vs {m.get('away')} [{city}] "
              + (f"{w['temp_c']}°C 降水{w['precip_prob']}% "
                 f"风{w['wind_kmh']}km/h 码{w['weather_code']}"
                 + (f" 海拔{c[2]:.0f}m" if c[2] else "")
                 + (f" [空调]" if v.get("venue_type") == "ac" else "")
                 if w else "天气获取失败"))
        time.sleep(0.4)

    _save_coords(coords)
    out = OUT_DIR / f"{args.date}.json"
    out.write_text(json.dumps({
        "date": args.date, "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "matches": results, "missing": missing,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 已保存 {out}（{len(results)} 场有天气，{len(missing)} 场缺失）")
    if missing:
        print("缺失:", "; ".join(missing[:15]))


if __name__ == "__main__":
    main()
