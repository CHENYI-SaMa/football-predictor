# -*- coding: utf-8 -*-
"""fetch_injuries.py —— api-football 伤停/停赛抓取（curl 通道，走系统代理）
================================================================
免费层 100 请求/天。预算分配：
  - fixtures(当天列表) 1 请求
  - injuries(伤停+停赛) 逐场 1 请求（61 场 → 62 请求，预算内）
  - lineups(首发) 逐场 1 请求（仅 --lineups 模式，对高价值场次）

队名匹配：竞彩中文队名 → TEAM_NAME_MAP 英文 → api 英文队名（归一化模糊匹配）。
用法: python fetch_injuries.py [--date 2026-08-15] [--limit 60] [--lineups]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data" / "injuries"
API = "https://v3.football.api-sports.io"
KEY = "ba9bac2584ebe1375572e450c68e08e9"


def api_get(path: str, params: dict | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        q = "&".join(f"{k}={v}" for k, v in params.items())
        url += f"?{q}"
    for attempt in range(3):
        r = subprocess.run(
            ["curl", "-s", "--max-time", "40", "-H", f"x-apisports-key: {KEY}", url],
            capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0 and r.stdout.strip():
            try:
                return json.loads(r.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"curl 失败: {r.stderr[:100]}")


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def match_fixture(cn_home: str, cn_away: str, fixtures: list[dict]) -> dict | None:
    """竞彩中文队名 → 匹配 api fixture（队名归一化精确/包含匹配）。"""
    import json as _j
    sys.path.insert(0, str(BASE_DIR))
    from src.jczq_fetcher import TEAM_NAME_MAP
    # 补充映射表（竞彩常客联赛中文→英文），TEAM_NAME_MAP 兜底
    extra = {}
    mp = BASE_DIR / "data" / "team_map_cn_en.json"
    if mp.exists():
        extra = _j.loads(mp.read_text(encoding="utf-8"))
    en_h = norm(extra.get(cn_home) or TEAM_NAME_MAP.get(cn_home, cn_home))
    en_a = norm(extra.get(cn_away) or TEAM_NAME_MAP.get(cn_away, cn_away))
    best, best_score = None, 0.0
    for fx in fixtures:
        fh = norm(fx["teams"]["home"]["name"])
        fa = norm(fx["teams"]["away"]["name"])
        # 精确匹配（主客双队名完全一致）→ 最高优先级，直接返回
        if fh == en_h and fa == en_a:
            return fx
        # 子串匹配：短队名（<4 字符，如 AIK/Brann）只接受精确，防误配
        def sub_ok(en, f, is_short):
            return (not is_short) and bool(en) and (en in f or f in en)
        sc_h = sub_ok(en_h, fh, 0 < len(en_h) < 4)
        sc_a = sub_ok(en_a, fa, 0 < len(en_a) < 4)
        # 只接受主客双队都命中的候选（宁可漏掉，不可贴错）
        score = 2.0 if (sc_h and sc_a) else 0.0
        if score > best_score:
            best_score = score
            best = fx
    return best if best_score >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--limit", type=int, default=200, help="最多处理场次（控制请求量）")
    ap.add_argument("--lineups", action="store_true", help="抓首发(lineups)而非伤停")
    args = ap.parse_args()

    sys.path.insert(0, str(BASE_DIR))
    from src.jczq_fetcher import fetch_matches

    # 1. 竞彩比赛
    matches = fetch_matches(use_cache=False)
    print(f"竞彩场次: {len(matches)}")
    # 2. 当天 + 后两天 fixtures（欧洲晚场在北京时区跨天）
    import datetime
    fixtures = []
    for off in (0, 1, 2):
        day = (datetime.date.fromisoformat(args.date)
               + datetime.timedelta(days=off)).isoformat()
        fx = api_get("/fixtures", {"date": day, "timezone": "Asia/Shanghai"})
        fixtures += fx.get("response", [])
        print(f"fixtures {day}: +{len(fx.get('response', []))} 场")
    print(f"api fixtures 合计: {len(fixtures)} 场")

    # 3. 匹配
    linked = []
    for m in matches:
        f = match_fixture(m.get("home", ""), m.get("away", ""), fixtures)
        if f:
            linked.append((m, f))
    print(f"匹配成功: {len(linked)}/{len(matches)}")
    if not linked:
        print("无匹配，检查队名映射")
        return

    # 4. 逐场抓伤停/首发
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, fails = [], []
    for i, (m, f) in enumerate(linked[:args.limit]):
        fid = f["fixture"]["id"]
        try:
            if args.lineups:
                d = api_get("/fixtures/lineups", {"fixture": fid})
                payload = [{"team": x["team"]["name"],
                            "formation": x.get("formation", ""),
                            "startXI": [p["player"]["name"] for p in x.get("startXI", [])],
                            "subs": [p["player"]["name"] for p in x.get("substitutes", [])]}
                           for x in d.get("response", [])]
            else:
                d = api_get("/injuries", {"fixture": fid})
                payload = [{"team": x["team"]["name"],
                            "player": x["player"]["name"],
                            "type": x["player"].get("type", ""),
                            "reason": x["player"].get("reason", "")}
                           for x in d.get("response", [])]
            results.append({
                "no": m.get("no"), "fixture_id": fid,
                "home": m.get("home"), "away": m.get("away"),
                "api_home": f["teams"]["home"]["name"],
                "api_away": f["teams"]["away"]["name"],
                "data": payload,
            })
            tag = "首发" if args.lineups else "伤停"
            print(f"  [{i+1}/{len(linked[:args.limit])}] {m.get('no')} "
                  f"{m.get('home')}vs{m.get('away')} {tag}{len(payload)}条")
        except Exception as e:
            fails.append(f"{m.get('no')}: {e}")
        time.sleep(0.8)

    out = OUT_DIR / f"{'lineups' if args.lineups else 'injuries'}_{args.date}.json"
    out.write_text(json.dumps({"date": args.date, "results": results, "fails": fails},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 已保存 {out}（{len(results)} 场，失败 {len(fails)}）")
    if fails:
        print("失败:", "; ".join(fails[:5]))


if __name__ == "__main__":
    main()
