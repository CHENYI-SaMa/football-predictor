# -*- coding: utf-8 -*-
"""generate_backfill_week.py —— 补齐上周(8/11~8/17)竞彩比赛文件
================================================================
对 7 天内每个比赛日：
1. 确保 预测_v3_{date}.json 存在且含 probs（缺失则抓赔率+模型推理生成/回填）
2. 生成 让球胜平负推荐_{date}.txt/.csv -> 桌面 竞彩预测/{date}/ 根目录
3. 数据 json 副本 -> 桌面 {date}/其他/
4. 有赛果则生成 赛果复盘V3_{date}.txt（调 check_v3_results.py，自动同步桌面根目录）
用法(系统python+venv依赖): python generate_backfill_week.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation import odds_baseline_probs          # noqa: E402
from src.features_v3 import FeatureBuilderV3            # noqa: E402
from src.jczq_fetcher import fetch_matches              # noqa: E402
from gui import format_handicap_txt, format_handicap_csv  # noqa: E402
import predict_v3 as P                                  # noqa: E402

DESK = Path("C:/Users/CHENYI/Desktop/竞彩预测")
VENV_PY = "C:/Users/CHENYI/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
DOWN_LEAGUES = ("欧协联", "欧会杯", "欧罗巴")


def ensure_pred(date: str) -> Path:
    dd = date.replace("-", "")
    path = BASE_DIR / "outputs" / f"预测_v3_{dd}.json"
    existing = path.exists()
    data = json.loads(path.read_text(encoding="utf-8")) if existing \
        else {"model": "models_v3", "date": date, "results": []}
    results = data["results"]
    model, meta, state = P.load_v3()
    cols = meta["feature_cols"]
    # 抓 date ~ date+2 三天的比赛合并（软件当时"可投注"规模 = json 快照全部场次；
    # 快照缺失 probs 的用相邻日期当前赔率按编号匹配补算；无 json 的日期用跨3天全集生成）
    import datetime as _dt
    base = _dt.date.fromisoformat(date)
    m_by_no = {}
    for i in range(3):
        d2 = (base + _dt.timedelta(days=i)).isoformat()
        try:
            for m in fetch_matches(use_cache=False, date=d2):
                m_by_no.setdefault(m.get("no"), m)
        except Exception as e:  # noqa: BLE001
            print(f"    [警告] {d2} 抓取失败: {e}")
    updated = 0
    for m in m_by_no.values():
        no = m.get("no")
        r = next((x for x in results if x.get("no") == no), None)
        if r is not None and r.get("probs"):
            continue
        if r is None and existing:
            continue  # 已有快照的日期不新增场次（保持软件当时规模）
        try:
            rsp = (m.get("range_sp_h"), m.get("range_sp_d"), m.get("range_sp_a"))
            odds = rsp if all(x is not None for x in rsp) \
                else (m.get("sp_h"), m.get("sp_d"), m.get("sp_a"))
            row = P.jczq_row(m, date, odds)
            fdict = FeatureBuilderV3.build_single(state, pd.Series(row))
            X = np.array([[fdict.get(c, -1.0) for c in cols]], dtype=float)
            pr = P.range_probs(model, X, int(m.get("rang", 0) or 0))
            pred_i = int(np.argmax(pr))
            conf = float(pr[pred_i])
            rp_second = int(np.argsort(pr)[-2]) == 1
            if all(x is not None for x in rsp):
                mo = odds_baseline_probs([rsp[0]], [rsp[1]], [rsp[2]])[0]
                mkt_i = int(np.argmax(mo))
                val = float(conf - mo[pred_i])
                mkt_probs = [round(float(x), 3) for x in mo]
            else:
                mkt_i, val, mkt_probs = -1, 0.0, None
            down = bool(m.get("league") in DOWN_LEAGUES)
            rec = {
                "no": no, "league": m.get("league"),
                "home": m.get("home"), "away": m.get("away"),
                "rang": int(m.get("rang", 0) or 0),
                "pred_range": ["让胜", "让平", "让负"][pred_i],
                "conf": round(conf, 3), "rp_second": bool(rp_second),
                "probs": [round(float(x), 3) for x in pr],
                "mkt_range": (["让胜", "让平", "让负"][mkt_i] if mkt_i >= 0 else "-"),
                "mkt_conf": round(float(mo[mkt_i]), 3) if mkt_i >= 0 else 0.0,
                "mkt_probs": mkt_probs, "val": round(val, 3),
                "pick": bool(not down and conf >= 0.60 and val >= 0.10),
                "downgrade": down, "injuries": None, "model": "v3-60.2%",
            }
            if r is not None:
                for k, v in rec.items():
                    r[k] = v
            else:
                results.append(rec)
            updated += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [跳过] {no}: {e}")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  json: {'补算/生成' if existing else '新生成'} {updated} 场（预测 {len(results)} 场）-> {path.name}")
    return path


def gen_recommend(date: str) -> None:
    dd = date.replace("-", "")
    path = BASE_DIR / "outputs" / f"预测_v3_{dd}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rs = data["results"]
    matches = [{"no": r["no"], "home": r["home"], "away": r["away"],
                "league": r.get("league", "")} for r in rs]
    v3 = {r["no"]: r for r in rs}
    txt = format_handicap_txt(matches, v3, dd)
    csv = format_handicap_csv(matches, v3)
    folder = DESK / dd
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"让球胜平负推荐_{dd}.txt").write_text(txt, encoding="utf-8")
    (folder / f"让球胜平负推荐_{dd}.csv").write_text(csv, encoding="utf-8")
    (folder / "其他").mkdir(exist_ok=True)
    shutil.copy(path, folder / "其他" / f"竞彩预测数据_{dd}.json")
    n_pick = sum(1 for r in rs if r.get("pick"))
    print(f"  推荐: {n_pick} 场 pick -> {folder}")


def gen_review(date: str) -> None:
    r = subprocess.run([VENV_PY, str(BASE_DIR / "check_v3_results.py"),
                        "--date", date], capture_output=True, text=True,
                       encoding="utf-8")
    out = (r.stdout or "").strip()
    last = out.splitlines()[-1] if out else "(无输出)"
    print(f"  复盘: {last}")


def main():
    for date in ["2026-08-11", "2026-08-12", "2026-08-13",
                 "2026-08-14", "2026-08-15", "2026-08-16", "2026-08-17"]:
        print(f"== {date} ==")
        ensure_pred(date)
        gen_recommend(date)
        gen_review(date)
    print("\n全部完成")


if __name__ == "__main__":
    main()
