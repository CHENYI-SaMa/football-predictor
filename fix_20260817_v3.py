# -*- coding: utf-8 -*-
"""fix_20260817_v3.py —— 补正 8/17 的 V3 让球预测（旧版本 val 方向反了）。
=================================================================
只修正 val / mkt_probs / mkt_range / mkt_conf / pick，其余字段（模型 probs、
injuries 等）原样保留；随后生成桌面 竞彩预测/20260817/让球胜平负推荐/ 文件。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.jczq_fetcher import fetch_matches          # noqa: E402
from src.evaluation import odds_baseline_probs       # noqa: E402

DATE = "2026-08-17"
D = DATE.replace("-", "")
RMAP = {"让胜": 0, "让平": 1, "让负": 2}
PRED_LABELS = ["让胜", "让平", "让负"]
MIN_CONF, MIN_VAL = 0.60, 0.10

# ---------------------------------------------------------------------------
# 1. 修正 预测_v3_20260817.json
# ---------------------------------------------------------------------------
pred_path = BASE / "outputs" / f"预测_v3_{D}.json"
data = json.loads(pred_path.read_text(encoding="utf-8"))
results = data["results"]

# 抓取 8/17 让球 SP（带抽水），按编号索引
matches = fetch_matches(use_cache=False, date=DATE)
odds_map = {}
for m in matches:
    rsp = (m.get("range_sp_h"), m.get("range_sp_d"), m.get("range_sp_a"))
    if all(x is not None for x in rsp):
        odds_map[m["no"]] = (float(rsp[0]), float(rsp[1]), float(rsp[2]))

fixed = []
for r in results:
    no = r["no"]
    rsp = odds_map.get(no)
    if not rsp:
        fixed.append(r)
        continue
    mo = odds_baseline_probs([rsp[0]], [rsp[1]], [rsp[2]])[0]
    pred_i = RMAP[r["pred_range"]]
    mkt_i = int(np.argmax(mo))
    mkt_label = PRED_LABELS[mkt_i]
    mkt_conf = float(mo[mkt_i])
    val = float(r["conf"] - mo[pred_i])
    mkt_probs = [round(float(x), 3) for x in mo]
    r["mkt_range"] = mkt_label
    r["mkt_conf"] = round(mkt_conf, 3)
    r["mkt_probs"] = mkt_probs
    r["val"] = round(val, 3)
    r["pick"] = bool(not r.get("downgrade") and r["conf"] >= MIN_CONF
                     and val >= MIN_VAL)
    fixed.append(r)

data["results"] = fixed
pred_path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                     encoding="utf-8")
print(f"→ 已补正并保存 {pred_path}")

# ---------------------------------------------------------------------------
# 2. 复用 gui.py 的排版函数（V3 专项，复制自 gui.py 以保证一致）
# ---------------------------------------------------------------------------
def _hl_rang(rang):
    rang = int(rang or 0)
    return ("主让" + str(-rang) if rang < 0 else
            "客让" + str(rang) if rang > 0 else "平手")


def format_handicap_txt(matches, v3_map, d):
    mm = f"{int(d[4:6])}月{int(d[6:8])}日"
    DAY_ORDER = {"周五": 0, "周六": 1, "周日": 2, "周一": 3,
                 "周二": 4, "周三": 5, "周四": 6}

    def day_key(x):
        return DAY_ORDER.get((x.get("no", "") or "")[:2], 9)

    n_v3 = sum(1 for m in matches
               if v3_map.get(m.get("no")) and v3_map[m.get("no")].get("probs"))
    L = [f"{mm} 竞彩让球推荐（V3 让球模型）",
         f"共{n_v3}场 | ★=置信≥80%", ""]
    cur = None
    for m in sorted(matches, key=lambda x: (day_key(x), x.get("no", ""))):
        no = m.get("no")
        v = v3_map.get(no)
        if not (v and v.get("probs")):
            continue
        p = v["probs"]
        pred_label = v["pred_range"]
        conf = v["conf"]
        star = "★" if conf >= 0.80 else ""
        rng = _hl_rang(v.get("rang", 0))
        main_pick = pred_label + ("/让平" if v.get("rp_second") else "")
        block = [
            f"{no} {m.get('home')}vs{m.get('away')} [{rng}]",
            f"  {main_pick} {conf:.0%}{star}",
            f"  让球三项 RS{p[0]:.0%} RP{p[1]:.0%} RF{p[2]:.0%}",
        ]
        if v.get("mkt_range"):
            flag = " ⚠与模型反向" if v["mkt_range"] != pred_label else ""
            vline = f"；价值差 {v['val']:+.0%}" if v.get("val") is not None else ""
            block.append(f"  市场：{v['mkt_range']} {v['mkt_conf']:.0%}{flag}{vline}")
        if v.get("downgrade"):
            block.append("  ⚠ 低可信联赛（杯赛特征缺失），本场不主推")
        inj = v.get("injuries")
        if inj and inj.get("n_total"):
            teams = " ".join(f"{t['team']}{t['n']}伤" for t in inj["teams"])
            block.append(f"  伤停：{inj['n_total']}条 | {teams}")
        if cur != day_key(m):
            cur = day_key(m)
            L.append("")
            L.append(f"━━━ {m.get('no', '')[:2]} ━━━")
        L.append("\n".join(block))
    return "\n".join(L).rstrip() + "\n"


def format_handicap_csv(matches, v3_map):
    rows = ["编号,联赛,主队,客队,让球,主推,置信,RS让胜,RP让平,RF让负,"
            "市场主推,市场主推概率,市场同方向概率,价值差,低可信,来源"]
    for m in sorted(matches, key=lambda x: x.get("no", "")):
        no = m.get("no")
        v = v3_map.get(no)
        if not (v and v.get("probs")):
            continue
        p = v["probs"]
        main = v["pred_range"] + ("/让平" if v.get("rp_second") else "")
        conf = f"{v['conf']:.0%}"
        rng = _hl_rang(v.get("rang", 0))
        pred_i = RMAP[v["pred_range"]]
        mp = v.get("mkt_probs")
        same_p = f"{mp[pred_i]:.0%}" if mp else ""
        mkt = v.get("mkt_range", "")
        mktc = f"{v['mkt_conf']:.0%}" if v.get("mkt_range") else ""
        val = f"{v['val']:+.0%}" if v.get("val") is not None else ""
        down = "是" if v.get("downgrade") else "否"
        rows.append(",".join([no, m.get("league", ""), m.get("home", ""),
                              m.get("away", ""), rng, main, conf,
                              f"{p[0]:.0%}", f"{p[1]:.0%}", f"{p[2]:.0%}",
                              mkt, mktc, same_p, val, down, "V3"]))
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# 3. 生成桌面文件
# ---------------------------------------------------------------------------
matches = results  # 已有 no/league/home/away/rang
v3_map = {r["no"]: r for r in fixed}

desk_root = Path("C:/Users/CHENYI/Desktop/竞彩预测") / D / "让球胜平负推荐"
desk_root.mkdir(parents=True, exist_ok=True)

txt = format_handicap_txt(matches, v3_map, D)
csv = format_handicap_csv(matches, v3_map)
(desk_root / f"让球胜平负推荐_{D}.txt").write_text(txt, encoding="utf-8")
(desk_root / f"让球胜平负推荐_{D}.csv").write_text(csv, encoding="utf-8")
print(f"→ 已生成 {desk_root / ('让球胜平负推荐_' + D + '.txt')}")
print("---- txt 预览 ----")
print(txt)

# 4. V3 数据 dump（与桌面其它日期的 竞彩预测数据_{d}.json 对齐，V3 专项）
dump = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "models_v3（让球胜平负 60.2%）",
    "date": DATE,
    "results": [
        {
            "编号": r["no"], "赛事": r["league"], "主队": r["home"],
            "客队": r["away"], "让球": _hl_rang(r["rang"]) + "球",
            "主推让球胜平负": r["pred_range"] + ("/让平" if r.get("rp_second") else ""),
            "置信": r["conf"], "RS让胜": r["probs"][0],
            "RP让平": r["probs"][1], "RF让负": r["probs"][2],
            "市场主推": r["mkt_range"], "市场主推概率": r["mkt_conf"],
            "市场同方向概率": (r["mkt_probs"][RMAP[r["pred_range"]]]
                              if r.get("mkt_probs") else None),
            "价值差": r["val"], "是否入选": r["pick"],
        } for r in fixed
    ],
}
desk_data = Path("C:/Users/CHENYI/Desktop/竞彩预测") / D
desk_data.mkdir(parents=True, exist_ok=True)
(desk_data / f"竞彩预测数据_{D}.json").write_text(
    json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"→ 已生成 {(desk_data / ('竞彩预测数据_' + D + '.json'))}")

print("\n=== 修正摘要 ===")
for r in fixed:
    print(f"{r['no']} {r['pred_range']} 置信{r['conf']:.0%} "
          f"市场{r['mkt_range']}{r['mkt_conf']:.0%} 价值差{r['val']:+.0%} "
          f"入选={r['pick']}")
