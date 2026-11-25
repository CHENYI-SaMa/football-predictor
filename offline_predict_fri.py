# -*- coding: utf-8 -*-
"""离线预测全部周五比赛（本地缓存 + 本地模型，不联网）"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict_jczq import predict_jczq

with open("data/jczq_cache/matches.json", encoding="utf-8") as f:
    ms = json.load(f)

friday = [m for m in ms if m["no"].startswith("周五")]
friday.sort(key=lambda x: int(x["no"][2:]))

print(f"共 {len(friday)} 场周五比赛\n")
records = []
for entry in friday:
    r = predict_jczq(entry, verbose=False)
    rec = r["recommendation"]
    scores = rec.get("scores", [])
    sc = " ".join(s["score"] for s in scores)
    rq = rec.get("rqspf", {})
    top1 = f"{rq['label']} {rq['prob']:.0%}"
    alt = rq.get("alt", "")
    pick = top1 if not alt else f"{top1} / {alt}"
    dp = rec["double_p"]
    tgd = r["total_goals_distribution"]
    over = sum(p for g, p in tgd.items() if g >= 3)
    rg = int(entry["rang"] or 0)
    rang_txt = "无让球" if rg == 0 else (f"客让{rg}球" if rg > 0 else f"主让{abs(rg)}球")
    print(f"{entry['no']} {entry['home']} vs {entry['away']}（{entry['league']}，{rang_txt}）")
    print(f"   比分参考：{sc}")
    print(f"   赛果预测：{pick}")
    print(f"   双P：{'★ 推荐' if dp['recommend'] else '不推'}（让平 {dp['range_draw']:.0%}）")
    print(f"   预期进球：主 {r['lambda_home']:.2f} / 客 {r['lambda_away']:.2f}，大2.5 {over:.0%}")
    print()
    records.append({
        "编号": entry["no"], "赛事": entry["league"],
        "主队": entry["home"], "客队": entry["away"],
        "让球": rang_txt, "日期": "2026-08-07",
        "最可能比分": r["most_likely_score"],
        "比分参考": sc,
        "推荐让球胜平负": rq.get("label", "-"),
        "让胜概率RS": round(r.get("range_home_win_prob", 0), 3) if r.get("range_home_win_prob") is not None else None,
        "让平概率RP": round(r.get("range_draw_prob", 0), 3) if r.get("range_draw_prob") is not None else None,
        "让负概率RF": round(r.get("range_away_win_prob", 0), 3) if r.get("range_away_win_prob") is not None else None,
        "双P": "推荐" if dp["recommend"] else "不推",
    })

out = Path("outputs") / "竞彩预测数据_20260807周五.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "results": records}, ensure_ascii=False, indent=2),
               encoding="utf-8")
print(f"→ 已保存 {out.name}")

