# -*- coding: utf-8 -*-
"""predict_value.py —— 价值投注/选择性推荐（任务5）
=====================================================
对当天竞彩比赛逐场预测，计算：
  - 模型让球主推概率 conf（max of 让胜/让平/让负）
  - 市场让球 SP 去抽水主推概率 mkt_conf
  - 价值差 val = conf - mkt_conf
按策略过滤（默认：conf>=0.60 且 val>=0.10，回测命中 70.1%），
输出高置信推荐列表（JSON + 控制台），供三快照自动化调用。

用法:
  python predict_value.py                     # 默认策略过滤
  python predict_value.py --min-conf 0.55 --min-val 0.05
  python predict_value.py --no-filter         # 全部比赛（含价值标注）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from predict_jczq import predict_jczq           # noqa: E402
from src.jczq_fetcher import fetch_matches      # noqa: E402
from src.poisson_utils import half_full_probs   # noqa: E402


def implied(oh, od, oa):
    try:
        arr = np.array([oh, od, oa], dtype=float)
    except (TypeError, ValueError):
        return None
    if np.any(~np.isfinite(arr)) or np.any(arr <= 1):
        return None
    inv = 1.0 / arr
    return tuple(inv / inv.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.60,
                    help="模型让球主推概率下限（默认0.60）")
    ap.add_argument("--min-val", type=float, default=0.10,
                    help="价值差下限=模型主推-市场主推（默认0.10）")
    ap.add_argument("--no-filter", action="store_true", help="不过滤，输出全部")
    ap.add_argument("--snapshot", default="", help="快照标识(open/1700/2100)，写进文件名")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    matches = fetch_matches(use_cache=False)
    if not matches:
        print("未获取到比赛列表（可能无赛程或网络异常）")
        return
    print(f"共 {len(matches)} 场\n")

    # 读取欧赔初终盘变化（任务4：data/odds_change/{date}.json）
    odds_map = {}
    try:
        oc = BASE_DIR / "data" / "odds_change" / f"{time.strftime('%Y-%m-%d')}.json"
        if oc.exists():
            for m in json.loads(oc.read_text(encoding="utf-8")).get("matches", []):
                odds_map[str(m["match_id"])] = m
            print(f"[欧赔] 加载 {len(odds_map)} 场初终盘变化\n")
    except Exception:
        pass

    results = []
    for m in matches:
        if not m.get("home") or not m.get("away"):
            print(f"[跳过] {m.get('no')} 主客队名为空")
            continue
        try:
            r = predict_jczq(m, verbose=False)
        except Exception as e:
            print(f"[跳过] {m.get('no')} {m.get('home')}vs{m.get('away')}: {e}")
            continue
        rang = int(m.get("rang") or 0)
        # 模型让球主推
        conf = max(r.get("range_home_win_prob", 0), r.get("range_draw_prob", 0),
                   r.get("range_away_win_prob", 0))
        pred_r = ["RS", "RP", "RF"][int(np.argmax(
            [r.get("range_home_win_prob", 0), r.get("range_draw_prob", 0),
             r.get("range_away_win_prob", 0)]))]
        # 市场让球 SP 去抽水主推
        mkt = implied(m.get("range_sp_h"), m.get("range_sp_d"), m.get("range_sp_a"))
        mkt_conf = 0.0
        mkt_r = "-"
        if mkt:
            mkt_conf = max(mkt)
            mkt_r = ["RS", "RP", "RF"][int(np.argmax(mkt))]
        val = conf - mkt_conf
        # 半全场 top2（供推荐卡）
        ht = r.get("ht_ft_probs") or {}
        ht_top2 = " ".join(f"{k}{v:.0%}" for k, v in
                           sorted([(k, v) for k, v in ht.items() if k != "half_draw_prob"],
                                  key=lambda x: -x[1])[:2])
        # 欧赔初终盘异动（任务4 数据）
        oc = odds_map.get(str(m.get("match_id")))
        odds_delta = None
        if oc:
            d = oc.get("delta_implied", [])
            odds_delta = {"open": oc.get("avg_open"), "close": oc.get("avg_close"),
                          "d_h": d[0] if d else None, "d_d": d[1] if len(d) > 1 else None,
                          "d_a": d[2] if len(d) > 2 else None}
        results.append({
            "no": m.get("no"), "league": m.get("league"),
            "home": m.get("home"), "away": m.get("away"), "rang": rang,
            "pred_range": pred_r, "conf": round(conf, 3),
            "mkt_range": mkt_r, "mkt_conf": round(mkt_conf, 3),
            "val": round(val, 3),
            "model_spf": {"H": round(r.get("home_win_prob", 0), 3),
                          "D": round(r.get("draw_prob", 0), 3),
                          "A": round(r.get("away_win_prob", 0), 3)},
            "most_likely": r.get("most_likely_score"),
            "ht_ft": ht_top2,
            "lambda": (round(r.get("lambda_home", 0), 2), round(r.get("lambda_away", 0), 2)),
            "confidence": r.get("confidence"),
            "odds_delta": odds_delta,
        })

    # 排序：价值差降序
    results.sort(key=lambda x: -x["val"])
    if not args.no_filter:
        picks = [x for x in results if x["conf"] >= args.min_conf and x["val"] >= args.min_val]
    else:
        picks = results

    print("=" * 64)
    print(f"高置信推荐（conf≥{args.min_conf} 且 val≥{args.min_val}）：{len(picks)}/{len(results)} 场")
    print("=" * 64)
    for x in picks:
        rng = ("主让" + str(-x["rang"]) if x["rang"] < 0
               else "客让" + str(x["rang"]) if x["rang"] > 0 else "平手")
        print(f"{x['no']} {x['home']} vs {x['away']} [{rng}]")
        print(f"  → {x['pred_range']} {x['conf']:.0%}（市场{x['mkt_range']} {x['mkt_conf']:.0%}，"
              f"价值 {x['val']:+.0%}）")
        print(f"  比分 {x['most_likely']} | 半全场 {x['ht_ft']} | 置信 {x['confidence']}")
        od = x.get("odds_delta")
        if od and od.get("d_h") is not None:
            flag = "⚠️" if max(abs(od["d_h"]), abs(od["d_a"])) >= 0.03 else ""
            print(f"  欧赔初→终: {od['open'][0]:.2f}/{od['open'][1]:.2f}/{od['open'][2]:.2f}"
                  f" → {od['close'][0]:.2f}/{od['close'][1]:.2f}/{od['close'][2]:.2f} "
                  f"(主{od['d_h']:+.0%} 客{od['d_a']:+.0%}){flag}")
        print()
    if not picks:
        print("（无符合条件的比赛）")

    if args.save:
        tag = f"_{args.snapshot}" if args.snapshot else ""
        date = time.strftime("%Y%m%d")
        p = BASE_DIR / "outputs" / f"价值推荐{tag}_{date}.json"
        p.write_text(json.dumps({
            "date": time.strftime("%Y-%m-%d"), "snapshot": args.snapshot,
            "min_conf": args.min_conf, "min_val": args.min_val,
            "n_total": len(results), "n_pick": len(picks),
            "picks": picks, "all": results,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ 已保存 {p}")


if __name__ == "__main__":
    main()
