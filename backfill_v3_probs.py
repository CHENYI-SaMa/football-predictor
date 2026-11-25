# -*- coding: utf-8 -*-
"""backfill_v3_probs.py —— 给缺 probs/rp_second/pick 的历史 预测_v3 json 补算模型概率
================================================================
场景：8/16 及更早的 预测_v3_{date}.json 由旧版生成（无 probs / rp_second / pick 或值不完整），
用当前 models_v3 重新推理（复用 predict_v3.py 逻辑），写回原文件，保留 injuries 等原字段。
用法: python backfill_v3_probs.py --date 2026-08-16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation import odds_baseline_probs          # noqa: E402
from src.features_v3 import FeatureBuilderV3            # noqa: E402
from src.jczq_fetcher import fetch_matches              # noqa: E402
import predict_v3 as P                                  # noqa: E402


def backfill(date: str) -> None:
    pred_path = BASE_DIR / "outputs" / f"预测_v3_{date.replace('-', '')}.json"
    if not pred_path.exists():
        print(f"[跳过] 未找到 {pred_path}")
        return

    data = json.loads(pred_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    model, meta, state = P.load_v3()
    cols = meta["feature_cols"]

    ms = fetch_matches(use_cache=False, date=date)
    m_by_no = {m.get("no"): m for m in ms}
    print(f"{date}: 预测 {len(results)} 场，抓取比赛 {len(ms)} 场")

    fixed = 0
    for r in results:
        m = m_by_no.get(r.get("no"))
        if not m:
            continue
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
            r["pred_range"] = ["让胜", "让平", "让负"][pred_i]
            r["conf"] = round(conf, 3)
            r["rp_second"] = bool(rp_second)
            r["probs"] = [round(float(x), 3) for x in pr]
            r["mkt_range"] = (["让胜", "让平", "让负"][mkt_i] if mkt_i >= 0 else "-")
            r["mkt_conf"] = round(float(mo[mkt_i]), 3) if mkt_i >= 0 else 0.0
            r["mkt_probs"] = mkt_probs
            r["val"] = round(val, 3)
            r["pick"] = bool(not r.get("downgrade", False)
                             and conf >= 0.60 and val >= 0.10)
            fixed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [跳过] {r.get('no')}: {e}")

    pred_path.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"→ 已更新 {fixed}/{len(results)} 场 -> {pred_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="比赛日 YYYY-MM-DD")
    args = ap.parse_args()
    backfill(args.date)
