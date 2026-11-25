# -*- coding: utf-8 -*-
"""predict_v3.py —— 用优化后的 models_v3（Glicko贝叶斯 + 去form + xG）预测当天竞彩
================================================================
与 predict_snapshot 同输出格式，但模型换成 models_v3（60.2% 让球版）。
用法: python predict_v3.py [--save] [--min-conf 0.60] [--min-val 0.10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation import odds_baseline_probs          # noqa: E402
from src.features_v3 import FeatureBuilderV3            # noqa: E402
from src.jczq_fetcher import fetch_matches              # noqa: E402
from src.model import PoissonLightGBM                   # noqa: E402

M3 = BASE_DIR / "models_v3"


def load_v3():
    model = PoissonLightGBM.load(M3 / "model.pkl")
    meta = json.loads((M3 / "train_meta.json").read_text(encoding="utf-8"))
    import pickle
    state = pickle.load(open(M3 / "feature_state.pkl", "rb"))
    return model, meta, state


def jczq_row(entry: dict, date: str, odds) -> dict:
    """与训练数据同 schema 的单场行（同 predict_jczq._jczq_row）。"""
    h, d, a = (odds if odds else (np.nan, np.nan, np.nan))
    return {
        "date": pd.Timestamp(date),
        "league": entry.get("league", ""),
        "home_team": entry["home"], "away_team": entry["away"],
        "fthg": -1, "ftag": -1, "ftr": "X",
        "htg": np.nan, "htag": np.nan,
        "hs": np.nan, "as": np.nan, "hst": np.nan, "ast": np.nan,
        "b365h": h, "b365d": d, "b365a": a,
        "ou_over": np.nan, "ou_under": np.nan,
        "ahh": float(int(entry.get("rang", 0) or 0)),
    }


def entry_date(entry: dict) -> str:
    mm_dd = entry.get("match_time", "").split()[0]
    if len(mm_dd) == 5 and mm_dd[2] == "-":
        return f"{time.strftime('%Y')}-{mm_dd}"
    return time.strftime("%Y-%m-%d")


def range_probs(model, X, rang: int):
    """让球胜平负概率（泊松网格 + 让球偏移）。"""
    grids = model.predict_grids(X)
    n = grids.shape[1]
    ih, ja = np.indices((n, n))
    shifted = ih[None, :, :] + rang
    pr = np.clip(np.stack([
        np.sum(grids * (shifted > ja), axis=(1, 2)),
        np.sum(grids * (shifted == ja), axis=(1, 2)),
        np.sum(grids * (shifted < ja), axis=(1, 2))], 1), 1e-12, 1.0)
    pr /= pr.sum(axis=1, keepdims=True)
    return pr[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--min-val", type=float, default=0.10)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    model, meta, state = load_v3()
    print(f"V3 模型: 特征 {len(meta['feature_cols'])} 个, "
          f"训练至 {meta['data_range'][1]}, 让球Acc {meta['metrics_test']['range_acc']:.1%}")
    print(f"特征列前5: {meta['feature_cols'][:5]} ...")
    cols = meta["feature_cols"]

    matches = fetch_matches(use_cache=False)
    # 预测范围：今天+明天两个期次（编号前缀）的全部比赛。
    # 按日期过滤会漏掉"明天期次但截止到后天凌晨"的比赛（如周三002 截止08-20），
    # 用户要求周三期次全部预测；更远的期次（后天起）赛程/阵容/赔率变化大，不预测。
    import datetime as _dt
    _week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    _tp = _week[_dt.date.today().weekday()]
    _yp = _week[(_dt.date.today() + _dt.timedelta(days=1)).weekday()]
    _before = len(matches)
    matches = [m for m in matches if m.get("no", "")[:2] in (_tp, _yp)]
    print(f"预测 {len(matches)} 场（{_tp}+{_yp} 期次；已过滤其他期次 {_before - len(matches)} 场）\n")

    # 加载当日伤停数据（fetch_injuries 产出），按编号索引
    inj_map = {}
    inj_path = BASE_DIR / "data" / "injuries" / f"injuries_{time.strftime('%Y-%m-%d')}.json"
    if inj_path.exists():
        for r in json.loads(inj_path.read_text(encoding="utf-8")).get("results", []):
            inj_map[r["no"]] = r["data"]
        print(f"伤停数据: {len(inj_map)} 场已载入\n")

    RNG_LABEL = {-3: "主让3", -2: "主让2", -1: "主让1", 0: "平手",
                 1: "客让1", 2: "客让2", 3: "客让3"}
    results = []
    for m in matches:
        if not m.get("home") or not m.get("away"):
            continue
        try:
            date = entry_date(m)
            rsp = (m.get("range_sp_h"), m.get("range_sp_d"), m.get("range_sp_a"))
            odds = rsp if all(x is not None for x in rsp) \
                else (m.get("sp_h"), m.get("sp_d"), m.get("sp_a"))
            row = jczq_row(m, date, odds)
            fdict = FeatureBuilderV3.build_single(state, pd.Series(row))
            X = np.array([[fdict.get(c, -1.0) for c in cols]], dtype=float)
            pr = range_probs(model, X, int(m.get("rang", 0) or 0))
            pred_i = int(np.argmax(pr))
            pred_label = ["让胜", "让平", "让负"][pred_i]
            conf = float(pr[pred_i])
            # 让平是否老二（第二大概率）→ 加防平
            rp_second = int(np.argsort(pr)[-2]) == 1
            # 市场让球 SP 去抽水
            if all(x is not None for x in rsp):
                mo = odds_baseline_probs([rsp[0]], [rsp[1]], [rsp[2]])[0]
                mkt_i = int(np.argmax(mo))
                mkt_label = ["让胜", "让平", "让负"][mkt_i]
                mkt_conf = float(mo[mkt_i])
                # 价值差 = 模型主推方向概率 - 市场同方向概率（pred_i 即模型主推索引）。
                # 旧写法 conf - mkt_conf 会把「让负48%」与「让胜52%」相减，方向
                # 相反时严重失真（如本例得出 -4%）；应取市场同方向概率比较。
                val = float(conf - mo[pred_i])
                mkt_probs = [round(float(x), 3) for x in mo]
            else:
                mkt_label, mkt_conf, val = "-", 0.0, 0.0
                mkt_probs = None
            # 伤停标注（按 no 匹配，主客分开统计）
            inj_summary = None
            inj = inj_map.get(m.get("no"))
            if inj:
                home_n = sum(1 for x in inj if x.get("team") == inj[0].get("team") and x.get("reason"))
                # 用 api_home/away 对不上中文，按伤停条目里的队名分别统计（两队的条目）
                import collections
                by_team = collections.Counter(x.get("team") for x in inj if x.get("reason"))
                top = by_team.most_common(2)
                inj_summary = {
                    "n_total": len(inj),
                    "teams": [{"team": t, "n": n} for t, n in top],
                    "key_absences": [x["player"] + "(" + (x.get("reason") or "")[:18] + ")"
                                     for x in inj[:4] if x.get("player")],
                }
            # 低可信联赛降级：欧协联等杯赛小国球队特征缺失，历史错误率66%（主让1球盘72.9%）
            downgrade = m.get("league") in ("欧协联", "欧会杯", "欧罗巴")
            results.append({
                "no": m.get("no"), "league": m.get("league"),
                "home": m.get("home"), "away": m.get("away"),
                "rang": int(m.get("rang", 0) or 0),
                "pred_range": pred_label, "conf": round(conf, 3),
                "rp_second": bool(rp_second),
                "probs": [round(float(x), 3) for x in pr],   # 让胜/让平/让负
                "mkt_range": mkt_label, "mkt_conf": round(mkt_conf, 3),
                "mkt_probs": mkt_probs,
                "val": round(val, 3),
                "pick": bool(not downgrade and conf >= args.min_conf
                             and val >= args.min_val),
                "downgrade": bool(downgrade),
                "injuries": inj_summary,
                "model": "v3-60.2%",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[跳过] {m.get('no')} {m.get('home')}vs{m.get('away')}: {e}")

    picks = sorted([r for r in results if r["pick"]], key=lambda x: -x["val"])
    print(f"=== V3 模型高置信推荐（conf≥{args.min_conf:.0%} 且 价值≥{args.min_val:.0%}）：{len(picks)} 场 ===")
    for x in picks:
        rng = RNG_LABEL.get(x["rang"], f"让{x['rang']}")
        main_pick = x["pred_range"] + ("/让平" if x.get("rp_second") else "")
        print(f"{x['no']} {x['home']} vs {x['away']} [{rng}]")
        print(f"  → {main_pick} {x['conf']:.0%}")
    if not picks:
        print("（无符合条件）")

    if args.save:
        # 打包版（frozen）：输出到桌面存档（可写、持久），与 GUI 的 get_outputs_dir 一致；
        # 源码版：项目 outputs/。
        if getattr(sys, "frozen", False):
            out_dir = Path.home() / "Desktop" / "竞彩预测" / "存档"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"预测_v3_{time.strftime('%Y%m%d')}.json"
        else:
            out = BASE_DIR / "outputs" / f"预测_v3_{time.strftime('%Y%m%d')}.json"
        out.write_text(json.dumps({"model": "models_v3", "date": time.strftime("%Y-%m-%d"),
                                   "results": results}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n→ 已保存 {out}")


if __name__ == "__main__":
    main()
