# -*- coding: utf-8 -*-
"""predict_v4.py —— 用 models_v4 预测当天竞彩（替代 predict_v3 的 V3→V4 升级）
================================================================
GUI 入口不要求返回结构变化，仅换模型：models_v3 → models_v4。
用法: python predict_v4.py [--save]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

def _find_models_dir() -> Path:
    """定位 models_v4：优先外部 data_v4/models_v4（bayes 重训位置），回退内置 models_v4。
    打包后：_MEIPASS=内置解包目录，exe 同级/上级可能有 data_v4；源码：脚本目录。"""
    cands = []  # (优先级, 路径)
    # 打包场景候选
    for base in (getattr(sys, "_MEIPASS", None),
                 sys.executable,
                 Path(sys.executable).resolve().parent,
                 Path(sys.executable).resolve().parent.parent,
                 BASE_DIR):
        if base is None:
            continue
        base = Path(base)   # 统一转 Path（_MEIPASS 可能是 str）
        cands.append(base / "data_v4" / "models_v4")
        cands.append(base / "models_v4")
    # 去重并按存在性返回：优先 data_v4(外部), 其次 data_v4 上级, 最后 models_v4(内置)
    seen = set(); result = []
    for c in cands:
        c = Path(c)
        if str(c) in seen:
            continue
        seen.add(str(c))
        if c.exists():
            result.append(c)
    # 外部 data_v4 优先：把含 data_v4 的放前面
    ext = [r for r in result if r.parts[-2] == "data_v4"]
    int_ = [r for r in result if r.parts[-2] != "data_v4"]
    ordered = ext + int_
    if ordered:
        return ordered[0]
    return BASE_DIR / "data_v4" / "models_v4"


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from src.evaluation import odds_baseline_probs          # noqa: E402
from src.features_v3 import FeatureBuilderV3            # noqa: E402
from src.jczq_fetcher import fetch_matches              # noqa: E402
from src.model import PoissonLightGBM                   # noqa: E402
M4 = _find_models_dir()


class _SafeStream:
    """窗口进程(console=False)下 stdout 可能为 None/无效，写失败会崩 OSError；
    包一层把写错误静默吞掉，并统一 UTF-8 编码避免 GBK 编不了 → ≥ ✓ 等字符。"""
    def __init__(self, dst):
        self._dst = dst

    def write(self, s):
        if self._dst is not None:
            try:
                return self._dst.write(s)
            except Exception:
                return None

    def flush(self):
        if self._dst is not None:
            try:
                self._dst.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        try:
            return self._dst.fileno()
        except Exception:
            return -1

    def reconfigure(self, *a, **k):
        try:
            self._dst.reconfigure(*a, **k)
        except Exception:
            pass

    @property
    def encoding(self):
        try:
            return self._dst.encoding or "utf-8"
        except Exception:
            return "utf-8"


if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_v4():
    model = PoissonLightGBM.load(M4 / "model.pkl")
    meta = json.loads((M4 / "train_meta.json").read_text(encoding="utf-8"))
    import pickle
    state = pickle.load(open(M4 / "feature_state.pkl", "rb"))
    return model, meta, state


def jczq_row(entry: dict, date: str, odds) -> dict:
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


def derive_extra(model, X, sp_odds=None) -> dict:
    """反推【普通胜平负】：用竞彩「普通胜平负赔率」(sp_h/d/a) 取最低赔率项（最被看好）。
    主队被看好 → 胜；无赔率时回退模型网格。返回 {"sp_label", "sp_probs"}。"""
    grids = model.predict_grids(X)          # (1,11,11)
    g = grids[0]
    n = g.shape[0]
    if sp_odds and all(isinstance(o, (int, float)) and o > 0 for o in sp_odds):
        _sp = np.array([1.0 / o for o in sp_odds], dtype=float)
        _sp /= _sp.sum()
        sp_label = ["胜", "平", "负"][int(_sp.argmax())]
        sp_probs = [round(float(x), 3) for x in _sp]
    else:                                   # 无赔率回退：模型网格
        ih, ja = np.indices((n, n))
        _sp = np.clip(np.stack([g[ih > ja].sum(), g[ih == ja].sum(),
                                g[ih < ja].sum()], 0), 1e-12, None)
        _sp /= _sp.sum()
        sp_label = ["胜", "平", "负"][int(_sp.argmax())]
        sp_probs = [round(float(x), 3) for x in _sp]
    return {"sp_label": sp_label, "sp_probs": sp_probs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.62)
    ap.add_argument("--min-val", type=float, default=0.10)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    model, meta, state = load_v4()
    print(f"V4 模型: 特征 {len(meta['feature_cols'])} 个, "
          f"训练至 {meta['data_range'][1]}")
    cols = meta["feature_cols"]

    matches = fetch_matches(use_cache=False)
    import datetime as _dt
    _week = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    _tp = _week[_dt.date.today().weekday()]
    _yp = _week[(_dt.date.today() + _dt.timedelta(days=1)).weekday()]
    _before = len(matches)
    matches = [m for m in matches if m.get("no", "")[:2] in (_tp, _yp)]
    print(f"预测 {len(matches)} 场（{_tp}+{_yp} 期次；已过滤其他期次 {_before - len(matches)} 场）\n")

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
            rp_second = int(np.argsort(pr)[-2]) == 1
            if all(x is not None for x in rsp):
                mo = odds_baseline_probs([rsp[0]], [rsp[1]], [rsp[2]])[0]
                mkt_i = int(np.argmax(mo))
                mkt_label = ["让胜", "让平", "让负"][mkt_i]
                mkt_conf = float(mo[mkt_i])
                val = float(conf - mo[pred_i])
                mkt_probs = [round(float(x), 3) for x in mo]
            else:
                mkt_label, mkt_conf, val = "-", 0.0, 0.0
                mkt_probs = None
            downgrade = m.get("league") in ("欧协联", "欧会杯", "欧罗巴")
            # ---- 反推普通胜平负：按普通赔率最低项（主队被看好=胜）----
            sp_odds = (m.get("sp_h"), m.get("sp_d"), m.get("sp_a"))
            try:
                ex = derive_extra(model, X, sp_odds=sp_odds)
            except Exception:
                ex = {"sp_label": "-", "sp_probs": None}
            results.append({
                "no": m.get("no"), "league": m.get("league"),
                "home": m.get("home"), "away": m.get("away"),
                "rang": int(m.get("rang", 0) or 0),
                "pred_range": pred_label, "conf": round(conf, 3),
                "rp_second": bool(rp_second),
                "probs": [round(float(x), 3) for x in pr],
                "sp_label": ex["sp_label"], "sp_probs": ex["sp_probs"],
                "mkt_range": mkt_label, "mkt_conf": round(mkt_conf, 3),
                "mkt_probs": mkt_probs,
                "val": round(val, 3),
                "pick": bool(not downgrade and conf >= args.min_conf
                             and val >= args.min_val),
                "downgrade": bool(downgrade),
                "model": "v4",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[跳过] {m.get('no')} {m.get('home')}vs{m.get('away')}: {e}")

    picks = sorted([r for r in results if r["pick"]], key=lambda x: -x["val"])
    print(f"=== V4 模型高置信推荐（conf≥{args.min_conf:.0%} 且 价值≥{args.min_val:.0%}）：{len(picks)} 场 ===")
    for x in picks:
        rng = RNG_LABEL.get(x["rang"], f"让{x['rang']}")
        main_pick = x["pred_range"] + ("/让平" if x.get("rp_second") else "")
        print(f"{x['no']} {x['home']} vs {x['away']} [{rng}] → {main_pick} {x['conf']:.0%}")
    if not picks:
        print("（无符合条件）")

    if args.save:
        if getattr(sys, "frozen", False):
            out_dir = Path.home() / "Desktop" / "竞彩预测" / "存档"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"预测_v4_{time.strftime('%Y%m%d')}.json"
        else:
            out = BASE_DIR / "outputs" / f"预测_v4_{time.strftime('%Y%m%d')}.json"
        out.write_text(json.dumps({"model": "models_v4", "date": time.strftime("%Y-%m-%d"),
                                   "results": results}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n→ 已保存 {out}")


if __name__ == "__main__":
    main()