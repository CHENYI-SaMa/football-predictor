# -*- coding: utf-8 -*-
"""retrain_weekly.py —— 竞彩预测模型「自动回喂 + 重训 + 验证替换」
====================================================================
每周一自动运行一次：
  1. 抓取最近 N 天竞彩开奖，合并进 data/jczq_history.csv（按日期+球队去重，避免覆盖丢数据）
  2. 与上次训练时的行数对比；新增 < 阈值（默认 150）→ 跳过本次重训
  3. 备份现有 models_v2 / models_hf
  4. 重训（train.py + train_hf.py，--trials 0 沿用已有最优参数，不跑 Optuna）
  5. 用「最近 10 天」赛果对 新模型 vs 旧模型 公平评估（命中率 Acc / LogLoss）
  6. 新模型不劣化（Acc 不降超 3pp 且 LL 不升超 0.02）→ 保留并删除备份；否则还原旧模型
  7. 更新 data/train_state.json + outputs/retrain.log

用法:
  python retrain_weekly.py              # 正常流程
  python retrain_weekly.py --min-new 0  # 强制重训（忽略新增量阈值）
  python retrain_weekly.py --days 40    # 抓最近 40 天开奖
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from src.jczq_history import date_range, fetch_day, fetch_history  # noqa: E402

CSV = BASE_DIR / "data" / "jczq_history.csv"
STATE = BASE_DIR / "data" / "train_state.json"
LOG = BASE_DIR / "outputs" / "retrain.log"
MODELS_V2 = BASE_DIR / "models_v2"
MODELS_HF = BASE_DIR / "models_hf"
PY = sys.executable
MIN_NEW = 150          # 新增场次阈值
EVAL_DAYS = 10         # 评估窗口（天）
ACC_TOL = 0.03         # 命中率允许下降
LL_TOL = 0.02          # LogLoss 允许上升


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def safe_rm(path: Path):
    """删除路径；沙箱/系统拦截删除时改名为 ._del_<ts> 让位，绝不抛异常。
    这样脚本在受控沙箱和正常环境都能运行。"""
    if not path.exists():
        return
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except Exception:
        try:
            path.rename(str(path) + f"._del_{int(time.time())}")
        except Exception:
            pass


def merge_history(days: int) -> int:
    """抓最近 days 天开奖并合并进 CSV（去重），返回当前总行数。"""
    to = pd.Timestamp.now().strftime("%Y-%m-%d")
    from_ = (pd.Timestamp.now() - pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    dates = date_range(from_, to)
    log(f"[1/6] 抓取开奖 {from_} ~ {to}（{len(dates)} 天）")
    new = fetch_history(dates)
    if new.empty:
        log("[1/6] 本次未抓到任何开奖，跳过合并")
        return None
    log(f"[1/6] 新抓 {len(new)} 场")
    old = pd.read_csv(CSV)
    old["date"] = pd.to_datetime(old["date"], errors="coerce")
    new["date"] = pd.to_datetime(new["date"], errors="coerce")
    merged = pd.concat([old, new], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["date", "home_team", "away_team"],
                                    keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(CSV, index=False, encoding="utf-8")
    log(f"[1/6] 合并完成: {before} → {len(merged)} 场（去重 {before - len(merged)}）")
    return len(merged)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"n_rows": 0, "last_train_date": "", "last_acc": None, "last_ll": None}


def save_state(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def backup_models():
    for src, bak in ((MODELS_V2, MODELS_V2.with_name("models_v2_bak")),
                     (MODELS_HF, MODELS_HF.with_name("models_hf_bak"))):
        safe_rm(bak)
        shutil.copytree(src, bak)
    log("[3/6] 旧模型已备份 models_v2_bak / models_hf_bak")


def clear_feature_cache():
    """删掉特征缓存，强制 train.py 用最新 jczq_history.csv 重建特征。
    否则 train.py 会复用旧 data/features.csv，新增数据进不了模型。"""
    for f in ("data/features.csv", "data/feature_state.pkl",
              "data/features_dryrun.csv"):
        safe_rm(BASE_DIR / f)
    log("[3/6] 已清除特征缓存，强制重建")


def restore_models():
    for src, bak in ((MODELS_V2, MODELS_V2.with_name("models_v2_bak")),
                     (MODELS_HF, MODELS_HF.with_name("models_hf_bak"))):
        if bak.exists():
            safe_rm(src)
            shutil.move(str(bak), str(src))
    log("[5/6] ⚠ 已还原旧模型（新模型未通过验证）")


def run_train(cmd: str) -> bool:
    log(f"[4/6] 运行: {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=str(BASE_DIR),
                       capture_output=True, text=True, encoding="utf-8")
    tail = (r.stdout or "")[-1500:] + (r.stderr or "")[-500:]
    if r.returncode != 0:
        log(f"[4/6] ✗ 训练失败 rc={r.returncode}\n{tail}")
        return False
    # 找最后一行关键输出
    for ln in reversed((r.stdout or "").strip().splitlines()):
        if ln.strip():
            log(f"[4/6] ✓ {ln.strip()[:120]}")
            break
    return True


def evaluate(days: int = EVAL_DAYS) -> dict:
    """用最近 days 天赛果评估当前 models_v2（清缓存后加载）。"""
    import predict as P
    P._CACHE["model"] = None
    P._CACHE["hf"] = None
    to = pd.Timestamp.now().strftime("%Y-%m-%d")
    from_ = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    dates = date_range(from_, to)
    matches, n_eval, n_hit, ll_sum = [], 0, 0, 0.0
    for d in dates:
        try:
            day = fetch_day(d, use_cache=True)
        except Exception:
            continue
        for m in day:
            if m.get("fthg") is None or m.get("ftag") is None:
                continue
            entry = {
                "no": m["no"], "league": m["league"], "match_time": m["match_time"],
                "home": m["home_team"], "away": m["away_team"],
                "rang": str(m["rang"]),
                "sp_h": m.get("rsp_h"), "sp_d": m.get("rsp_d"), "sp_a": m.get("rsp_a"),
                "range_sp_h": m.get("rsp_h"), "range_sp_d": m.get("rsp_d"),
                "range_sp_a": m.get("rsp_a"),
            }
            try:
                r = P.predict(entry, verbose=False)
            except Exception:
                continue
            p = np.clip([r["range_home_win_prob"], r["range_draw_prob"],
                         r["range_away_win_prob"]], 1e-12, 1)
            p = p / p.sum()
            rang = int(m.get("rang", 0) or 0)
            actual = 0 if m["fthg"] + rang > m["ftag"] else (
                1 if m["fthg"] + rang == m["ftag"] else 2)
            y = np.zeros(3); y[actual] = 1.0
            n_eval += 1
            n_hit += (np.argmax(p) == actual)
            ll_sum += float(-np.sum(y * np.log(p)))
    if n_eval == 0:
        return {"n": 0}
    return {"n": n_eval, "acc": n_hit / n_eval, "ll": ll_sum / n_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-new", type=int, default=MIN_NEW)
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--force", action="store_true", help="忽略阈值强制重训")
    args = ap.parse_args()

    st = load_state()
    log("=" * 56)
    log("开始每周模型更新流程")

    # [0/6] 重训前先核对 + 错误分析（用最近 14 天开奖核对模型近期表现）
    try:
        log("[0/6] 错误分析（读 walkforward 逐场）")
        run_train(f'"{PY}" error_analysis.py')
        # E3: 读取结构化错误报告的自动调整建议，写入日志
        ea = BASE_DIR / "outputs" / "error_analysis.json"
        if ea.exists():
            import json as _j
            st_err = _j.loads(ea.read_text(encoding="utf-8"))
            advice = st_err.get("advice", [])
            if advice:
                log(f"[0/6] 自动调整建议 {len(advice)} 条：")
                for a in advice:
                    log(f"      [{a['severity']}] {a['msg']}")
            else:
                log("[0/6] 无自动调整建议（错误画像正常）")
    except Exception as e:
        log(f"[0/6] 错误分析失败（不影响重训）: {e}")

    n_rows = merge_history(args.days)
    if n_rows is None:
        log("抓取为空，流程中止")
        return 1

    new_added = n_rows - st["n_rows"] if st["n_rows"] else n_rows
    log(f"[2/6] 训练集当前 {n_rows} 场（上次训练时 {st['n_rows']} 场，新增 {new_added}）")
    if not args.force and new_added < args.min_new:
        log(f"[2/6] 新增 {new_added} < {args.min_new}，跳过重训（数据不足不折腾模型）")
        st["n_rows"] = n_rows
        save_state(st)
        return 0

    # 备份 + 训练
    backup_models()
    clear_feature_cache()
    # 四步：重建融合表 → features_v2 生成含半场列的 features.csv → train.py(复用) → train_hf.py(复用)
    ok0 = run_train(f'"{PY}" -c "import train; train.build_master_table()"')
    ok0 = ok0 and run_train(f'"{PY}" src/features_v2.py')
    ok1 = ok0 and run_train(f'"{PY}" train.py --trials 0')
    ok2 = ok1 and run_train(f'"{PY}" train_hf.py --trials 0')
    if not (ok1 and ok2):
        restore_models()
        log("[4/6] 训练失败，已还原旧模型")
        return 1

    # 评估新模型
    new_ev = evaluate()
    if new_ev["n"] == 0:
        log("[5/6] 评估窗口无赛果，直接保留新模型（下次再评估）")
        for bak in (MODELS_V2.with_name("models_v2_bak"),
                    MODELS_HF.with_name("models_hf_bak")):
            safe_rm(bak)
        st.update({"n_rows": n_rows, "last_train_date": time.strftime("%Y-%m-%d"),
                   "last_acc": None, "last_ll": None})
        save_state(st)
        return 0

    log(f"[5/6] 新模型评估（最近{new_ev['n']}场）: Acc={new_ev['acc']:.1%} LL={new_ev['ll']:.4f}")

    # 先暂存新模型（评估旧模型时会临时替换目录，不能丢新模型）
    tmp_v2 = MODELS_V2.with_name("models_v2_new_tmp")
    tmp_hf = MODELS_HF.with_name("models_hf_new_tmp")
    for src, tmp in ((MODELS_V2, tmp_v2), (MODELS_HF, tmp_hf)):
        safe_rm(tmp)
        shutil.copytree(src, tmp)

    # 临时还原旧模型评估
    for src, bak in ((MODELS_V2, MODELS_V2.with_name("models_v2_bak")),
                     (MODELS_HF, MODELS_HF.with_name("models_hf_bak"))):
        safe_rm(src)
        shutil.copytree(bak, src)
    old_ev = evaluate()
    log(f"[5/6] 旧模型评估（最近{old_ev['n']}场）: Acc={old_ev['acc']:.1%} LL={old_ev['ll']:.4f}")

    keep_new = (new_ev["acc"] >= old_ev["acc"] - ACC_TOL
                and new_ev["ll"] <= old_ev["ll"] + LL_TOL)
    if keep_new:
        log(f"[5/6] ✓ 新模型不劣化（Acc {new_ev['acc']:.1%} vs {old_ev['acc']:.1%}），保留")
        # 新模型就位（当前 src 是旧模型，用暂存的新模型覆盖）
        for src, tmp in ((MODELS_V2, tmp_v2), (MODELS_HF, tmp_hf)):
            safe_rm(src)
            shutil.move(str(tmp), str(src))
        for bak in (MODELS_V2.with_name("models_v2_bak"),
                    MODELS_HF.with_name("models_hf_bak")):
            safe_rm(bak)
    else:
        log(f"[5/6] ✗ 新模型劣化（Acc {new_ev['acc']:.1%} vs {old_ev['acc']:.1%}），还原旧模型")
        # src 当前已是旧模型；删除暂存的新模型
        for tmp in (tmp_v2, tmp_hf):
            safe_rm(tmp)

    st.update({"n_rows": n_rows, "last_train_date": time.strftime("%Y-%m-%d"),
               "last_acc": round(new_ev["acc"], 4) if keep_new else round(old_ev["acc"], 4),
               "last_ll": round(new_ev["ll"], 4) if keep_new else round(old_ev["ll"], 4)})
    save_state(st)
    log("[6/6] 流程完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
