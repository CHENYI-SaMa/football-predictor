# -*- coding: utf-8 -*-
"""daily_predict.py —— 每日预测流程（数据回填 + 重训 + 预测）
================================================================
按用户约定：当天赛果第二天才完整，所以预测前先把最近几天开奖
合并进训练数据（含 xG 补全）→ 重训 → 用最新模型预测（今明两天）。

流程：
  1. 合并最近 7 天开奖 → data/jczq_history.csv（去重，昨天赛果进模型）
  2. merge_xg.py 重跑 → data/jczq_history_xg.csv（xG/角球/大小球补全）
  3. train_v3.py 重训（几秒）→ models_v3 更新
  4. predict_v3.py 预测（今明两天，伤停标注）→ outputs/预测_v3_{date}.json
用法: python daily_predict.py [--save]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run(name: str, cmd: list[str], timeout: int = 600) -> bool:
    print(f"\n===== [{name}] =====")
    r = subprocess.run(cmd, timeout=timeout)
    ok = r.returncode == 0
    print(f"[{name}] {'✅ 成功' if ok else '❌ 失败 exit=' + str(r.returncode)}")
    return ok


def main():
    t0 = time.time()
    # 1. 合并最近 7 天开奖（昨天赛果进模型）
    ok1 = run("合并开奖(近7天)", [
        PY, "-c",
        "import sys; sys.path.insert(0, '.'); "
        "from retrain_weekly import merge_history; "
        "n = merge_history(7); print('训练数据总行数:', n)"])

    # 2. xG/角球/大小球补全
    ok2 = run("xG 补全(merge_xg)", [PY, "merge_xg.py"], timeout=300)

    # 3. 重训（用最新数据）
    ok3 = run("重训 V3", [PY, "train_v3.py"], timeout=600)

    # 4. 预测（今明两天，伤停标注）
    ok4 = run("预测 V3", [PY, "predict_v3.py", "--save"], timeout=600)

    print("\n" + "=" * 50)
    print(f"每日流程完成（耗时 {time.time()-t0:.0f}s）")
    print(f"  合并开奖: {'✅' if ok1 else '❌'} | xG补全: {'✅' if ok2 else '❌'} "
          f"| 重训: {'✅' if ok3 else '❌'} | 预测: {'✅' if ok4 else '❌'}")
    if not (ok1 and ok3 and ok4):
        print("⚠️ 有步骤失败，请检查上方日志")


if __name__ == "__main__":
    main()
