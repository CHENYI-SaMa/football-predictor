# -*- coding: utf-8 -*-
"""restore_week.py —— 恢复 8/14~8/16 当时的预测值（模型 8/17 更新前的快照）
================================================================
背景：8/16 json 回填时用当前模型（8/17 16:59 更新）覆盖了当时的 conf/pred_range，
导致推荐文件数值与当时推送不一致。本脚本：
1. 8/16：直接用用户提供的样例原文恢复推荐文件；解析样例恢复 json（pred_range/conf/rp_second/pick）
2. 8/14/15：解析桌面删除前的备份文件（当时推送 txt），恢复 json 并重新生成推荐文件
用法: python restore_week.py（系统python + venv依赖）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from gui import format_handicap_txt, format_handicap_csv  # noqa: E402

DESK = Path("C:/Users/CHENYI/Desktop/竞彩预测")
BAK = BASE_DIR / "备份" / "竞彩预测_备份_20260818"
SAMPLE_816 = BASE_DIR / "outputs" / "样例_20260816_用户提供.txt"


def parse_main(line: str):
    """主推行 -> (main, rp_second, conf)。兼容两种格式：
    让负/让平 85%★ | 让胜 76%（市场 让胜 49%，价值 +27%）"""
    m = re.match(r"^\s*(让胜|让平|让负)(?:/(让平))?\s*(\d+)%", line)
    if not m:
        return None
    return m.group(1), bool(m.group(2)), int(m.group(3)) / 100


def parse_file(text: str) -> dict:
    """解析推荐文件 -> {no: (main, rp_second, conf)}"""
    out = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        m = re.match(r"^(周[一二三四五六日]\d{3})\s", lines[i])
        if m and i + 1 < len(lines):
            pm = parse_main(lines[i + 1])
            if pm:
                out[m.group(1)] = pm
            i += 2
        else:
            i += 1
    return out


def restore_json(date: str, parsed: dict) -> None:
    dd = date.replace("-", "")
    path = BASE_DIR / "outputs" / f"预测_v3_{dd}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    n_pick = 0
    for r in data["results"]:
        if r["no"] in parsed:
            main, rp2, conf = parsed[r["no"]]
            r["pred_range"] = main
            r["conf"] = conf
            r["rp_second"] = rp2
            r["pick"] = True
            n_pick += 1
        else:
            r["pick"] = False
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {date}: json 恢复 {n_pick} 场当时预测 -> {path.name}")


def gen_desktop(date: str) -> None:
    dd = date.replace("-", "")
    path = BASE_DIR / "outputs" / f"预测_v3_{dd}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rs = data["results"]
    matches = [{"no": r["no"], "home": r["home"], "away": r["away"],
                "league": r.get("league", "")} for r in rs]
    v3 = {r["no"]: r for r in rs}
    folder = DESK / dd
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"让球胜平负推荐_{dd}.txt").write_text(
        format_handicap_txt(matches, v3, dd), encoding="utf-8")
    (folder / f"让球胜平负推荐_{dd}.csv").write_text(
        format_handicap_csv(matches, v3), encoding="utf-8")
    n = sum(1 for r in rs if r.get("pick"))
    print(f"  {date}: 推荐重新生成 {n} 场 -> {folder}")


def main():
    # 8/16：用户样例
    print("== 2026-08-16（用户样例恢复）==")
    sample = SAMPLE_816.read_text(encoding="utf-8")
    parsed = parse_file(sample)
    restore_json("2026-08-16", parsed)
    (DESK / "20260816" / "让球胜平负推荐_20260816.txt").write_text(
        sample if sample.endswith("\n") else sample + "\n", encoding="utf-8")
    print(f"  2026-08-16: 推荐文件 = 用户样例原文 ({len(parsed)} 场)")

    # 8/14：备份 周五/周六/周日
    print("== 2026-08-14（备份恢复）==")
    p = {}
    for f in BAK.glob("20260814/竞彩预测推荐V3_20260814_*.txt"):
        p.update(parse_file(f.read_text(encoding="utf-8")))
        print(f"    解析 {f.name}: {len(parse_file(f.read_text(encoding='utf-8')))} 场")
    restore_json("2026-08-14", p)
    gen_desktop("2026-08-14")

    # 8/15：备份 周六/周日/周一
    print("== 2026-08-15（备份恢复）==")
    p = {}
    for f in BAK.glob("20260815/竞彩预测推荐V3_20260815_*.txt"):
        fp = parse_file(f.read_text(encoding="utf-8"))
        p.update(fp)
        print(f"    解析 {f.name}: {len(fp)} 场")
    restore_json("2026-08-15", p)
    gen_desktop("2026-08-15")

    print("\n恢复完成")


if __name__ == "__main__":
    main()
