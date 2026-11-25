# -*- coding: utf-8 -*-
"""daily_check.py —— 自动生成「昨天预测」的赛果核对表格
========================================================
读取 outputs/ 下某日的预测 JSON，联网抓 500 网开奖数据回填真实赛果，
生成美化版 Excel 核对表（.xlsx，openpyxl）：
  行号 / 编号 / 赛事 / 主队 / 客队 / 让球 / 推荐TOP1 / 推荐TOP2 /
  实际比分 / 实际结果 / 实际让球结果 / 是否命中(✅❌⏳)
美化：标题汇总行 + 深蓝表头 + 全表居中 + 细边框 + 命中绿/未中红着色 +
隔行浅蓝 + 冻结首行 + 自动筛选 + 底部合计。
输出位置：
  - outputs/竞彩赛果核对_{date}.xlsx
  - 桌面/竞彩预测/{date}/竞彩赛果核对_{date}.xlsx   （若桌面目录存在）
同时把回填结果写回预测 JSON（供 GUI 历史视图显示）。

用法:
  python daily_check.py                # 核对昨天的预测
  python daily_check.py --date 20260809
  python daily_check.py --no-desktop   # 不复制到桌面
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
from path_helper import get_outputs_dir  # noqa: E402
sys.path.insert(0, str(BASE_DIR))
from src.jczq_history import fetch_day, fetch_day_trade  # noqa: E402

RESULT_LABELS = ["RS 让胜", "RP 让平", "RF 让负"]
# 推荐文本 → 让球结果编码（0=让胜 1=让平 2=让负）。
# V3 导出字段为中文（"让胜/让平/让负"），旧格式/复盘文件可能带 RS/RP/RF 或 "主推/让平"。
_RQ_CN = {"让胜": 0, "让平": 1, "让负": 2}
_RQ_CODE = {"RS": 0, "RP": 1, "RF": 2}


def rq_code(pred) -> int | None:
    """推荐文本 → 让球结果编码；兼容 '让胜' / 'RS' / '让胜/让平' 等格式。"""
    if not pred:
        return None
    p = str(pred).strip()
    cands = [p, p[:2]]
    if "/" in p:
        head = p.split("/")[0].strip()
        cands += [head, head[:2]]
    for c in cands:
        if c in _RQ_CN:
            return _RQ_CN[c]
        if c in _RQ_CODE:
            return _RQ_CODE[c]
    return None


def _yesterday() -> str:
    return (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y%m%d")


def backfill_day(date_iso: str, results: list[dict]) -> dict:
    """回填一天赛果，返回命中统计。结果直接修改 results（加字段）。
    抓「当天 + 前一天」两天的开奖：竞彩周日批次（周日晚-周一凌晨开赛）
    的开奖编号挂在周日日期，预测日期可能晚一天。
    数据源双路兜底：fetch_day（zx.500.com 开奖页）+ fetch_day_trade
    （trade.500.com 赛事页，开奖页未录入的场次用 trade 页补）。"""
    day_matches = {}
    for d in (date_iso,
              (pd.Timestamp(date_iso) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")):
        for fn in (fetch_day, fetch_day_trade):
            try:
                for m in fn(d, use_cache=False) if fn is fetch_day else fn(d):
                    day_matches.setdefault(m["no"], m)
            except Exception:
                pass
    hit = miss = nores = 0
    for r in results:
        r["日期"] = date_iso
        m = day_matches.get(r.get("编号", ""))
        if m is None:
            nores += 1
            r["实际比分"] = "未开赛"
            r["是否命中"] = None
            continue
        r["实际比分"] = f"{m['fthg']}:{m['ftag']}"
        r["实际半全场结果"] = m.get("htft", "-")
        # 实际结果 = 去掉让球后的正常胜负（主队视角）
        r["实际结果"] = "主胜" if m["fthg"] > m["ftag"] else (
            "平" if m["fthg"] == m["ftag"] else "主负")
        rang = int(m.get("rang", 0) or 0)
        actual = 0 if m["fthg"] + rang > m["ftag"] else (
            1 if m["fthg"] + rang == m["ftag"] else 2)
        r["实际让球结果"] = RESULT_LABELS[actual]
        pred = str(r.get("推荐让球胜平负", ""))
        pred_code = rq_code(pred)   # 兼容中文"让胜"与 RS/RP/RF 编码
        if pred_code is None:
            nores += 1
            r["是否命中"] = None
            continue
        ok = (pred_code == actual)
        r["是否命中"] = ok
        if ok:
            hit += 1
        else:
            miss += 1
    return {"总预测": hit + miss, "总命中": hit,
            "总命中率": round(hit / (hit + miss), 4) if hit + miss else None,
            "按日": [{"日期": date_iso, "预测": hit + miss, "命中": hit,
                      "未中": miss, "无结果": nores}]}


def make_table(results: list[dict]) -> pd.DataFrame:
    """从预测 results 生成核对表格（含回填字段）。"""
    rows = []
    for i, r in enumerate(results, 1):
        top1, top2, conf = build_recommend_top(r)
        # ---- 三列独立命中判断 ----
        # 实际比分
        act_score = r.get("实际比分", "")
        # 实际结果（普通胜平负）
        act_res = r.get("实际结果", "")
        h_res = None
        # 实际让球结果：推荐让球项是否命中（编码比较，兼容中英格式）
        act_rq = r.get("实际让球结果", "")
        if act_rq:
            code_act = _RQ_CODE.get(act_rq.split()[0])     # "RF 让负" → RF → 2
            code_pred = rq_code(r.get("推荐让球胜平负", ""))
            h_rq = (code_pred == code_act) if (code_pred is not None and code_act is not None) else None
        else:
            h_rq = None

        # ---- 命中判定：实际让球结果 命中 TOP1/TOP2 任一即算对 ----
        c_act = _RQ_CODE.get(act_rq.split()[0]) if act_rq else None
        if c_act is None:
            row_hit, hit_item = False, ""
        else:
            row_hit = (c_act == _RQ_CN[top1]) or (c_act == _RQ_CN[top2])
            hit_item = "TOP1" if c_act == _RQ_CN[top1] else ("TOP2" if c_act == _RQ_CN[top2] else "")
        hi_conf = conf >= 0.7
        # 命中标注单独成列；实际让球结果列保持纯结果值
        hit_show = hit_item if row_hit else ""

        rows.append({
            "行号": i,
            "编号": r.get("编号", ""),
            "赛事": r.get("赛事", ""),
            "主队": r.get("主队", ""),
            "客队": r.get("客队", ""),
            "让球": r.get("让球", ""),
            "推荐TOP1": top1,
            "推荐TOP2": top2,
            "置信度": (f"{conf:.0%}★" if conf >= 0.7 else f"{conf:.0%}"),
            "实际比分": act_score,
            "实际结果": act_res,
            "实际让球结果": act_rq,
            "命中项": hit_show,
            "_h_res": h_res, "_h_rq": h_rq,
            "_hit": row_hit, "_item": hit_item, "_hi_conf": hi_conf,
        })
    df = pd.DataFrame(rows)
    # 过滤掉未开赛（实际让球结果为空）的比赛行，并重新编号
    df = df[df["实际让球结果"].astype(str).str.strip() != ""].reset_index(drop=True)
    df["行号"] = range(1, len(df) + 1)
    return df


def build_recommend_top(r: dict) -> tuple:
    """TOP1/TOP2：让球胜平负三项概率取前二（V3 专项；V2 胜平负/半全场字段已移除）。
    返回 (top1, top2, conf)，conf 为 TOP1 概率。"""
    cands = {
        "让胜": r.get("让胜概率RS", 0) or 0,
        "让平": r.get("让平概率RP", 0) or 0,
        "让负": r.get("让负概率RF", 0) or 0,
    }
    ranked = sorted(cands.items(), key=lambda x: -x[1])
    return ranked[0][0], ranked[1][0], ranked[0][1]


def save_excel(df: pd.DataFrame, path, summary: dict, date_iso: str):
    """带美化的 Excel：标题行 + 表头（深蓝白字）+ 居中 + 边框 + 命中着色 + 汇总。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "赛果核对"

    thin = Side(style="thin", color="C9CDD4")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    header_fill = PatternFill("solid", fgColor="305496")
    hit_fill = PatternFill("solid", fgColor="A9D08E")     # 命中 绿底
    miss_fill = PatternFill("solid", fgColor="F4A7B9")    # 未中 红底
    wait_fill = PatternFill("solid", fgColor="FFFFFF")    # 无赛果 白底
    bold_cols = {"推荐TOP1", "实际比分", "实际结果", "实际让球结果"}
    result_cols = {"实际结果": "_h_res", "实际让球结果": "_h_rq"}
    vis_cols = [c for c in df.columns if not c.startswith("_")]
    ncols = len(vis_cols)

    # 1) 标题行（合并）
    ymd = f"{int(date_iso[5:7])}月{int(date_iso[8:10])}日"
    _v = df["实际让球结果"].astype(str)
    _nt = int(((~_v.isin(["-", "None"])) & (_v != "")).sum()); _nh = int(df["_hit"].sum())
    rate = f"共中 {_nh}/{_nt}（{_nh / _nt:.1%}）" if _nt else "暂无赛果"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    c = ws.cell(1, 1, f"{ymd} 竞彩赛果核对 · {rate}")
    c.font = Font(name="微软雅黑", size=13, bold=True, color="305496")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30

    # 2) 表头
    for j, col in enumerate(vis_cols, 1):
        cell = ws.cell(2, j, col)
        cell.font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border
    ws.row_dimensions[2].height = 24

    # 3) 数据行：仅「实际让球结果」列着色（命中TOP1/TOP2→绿，有赛果未中→红，无赛果→白），其余列全白
    rq_col = "实际让球结果"
    j_rq = vis_cols.index(rq_col) + 1
    for i, (_, row) in enumerate(df.iterrows()):
        r = i + 3
        ws.row_dimensions[r].height = 20
        for j, col in enumerate(vis_cols, 1):
            cell = ws.cell(r, j, row[col])
            cell.font = Font(name="微软雅黑", size=10, bold=(col in bold_cols))
            cell.alignment = center
            cell.border = border
            if col == rq_col:
                v = str(row[col])
                if row["_hit"]:
                    cell.fill = hit_fill
                elif v in ("-", "", "None"):
                    cell.fill = wait_fill
                else:
                    cell.fill = miss_fill
            else:
                cell.fill = wait_fill

    # 4) 汇总行：分段统计（未开赛行已过滤，len(df) 即有效场次；命中 = 实际让球结果 命中 TOP1/TOP2 任一）
    sr = len(df) + 3
    ws.merge_cells(start_row=sr, start_column=1, end_row=sr, end_column=ncols)
    n_total = len(df)
    n_hit = int(df["_hit"].sum())
    hi = df[df["_hi_conf"]]
    n_hi = len(hi)
    n_hi_hit = int(hi["_hit"].sum()) if n_hi else 0
    n_t1 = int((df["_item"] == "TOP1").sum())
    n_t2 = int((df["_item"] == "TOP2").sum())
    hi_t1 = int((hi["_item"] == "TOP1").sum()) if n_hi else 0
    hi_t2 = int((hi["_item"] == "TOP2").sum()) if n_hi else 0
    rate = f"{n_hit / n_total:.1%}" if n_total else "-"
    txt = (f"合计：共中 {n_hit}/{n_total}（{rate}）｜高置信中 {n_hi_hit}/{n_hi}｜"
           f"TOP1中 {n_t1}｜TOP2中 {n_t2}｜高置信内 TOP1中 {hi_t1}、TOP2中 {hi_t2}")
    c = ws.cell(sr, 1, txt)
    c.font = Font(name="微软雅黑", size=10, bold=True, color="305496")
    c.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[sr].height = 22

    # 5) 列宽 / 冻结 / 筛选
    widths = {"行号": 6, "编号": 9, "赛事": 10, "主队": 13, "客队": 13, "让球": 10,
              "推荐TOP1": 10, "推荐TOP2": 10, "置信度": 10,
              "实际比分": 10, "实际结果": 10,
              "实际让球结果": 12, "命中项": 8}
    for j, col in enumerate(vis_cols, 1):
        ws.column_dimensions[get_column_letter(j)].width = widths.get(col, 12)
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{get_column_letter(ncols)}{len(df) + 2}"

    wb.save(path)


def make_predict_table(results: list[dict], date_iso: str) -> pd.DataFrame:
    """从预测 results 生成【赛果预测】表：预测列 + 实际列(预置空待回填)。
    含：编号/赛事/主队/客队/让球/推荐TOP1/推荐TOP2/置信度/
        推荐胜平负/实际比分/实际结果/实际让球结果。"""
    rows = []
    for i, r in enumerate(results, 1):
        top1, top2, conf = build_recommend_top(r)
        # 推荐胜平负：优先取 GUI 导出 JSON 的「胜平负」字段（赔率反推），
        # 兼容旧字段名「推荐胜平负」与 predict_v4 原始键 sp_label，均缺失才兜底
        sp_label = r.get("胜平负", r.get("推荐胜平负", r.get("sp_label", "胜")))
        rows.append({
            "行号": i,
            "编号": r.get("编号", ""),
            "赛事": r.get("赛事", ""),
            "主队": r.get("主队", ""),
            "客队": r.get("客队", ""),
            "让球": r.get("让球", ""),
            "推荐TOP1": top1,
            "推荐TOP2": top2,
            "置信度": (f"{conf:.0%}★" if conf >= 0.7 else f"{conf:.0%}"),
            "推荐胜平负": sp_label,
            "实际比分": r.get("实际比分", ""),
            "实际结果": r.get("实际结果", ""),
            "实际让球结果": r.get("实际让球结果", ""),
        })
    return pd.DataFrame(rows)


def save_predict_excel(df: pd.DataFrame, path, date_iso: str):
    """【赛果预测】表美化 Excel：标题/表头/居中/边框/冻结（无命中着色）。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    wb = Workbook(); ws = wb.active; ws.title = "赛果预测"
    thin = Side(style="thin", color="C9CDD4")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    header_fill = PatternFill("solid", fgColor="305496")
    vis = list(df.columns)
    ncols = len(vis)
    ymd = f"{int(date_iso[5:7])}月{int(date_iso[8:10])}日"
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    c = ws.cell(1, 1, f"{ymd} 竞彩赛果预测（V4模型）")
    c.font = Font(name="微软雅黑", size=13, bold=True, color="305496")
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 30
    for j, col in enumerate(vis, 1):
        cell = ws.cell(2, j, col)
        cell.font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
        cell.fill = header_fill; cell.alignment = center; cell.border = border
    ws.row_dimensions[2].height = 24
    for i, (_, row) in enumerate(df.iterrows()):
        rr = i + 3
        ws.row_dimensions[rr].height = 20
        for j, col in enumerate(vis, 1):
            cell = ws.cell(rr, j, row[col])
            cell.font = Font(name="微软雅黑", size=10)
            cell.alignment = center; cell.border = border
    widths = {"行号": 6, "编号": 9, "赛事": 10, "主队": 13, "客队": 13, "让球": 10,
              "推荐TOP1": 10, "推荐TOP2": 10, "置信度": 10, "推荐胜平负": 10,
              "实际比分": 10, "实际结果": 10, "实际让球结果": 12}
    for w, col in enumerate(vis, 1):
        ws.column_dimensions[get_column_letter(w)].width = widths.get(col, 11)
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{get_column_letter(ncols)}{len(df) + 2}"
    wb.save(path)


def run(d: str, no_desktop: bool = False, both: bool = True) -> int:
    """执行某日的赛果预测+核对（供命令行与 GUI 补跑共用）。d=YYYYMMDD。
    both=True：同时生成 赛果预测.xlsx(预测表，实际列预置) 和 赛果核对.xlsx(回填表)。
    输出到 桌面/竞彩预测/{d}/（平铺两张表）。"""
    date_iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    out = get_outputs_dir()
    jf = out / f"竞彩预测数据_{d}.json"
    if not jf.exists():
        print(f"[跳过] {d} 无预测记录（outputs/竞彩预测数据_{d}.json 不存在）")
        return 1
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        print(f"[跳过] {d} 预测记录为空")
        return 1

    print(f"[回填] {date_iso} 共 {len(results)} 场，抓取开奖…")
    summary = backfill_day(date_iso, results)
    hit, pred_n = summary["总命中"], summary["总预测"]
    rate = f"{hit/pred_n:.1%}" if pred_n else "-"
    print(f"[回填] 命中 {hit}/{pred_n}（{rate}），无赛果 {summary['按日'][0]['无结果']} 场")

    # 写回 JSON（供 GUI 历史视图）
    data["回填核对"] = summary
    jf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成【赛果预测】+【赛果核对】两张表到 桌面/竞彩预测/{d}/
    desktop = Path.home() / "Desktop" / "竞彩预测" / d
    desktop.mkdir(parents=True, exist_ok=True)
    pred_df = make_predict_table(results, date_iso)
    pred_xlsx = desktop / f"赛果预测_{d}.xlsx"
    save_predict_excel(pred_df, pred_xlsx, date_iso)
    print(f"→ 赛果预测表: {pred_xlsx}")
    df = make_table(results)
    chk_xlsx = desktop / f"赛果核对_{d}.xlsx"
    save_excel(df, chk_xlsx, summary, date_iso)
    print(f"→ 赛果核对表: {chk_xlsx}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYYMMDD，默认昨天")
    ap.add_argument("--no-desktop", action="store_true", help="不复制到桌面")
    args = ap.parse_args()
    return run(args.date or _yesterday(), args.no_desktop)


if __name__ == "__main__":
    sys.exit(main())
