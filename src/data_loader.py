# -*- coding: utf-8 -*-
"""
数据加载模块
============
职责：
1. 从 data/raw/ 加载 football-data.co.uk 下载的 CSV（五大联赛，2013-2025 赛季）
2. 统一列名 / 数据类型 / 日期格式，输出标准化的 pandas DataFrame
3. 提供 `generate_synthetic_data()` 作为数据获取失败的兜底方案（生成符合
   真实足球统计特��的模拟数据：主队优势、低比分偏好、强度随联赛浮动）

标准输出 schema（与任务书 3.2 节要求对齐）：
    date        : 比赛日期 (datetime64)
    league      : 联赛代码 (E0=英超, SP1=西甲, D1=德甲, I1=意甲, F1=法甲)
    home_team   : 主队名
    away_team   : 客队名
    fthg / ftag : 全场主队 / 客队进球
    ftr         : 赛果 (H/D/A)
    htg / htag  : 半场主队 / 客队进球 (可选)
    hs / as     : 全场主队 / 客队射门 (可选)
    hst / ast   : 全场主队 / 客队射正 (可选, 缺失时用 HS*0.35 兜底)
    b365h/d/a   : Bet365 初始主胜/平/客胜赔率 (关键特征)
    ou_over / ou_under : Bet365 初始大 2.5 / 小 2.5 赔率
    ahh         : 亚洲让球盘口 (负=主队让球)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# 联赛代码 -> 中文名（用于日志输出）
LEAGUE_NAMES = {"E0": "英超", "SP1": "西甲", "D1": "德甲", "I1": "意甲", "F1": "法甲"}

# 各赛季统一字段重命名映射（football-data.co.uk 列名随年份略有演进）
_COLUMN_MAP = {
    "Div": "league",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "fthg",
    "FTAG": "ftag",
    "FTR": "ftr",
    "HTHG": "htg",
    "HTAG": "htag",
    "HS": "hs",
    "AS": "as",
    "HST": "hst",
    "AST": "ast",
    "B365H": "b365h",
    "B365D": "b365d",
    "B365A": "b365a",
    "B365>2.5": "ou_over",
    "B365<2.5": "ou_under",
    "AHh": "ahh",
}


def _load_single_csv(path: Path) -> pd.DataFrame:
    """读取单个 CSV，统一列名与日期格式。"""
    # utf-8-sig 处理文件头的 BOM
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = df.rename(columns=_COLUMN_MAP)
    keep = [
        "league", "date", "home_team", "away_team",
        "fthg", "ftag", "ftr", "htg", "htag",
        "hs", "as", "hst", "ast",
        "b365h", "b365d", "b365a", "ou_over", "ou_under", "ahh",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = np.nan
    df = df[keep]
    # 日期: dd/mm/yyyy (部分赛季为 dd/mm/yy)
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        parsed = pd.to_datetime(df["date"], format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            df["date"] = parsed
            break
    # 数值列
    numeric_cols = ["fthg", "ftag", "htg", "htag", "hs", "as", "hst", "ast",
                    "b365h", "b365d", "b365a", "ou_over", "ou_under", "ahh"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_football_data(raw_dir: str | os.PathLike | None = None) -> pd.DataFrame:
    """
    加载 data/raw/ 下全部 CSV 并合并为一个 DataFrame。

    Returns
    -------
    pd.DataFrame
        标准化后的全量历史比赛数据，按日期升序排列。
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DIR
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"目录 {raw_dir} 下没有 CSV 文件。请先运行下载脚本，或调用 "
            "generate_synthetic_data() 生成模拟数据。"
        )
    frames = [_load_single_csv(p) for p in csv_files]
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("date").reset_index(drop=True)
    # 去掉无赛果或无数值的行（无法用于监督学习）
    data = data.dropna(subset=["fthg", "ftag", "ftr"]).reset_index(drop=True)
    # 射正缺失时用射门数估计 (xG 代理：射正占比约 35%)
    if "hst" in data.columns:
        data["hst"] = data["hst"].fillna(data["hs"] * 0.35)
        data["ast"] = data["ast"].fillna(data["as"] * 0.35)
    # 球队名归一化（去除前后空格）
    for col in ("home_team", "away_team"):
        data[col] = data[col].astype(str).str.strip()
    return data


def generate_synthetic_data(n_seasons: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    生成符合真实足球统计特征的模拟数据（仅当真实数据不可获取时使用）。

    生成模型（隐含强度型泊松过程）：
        log(λ_home) = μ_home + home_adv + attack_home - defence_away + 联赛偏移
        log(λ_away) = μ_away + attack_away - defence_home + 联赛偏移
    其中 attack/defence 为每支球队的潜在强度（随时间缓慢漂移的随机游走），
    home_adv ≈ 0.30（主队优势对应场均进球约 +0.35）。

    输出列与 load_football_data() 完全一致（赔率为根据真实强度换算的
    模拟"市场价"：先算无抽水概率，再加 6% 边际抽水转成赔率）。

    Parameters
    ----------
    n_seasons : int
        模拟赛季数，每赛季约 3800 场（20 队双循环 × 5 联赛）。
    seed : int
        随机种子，保证可复现。

    Returns
    -------
    pd.DataFrame
        与真实数据同 schema 的模拟比赛表。
    """
    rng = np.random.default_rng(seed)
    leagues = ["E0", "SP1", "D1", "I1", "F1"]
    n_teams = 20
    # 每队初始攻击/防守强度
    attack0 = rng.normal(0.0, 0.45, n_teams)
    defence0 = rng.normal(0.0, 0.40, n_teams)
    # 联赛水平偏移（五大联赛场均进球略有差异）
    league_offset = {"E0": 0.0, "SP1": 0.02, "D1": 0.05, "I1": -0.03, "F1": 0.0}
    base_rate = 1.30  # 中位球队主场场均进球基准
    home_adv = 0.30

    rows = []
    for season in range(n_seasons):
        # 跨赛季强度漂移（模拟球队实力随时间变化）
        attack = attack0 + rng.normal(0, 0.08, n_teams)
        defence = defence0 + rng.normal(0, 0.07, n_teams)
        team_names = [f"Team{i:02d}_{season}" for i in range(n_teams)]
        for lg in leagues:
            off = league_offset[lg]
            for home in range(n_teams):
                for away in range(n_teams):
                    if home == away:
                        continue
                    lam_h = base_rate + home_adv + attack[home] - defence[away] + off
                    lam_a = base_rate - 0.18 + attack[away] - defence[home] + off
                    lam_h, lam_a = max(0.05, lam_h), max(0.05, lam_a)
                    y_h, y_a = rng.poisson(lam_h), rng.poisson(lam_a)
                    if y_h > 10:
                        y_h = 10
                    if y_a > 10:
                        y_a = 10
                    # 由强度换算"公平概率"再加 6% 抽水 -> 市场赔率
                    p = _poisson_prob_grid(lam_h, lam_a)
                    pH, pD, pA = p[0, 0], p[0, 1], p[0, 2]
                    over = p[1, 0] + p[1, 1]
                    margin = 1.06
                    row = {
                        "date": pd.Timestamp("2014-08-01") + pd.Timedelta(
                            days=season * 365 + (home * n_teams + away) * 2),
                        "league": lg,
                        "home_team": team_names[home],
                        "away_team": team_names[away],
                        "fthg": y_h, "ftag": y_a,
                        "ftr": "H" if y_h > y_a else ("A" if y_h < y_a else "D"),
                        "htg": rng.poisson(lam_h * 0.42).clip(0, 10),
                        "htag": rng.poisson(lam_a * 0.42).clip(0, 10),
                        "hs": int(rng.normal(13, 3.5)), "as": int(rng.normal(11, 3.5)),
                        "hst": int(rng.normal(4.6, 1.8)), "ast": int(rng.normal(3.8, 1.6)),
                        "b365h": round(1.06 / pH, 2), "b365d": round(1.06 / pD, 2),
                        "b365a": round(1.06 / pA, 2),
                        "ou_over": round(1.06 / over, 2), "ou_under": round(1.06 / (1 - over), 2),
                        "ahh": round(lam_h - lam_a - home_adv, 1),
                    }
                    rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _poisson_prob_grid(lam_h: float, lam_a: float) -> np.ndarray:
    """返回 [ [pH, pD, pA], [p_over, p_under] ] 的小网格概率。"""
    from scipy.stats import poisson
    maxg = 12
    kh = np.arange(maxg + 1)
    ka = np.arange(maxg + 1)
    pH = poisson.pmf(kh, lam_h)
    pA = poisson.pmf(ka, lam_a)
    grid = np.outer(pH, pA)
    total = np.add.outer(kh, ka)
    ih, ja = np.indices(grid.shape)
    p_home = grid[ih > ja].sum()   # 主队进球 > 客队进球
    p_away = grid[ih < ja].sum()   # 主队进球 < 客队进球
    p_draw = grid[ih == ja].sum()
    p_over = grid[total >= 3].sum()
    return np.array([[p_home, p_draw, p_away], [p_over, 1 - p_over]])


if __name__ == "__main__":
    # 快速自检
    df = load_football_data()
    print(df.shape)
    print(df[["date", "league", "home_team", "away_team", "fthg", "ftag"]].head(5).to_string())
    print("空值占比:\n", df.isna().mean().round(3).to_string())
