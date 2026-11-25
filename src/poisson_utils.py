# -*- coding: utf-8 -*-
"""
泊松分布概率工具（对应任务书 4.3 节）
======================================
从 λ_h, λ_a 生成：
1. 独立泊松联合概率网格 P(i, j), i,j ∈ [0, 10]
2. Dixon-Coles 低比分修正（τ 因子），再归一化
3. 赛果概率（主胜/平/客胜）、总进球分布、最可能比分
4. 通过对数似然网格搜索优化 Dixon-Coles 相关性参数 ρ
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

MAX_GOALS = 10  # 比分网格上限 0..10


# ---------------------------------------------------------------------------
# 基础概率网格
# ---------------------------------------------------------------------------
def poisson_grid(lam_h: float, lam_a: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    独立泊松联合概率矩阵 P(i, j)，形状 (max_goals+1, max_goals+1)。
    P[i, j] = P(主队进 i 球) * P(客队进 j 球)。
    """
    kh = np.arange(max_goals + 1)
    ka = np.arange(max_goals + 1)
    ph = poisson.pmf(kh, lam_h)
    pa = poisson.pmf(ka, lam_a)
    grid = np.outer(ph, pa)
    # 截断到 [0, max_goals] 网格，尾部概率按比例摊回并归一化
    # （λ<4 时尾部概率 <2%，扭曲可忽略）
    s = grid.sum()
    return grid / s if s > 0 else np.full_like(grid, 1.0 / grid.size)


# ---------------------------------------------------------------------------
# Dixon-Coles 修正
# ---------------------------------------------------------------------------
def dixon_coles_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon-Coles τ 调整因子（任务书 4.3 节公式）：
        (0,0): 1 - λh·λa·ρ
        (0,1): 1 + λh·ρ
        (1,0): 1 + λa·ρ
        (1,1): 1 - ρ
        其他 : 1
    """
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_grid(lam_h: float, lam_a: float, rho: float,
                     max_goals: int = MAX_GOALS) -> np.ndarray:
    """独立泊松网格 × τ 因子，然后重新归一化。"""
    grid = poisson_grid(lam_h, lam_a, max_goals)
    for i in range(2):
        for j in range(2):
            grid[i, j] *= dixon_coles_tau(i, j, lam_h, lam_a, rho)
    # 保证概率非负（极端 ρ 可能产生负值，clip 后重归一化）
    grid = np.clip(grid, 0.0, None)
    s = grid.sum()
    return grid / s if s > 0 else np.full_like(grid, 1.0 / grid.size)


def optimize_rho(lam_h: np.ndarray, lam_a: np.ndarray,
                 y_h: np.ndarray, y_a: np.ndarray,
                 rho_grid: np.ndarray | None = None) -> float:
    """
    通过对数似然网格搜索优化 ρ（向量化）：
        ρ* = argmax_ρ Σ log P_τ(y_h, y_a | λh, λa)

    Dixon-Coles 修正后概率需重新归一化，似然为
        log P_τ(obs) = log p_obs + log τ_obs - log S
    其中 S = 1 + p00(τ00-1) + p01(τ01-1) + p10(τ10-1) + p11(τ11-1)，
    τ00=1-λhλaρ, τ01=1+λhρ, τ10=1+λaρ, τ11=1-ρ。
    所有量均可对样本向量化计算，ρ 在 [-0.15, 0.15] 网格上扫描。
    """
    if rho_grid is None:
        rho_grid = np.linspace(-0.15, 0.15, 301)
    lam_h = np.clip(np.asarray(lam_h, dtype=float), 1e-6, None)
    lam_a = np.clip(np.asarray(lam_a, dtype=float), 1e-6, None)
    y_h = np.clip(np.asarray(y_h, dtype=int), 0, MAX_GOALS)
    y_a = np.clip(np.asarray(y_a, dtype=int), 0, MAX_GOALS)

    # 四个关键格子的独立泊松概率（形状 N）
    p00 = poisson.pmf(0, lam_h) * poisson.pmf(0, lam_a)
    p01 = poisson.pmf(0, lam_h) * poisson.pmf(1, lam_a)
    p10 = poisson.pmf(1, lam_h) * poisson.pmf(0, lam_a)
    p11 = poisson.pmf(1, lam_h) * poisson.pmf(1, lam_a)
    # 观测比分处的独立概率
    p_obs = poisson.pmf(y_h, lam_h) * poisson.pmf(y_a, lam_a)
    # 观测比分所属 DC 类别: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1) 4=其他
    kind = np.where((y_h == 0) & (y_a == 0), 0,
            np.where((y_h == 0) & (y_a == 1), 1,
            np.where((y_h == 1) & (y_a == 0), 2,
            np.where((y_h == 1) & (y_a == 1), 3, 4))))
    lh_la = lam_h * lam_a

    best_rho, best_ll = 0.0, -np.inf
    for rho in rho_grid:
        # 各样本 τ 因子（对应观测比分）
        tau = np.ones(len(lam_h))
        tau[kind == 0] = np.maximum(1.0 - lh_la[kind == 0] * rho, 1e-12)
        tau[kind == 1] = np.maximum(1.0 + lam_h[kind == 1] * rho, 1e-12)
        tau[kind == 2] = np.maximum(1.0 + lam_a[kind == 2] * rho, 1e-12)
        tau[kind == 3] = np.maximum(1.0 - rho, 1e-12)
        # 归一化常数 S
        S = (1.0
             + p00 * ((1.0 - lh_la * rho) - 1.0)
             + p01 * ((1.0 + lam_h * rho) - 1.0)
             + p10 * ((1.0 + lam_a * rho) - 1.0)
             + p11 * ((1.0 - rho) - 1.0))
        S = np.maximum(S, 1e-12)
        ll = float(np.sum(np.log(p_obs) + np.log(tau) - np.log(S)))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho


# ---------------------------------------------------------------------------
# 衍生概率
# ---------------------------------------------------------------------------
def derive_probs(grid: np.ndarray) -> dict:
    """
    从联合概率网格衍生：
    - home_win / draw / away_win 概率
    - total_goals 分布 dict{0:.., 1:.., ...}
    - 最可能比分 "i:j"
    - score_probs: { "i:j": p, ... } 全网格
    """
    n = grid.shape[0]
    ih, ja = np.indices(grid.shape)
    home_win = grid[ih > ja].sum()
    draw = grid[ih == ja].sum()
    away_win = grid[ih < ja].sum()
    total = ih + ja
    totals = {}
    for k in range(2 * n - 1):
        p = grid[total == k].sum()
        if p > 0:
            totals[int(k)] = float(p)
    # 最可能比分
    flat_idx = int(np.argmax(grid))
    i, j = divmod(flat_idx, n)
    most_likely = f"{i}:{j}"
    score_probs = {
        f"{i}:{j}": float(grid[i, j])
        for i in range(n) for j in range(n)
        if grid[i, j] > 0
    }
    return {
        "home_win_prob": float(home_win),
        "draw_prob": float(draw),
        "away_win_prob": float(away_win),
        "total_goals_distribution": totals,
        "most_likely_score": most_likely,
        "score_probs": score_probs,
    }


def match_probs(lam_h: float, lam_a: float, rho: float = 0.0,
                max_goals: int = MAX_GOALS) -> dict:
    """一键接口：λ -> Dixon-Coles 网格 -> 全部衍生概率。"""
    grid = dixon_coles_grid(lam_h, lam_a, rho, max_goals)
    return derive_probs(grid)


# ---------------------------------------------------------------------------
# 半全场（9 结果）概率
# ---------------------------------------------------------------------------
HALF_LAMBDA_RATIO = 0.45  # 半场进球 ≈ 全场的 45%（经验值）


def half_full_probs(lam_h: float, lam_a: float, rho: float = 0.0,
                    ratio: float = HALF_LAMBDA_RATIO,
                    max_goals: int = MAX_GOALS) -> dict:
    """
    半全场 9 结果概率（胜胜/平胜/负胜/胜平/平平/负平/胜负/平负/负负）。
    半场 λ ≈ ratio × 全场 λ（独立泊松近似），半场胜平负 × 全场胜平负。
    命中率不是目标，提供完整 9 结果分布供参考即可。
    """
    from scipy.stats import poisson as _poisson

    def spf(lam1: float, lam2: float) -> tuple[float, float, float]:
        kh = np.arange(max_goals + 1)
        ka = np.arange(max_goals + 1)
        g = np.outer(_poisson.pmf(kh, lam1), _poisson.pmf(ka, lam2))
        s = g.sum()
        if s <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        g /= s
        ih, ja = np.indices(g.shape)
        return (float(g[ih > ja].sum()), float(g[ih == ja].sum()),
                float(g[ih < ja].sum()))

    hw, dr, aw = spf(lam_h, lam_a)
    hh_w, hh_d, hh_l = spf(lam_h * ratio, lam_a * ratio)

    # 半场(胜/平/负) × 全场(胜/平/负) → 9 结果
    ht = [hh_w, hh_d, hh_l]
    ft = [hw, dr, aw]
    names = ["胜", "平", "负"]
    out: dict[str, float] = {}
    for i, hn in enumerate(names):
        for j, fn in enumerate(names):
            out[f"{hn}{fn}"] = float(ht[i] * ft[j])
    # 归一化（浮点误差兜底）
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    out["half_draw_prob"] = hh_d   # 半场平概率（双P 参考用）
    return out
