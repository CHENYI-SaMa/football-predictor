# -*- coding: utf-8 -*-
"""
评估模块（对应任务书 4.4 节）
============================
指标：
1. 赛果（主胜/平/客胜）：
   - Log Loss（三分类交叉熵）
   - Brier Score（多分类 Brier = Σ_c (p_c - y_c)² / N）
2. 比分预测：每个样本真实比分处对数概率的均值（越高越好）
3. 进球数预测：
   - 主/客进球 MAE
   - 预测区间覆盖率：80% 预测区间包含真实进球的样本占比
4. 基线对比：赔率隐含概率（去抽水）的 Log Loss / Brier

所有函数输入统一为：
    y_h, y_a : 真实主/客进球数 (array)
    grid_pred: 联合概率网格数组, shape (N, 11, 11) 或 None
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from .poisson_utils import MAX_GOALS


# ---------------------------------------------------------------------------
# 赛果指标
# ---------------------------------------------------------------------------
def outcome_onehot(y_h: np.ndarray, y_a: np.ndarray) -> np.ndarray:
    """真实赛果 one-hot：[主胜, 平, 客胜]。(N, 3)"""
    y_h = np.asarray(y_h)
    y_a = np.asarray(y_a)
    out = np.zeros((len(y_h), 3))
    out[y_h > y_a, 0] = 1.0
    out[y_h == y_a, 1] = 1.0
    out[y_h < y_a, 2] = 1.0
    return out


def outcome_probs_from_grid(grid_pred: np.ndarray) -> np.ndarray:
    """由联合概率网格求赛果概率矩阵 (N, 3)。"""
    ih, ja = np.indices((grid_pred.shape[1], grid_pred.shape[2]))
    ph = grid_pred[:, ih > ja].sum(axis=1)
    pd_ = grid_pred[:, ih == ja].sum(axis=1)
    pa = grid_pred[:, ih < ja].sum(axis=1)
    return np.stack([ph, pd_, pa], axis=1)


def log_loss_outcome(y_h: np.ndarray, y_a: np.ndarray, grid_pred: np.ndarray) -> float:
    """三分类赛果 Log Loss。"""
    p = np.clip(outcome_probs_from_grid(grid_pred), 1e-12, 1.0)
    y = outcome_onehot(y_h, y_a)
    return float(-np.mean(np.sum(y * np.log(p), axis=1)))


def brier_outcome(y_h: np.ndarray, y_a: np.ndarray, grid_pred: np.ndarray) -> float:
    """多分类 Brier Score（越低越好）。"""
    p = outcome_probs_from_grid(grid_pred)
    y = outcome_onehot(y_h, y_a)
    return float(np.mean(np.sum((p - y) ** 2, axis=1)))


def accuracy_outcome(y_h: np.ndarray, y_a: np.ndarray, grid_pred: np.ndarray) -> float:
    """赛果命中率（参考指标）。"""
    pred = np.argmax(outcome_probs_from_grid(grid_pred), axis=1)
    y = np.argmax(outcome_onehot(y_h, y_a), axis=1)
    return float(np.mean(pred == y))


# ---------------------------------------------------------------------------
# 比分指标
# ---------------------------------------------------------------------------
def mean_log_score(y_h: np.ndarray, y_a: np.ndarray, grid_pred: np.ndarray) -> float:
    """真实比分处对数概率均值（越高越好）。"""
    y_h = np.clip(np.asarray(y_h, dtype=int), 0, MAX_GOALS)
    y_a = np.clip(np.asarray(y_a, dtype=int), 0, MAX_GOALS)
    logp = np.log(np.clip(grid_pred[y_h, y_a], 1e-12, 1.0))
    return float(np.mean(logp))


# ---------------------------------------------------------------------------
# 进球数指标
# ---------------------------------------------------------------------------
def goals_mae(y_true: np.ndarray, lam_pred: np.ndarray) -> float:
    """进球数 MAE（λ 作为点预测）。"""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(lam_pred))))


def interval_coverage(y_true: np.ndarray, lam_pred: np.ndarray,
                      level: float = 0.8) -> tuple[float, float]:
    """
    泊松 80% 预测区间覆盖率：对每个样本找最小区间 [a, b] 使
    P(a ≤ X ≤ b | λ) ≥ level，统计真实进球落入区间的比例。

    Returns
    -------
    (覆盖率, 平均区间宽度)
    """
    y = np.asarray(y_true, dtype=float)
    lam = np.clip(np.asarray(lam_pred, dtype=float), 0.05, None)
    covered, widths = [], []
    for yi, li in zip(y, lam):
        cdf = poisson.cdf(np.arange(30), li)
        best_w, best_a, best_b, best_ok = 100, -1, -1, False
        # 按宽度递增搜索最小预测区间
        for w in range(0, 20):
            for a in range(0, 30 - w):
                b = a + w
                if cdf[b] - (cdf[a - 1] if a > 0 else 0.0) >= level:
                    best_w, best_a, best_b, best_ok = w, a, b, True
                    break
            if best_ok:
                break
        covered.append(1.0 if best_ok and best_a <= yi <= best_b else 0.0)
        widths.append(best_w)
    return float(np.mean(covered)), float(np.mean(widths))


# ---------------------------------------------------------------------------
# 赔率基线
# ---------------------------------------------------------------------------
def odds_baseline_probs(odds_h: np.ndarray, odds_d: np.ndarray,
                        odds_a: np.ndarray) -> np.ndarray:
    """赔率隐含概率（Basic 去抽水），返回 (N, 3)。"""
    inv = 1.0 / np.stack([np.asarray(odds_h), np.asarray(odds_d), np.asarray(odds_a)], axis=1)
    return inv / inv.sum(axis=1, keepdims=True)


def evaluate(y_h: np.ndarray, y_a: np.ndarray, grid_pred: np.ndarray) -> dict:
    """一键汇总所有模型指标。"""
    return {
        "log_loss": log_loss_outcome(y_h, y_a, grid_pred),
        "brier": brier_outcome(y_h, y_a, grid_pred),
        "accuracy": accuracy_outcome(y_h, y_a, grid_pred),
        "mean_log_score": mean_log_score(y_h, y_a, grid_pred),
    }


def evaluate_all(y_h: np.ndarray, y_a: np.ndarray, grid_pred: np.ndarray,
                 lam_h: np.ndarray | None = None, lam_a: np.ndarray | None = None) -> dict:
    """含进球指标的完整评估。"""
    res = evaluate(y_h, y_a, grid_pred)
    if lam_h is not None:
        res["goals_h_mae"] = goals_mae(y_h, lam_h)
        res["goals_a_mae"] = goals_mae(y_a, lam_a)
        cov_h, wid_h = interval_coverage(y_h, lam_h)
        cov_a, wid_a = interval_coverage(y_a, lam_a)
        res["interval_cov_h"] = cov_h
        res["interval_cov_a"] = cov_a
        res["interval_width_h"] = wid_h
        res["interval_width_a"] = wid_a
    return res
