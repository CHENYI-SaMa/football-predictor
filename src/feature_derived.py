# -*- coding: utf-8 -*-
"""衍生特征模块：近期抢分率 / 净胜球滚动均值（按场次衰减加权）。

替代"主教练胜率"独立特征：教练战术调整会直接反映在球队近期
抢分能力和净胜球上，用滚动加权统计填补这部分状态波动。

权重设计（用户指定）：最近一场权重 1.0，往前每场 ×0.8。
    weight_i = 0.8 ** i ，i = 距当前场次（0=最近一场）
"""
from __future__ import annotations

import numpy as np

_DECAY = 0.8


def _weighted_mean(rows: list, window: int) -> float:
    """rows: [(date, value)]（按时间从旧到新），取最近 window 场做衰减加权均值。

    权重按场次位置：最新一场 1.0，往前每场 ×0.8。
    返回 nan 当无有效值。
    """
    recent = rows[-window:]
    wsum = wtot = 0.0
    n = len(recent)
    for i, (_, v) in enumerate(recent):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        w = _DECAY ** (n - 1 - i)   # 最新一场权重 0.8^0 = 1.0
        wsum += w * float(v)
        wtot += w
    return wsum / wtot if wtot > 0 else np.nan


def recent_form_feats(h: dict, a: dict) -> dict:
    """计算主客两队开赛前的加权抢分率与净胜球滚动均值。

    返回特征（缺失用 nan）：
        home_form5_pts_rate / home_form10_pts_rate   加权抢分率（场均积分）
        home_form5_gd     / home_form10_gd           加权净胜球均值
        away_form5_pts_rate / away_form10_pts_rate
        away_form5_gd     / away_form10_gd
    """
    fvals: dict = {}
    nan = np.nan

    for side, hist in (("home", h), ("away", a)):
        pts = hist.get("pts", [])      # [(date, 3/1/0)]
        gf = hist.get("gf", [])        # [(date, 进球)]
        ga = hist.get("ga", [])        # [(date, 失球)]

        # 净胜球序列
        gd = []
        for (d, g), (_, ga_v) in zip(gf, ga):
            gd.append((d, g - ga_v))

        for w in (5, 10):
            fvals[f"{side}_form{w}_pts_rate"] = _weighted_mean(pts, w)
            fvals[f"{side}_form{w}_gd"] = _weighted_mean(gd, w)

    return fvals


# 特征名清单（供 feature_names 拼接）
FORM_NAMES = [
    f"{side}_{kind}"
    for side in ("home", "away")
    for kind in ("form5_pts_rate", "form10_pts_rate", "form5_gd", "form10_gd")
]
