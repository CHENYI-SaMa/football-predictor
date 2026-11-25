# -*- coding: utf-8 -*-
"""features_v3.py —— V3 特征工程（在 v1 基础上追加已验证有效的特征）
====================================================================
v1 全部特征（滚动窗口状态/赔率/ELO/赛程）保持不变，追加：
  1. xG 族：近5/10场 xG/xGA 指数加权、xG 差、xG 效率（xG-实际进球）
     （xG 数据来自 Understat，仅五大联赛；缺失自动 -1+missing）
  2. 主客场分窗：近5场主场/客场分别统计进球、失球（v1 混合主客场的修正）
  3. 近期趋势：近8场积分斜率、进球斜率（线性回归封闭解）
  4. 情境：周中/周末（weekday≥4=周末）
  5. 交互：让球数 × ELO 差（ahh_elo_inter）
全部滚动窗口、严格赛前数据（无未来泄漏），缺失 -1 + *_missing 指示列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureBuilder
from .feature_derived import FORM_NAMES, recent_form_feats


def _ewma(rows: list, date, decay: float = 0.94) -> float:
    """rows: [(date, value)]，指数加权均值（权重 = decay^天数差）。"""
    vals = []
    wsum = wtot = 0.0
    for (d, v) in rows:
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        w = decay ** max((date - d).days, 0)
        wsum += w * float(v)
        wtot += w
    return wsum / wtot if wtot > 0 else np.nan


def _slope(rows: list) -> float:
    """rows: [(date, value)] → 线性回归斜率（封闭解，按天数标准化）。"""
    if len(rows) < 3:
        return np.nan
    xs = np.array([(d - rows[0][0]).days for (d, _) in rows], dtype=float)
    ys = np.array([v for (_, v) in rows], dtype=float)
    if np.std(xs) == 0:
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


class FeatureBuilderV3(FeatureBuilder):
    """V3：继承 v1，追加 xG/主客场/趋势/情境/交互/排名特征。"""

    def _precompute_ranks(self) -> dict:
        """预计算每场比赛前两队在其联赛的积分排名（无未来泄漏）。
        返回 {(league, team, date): rank, ...}，rank 从 1 开始（积分降序）。
        """
        pts: dict = {}            # (league, team) -> 累计积分
        gp: dict = {}             # (league, team) -> 本赛季已赛场次（轮次近似）
        rank_map: dict = {}
        # 联赛积分排名只在同一联赛内比较
        from collections import defaultdict
        league_teams: dict = defaultdict(set)
        df = self.data.sort_values("date")
        for _, row in df.iterrows():
            lg, h, a, date = row["league"], row["home_team"], row["away_team"], row["date"]
            for t in (h, a):
                league_teams[(lg, t)].add(lg)
            # 计算当前排名（该联赛内所有队按累计积分）
            if lg in pts or True:
                def rank_of(team):
                    # 该联赛内积分排序
                    if (lg, team) not in pts:
                        return len([1 for (l, t) in pts if l == lg]) + 1
                    my_pts = pts[(lg, team)]
                    return 1 + sum(1 for (l, t), p in pts.items()
                                   if l == lg and p > my_pts)
                rank_map[(lg, h, date)] = rank_of(h)
                rank_map[(lg, a, date)] = rank_of(a)
                rank_map[(lg, h, date, "gp")] = gp.get((lg, h), 0)
                rank_map[(lg, a, date, "gp")] = gp.get((lg, a), 0)
            # 更新积分（赛后）
            fh, fa = row["fthg"], row["ftag"]
            if pd.notna(fh) and pd.notna(fa):
                ph = 3 if fh > fa else (1 if fh == fa else 0)
                pa = 3 if fa > fh else (1 if fa == fh else 0)
                pts[(lg, h)] = pts.get((lg, h), 0) + ph
                pts[(lg, a)] = pts.get((lg, a), 0) + pa
                gp[(lg, h)] = gp.get((lg, h), 0) + 1
                gp[(lg, a)] = gp.get((lg, a), 0) + 1
        return rank_map

    def build(self, state: dict | None = None):
        self._rank_map = self._precompute_ranks()
        return super().build(state)

    @staticmethod
    def _new_hist() -> dict:
        d = FeatureBuilder._new_hist()
        d["xg"] = []       # (date, xg_for, xg_against)
        d["home"] = []     # (date, is_home 1/0)
        d["hc"] = []       # (date, 角球)
        d["hf"] = []       # (date, 犯规)
        return d

    def _v3_feats(self, h: dict, a: dict, row: pd.Series) -> dict:
        fvals: dict = {}
        date = row["date"]
        nan = np.nan

        # ---------- 1. xG 族 ----------
        for side, hist, opp in (("home", h, a), ("away", a, h)):
            xg_f = [(d, v) for (d, v, _) in hist["xg"]]
            xg_a = [(d, v) for (d, _, v) in hist["xg"]]
            x5 = _ewma(xg_f[-5:], date)
            x10 = _ewma(xg_f[-10:], date)
            xga5 = _ewma(xg_a[-5:], date)
            xga10 = _ewma(xg_a[-10:], date)
            fvals[f"{side}_xg_5"] = x5
            fvals[f"{side}_xg_10"] = x10
            fvals[f"{side}_xga_5"] = xga5
            fvals[f"{side}_xga_10"] = xga10
            fvals[f"{side}_xg_diff"] = (x5 - xga5) if np.isfinite(x5) and np.isfinite(xga5) else nan
            # xG 效率：近5场 xG 均值 - 实际进球均值
            gf5 = [v for (_, v) in hist["gf"][-5:]]
            if xg_f and gf5:
                g = np.mean([v for v in gf5 if v is not None])
                fvals[f"{side}_xg_eff"] = (np.mean([v for (_, v) in xg_f[-5:]]) - g) if np.isfinite(g) else nan
            else:
                fvals[f"{side}_xg_eff"] = nan

        # ---------- 2. 主客场分窗（近5场 主场/客场 进球、失球） ----------
        for side, hist in (("home", h), ("away", a)):
            home_gf, home_ga, away_gf, away_ga = [], [], [], []
            # gf/ga/home 三个列表按场次顺序对齐（同场次）
            pairs = [(gf_d, gf, ga, hm)
                     for (gf_d, gf), (_, ga), (_, hm) in
                     zip(hist["gf"], hist["ga"], hist["home"])]
            for (d, gf, ga, hm) in pairs[-5:]:
                if hm == 1:
                    home_gf.append(gf); home_ga.append(ga)
                else:
                    away_gf.append(gf); away_ga.append(ga)
            fvals[f"{side}_hm_gf5"] = float(np.mean(home_gf)) if home_gf else nan
            fvals[f"{side}_hm_ga5"] = float(np.mean(home_ga)) if home_ga else nan
            fvals[f"{side}_aw_gf5"] = float(np.mean(away_gf)) if away_gf else nan
            fvals[f"{side}_aw_ga5"] = float(np.mean(away_ga)) if away_ga else nan

        # ---------- 3. 趋势斜率（近8场积分/进球） ----------
        for side, hist in (("home", h), ("away", a)):
            fvals[f"{side}_trend_pts8"] = _slope([(d, v) for (d, v) in hist["pts"][-8:]])
            fvals[f"{side}_trend_gf8"] = _slope([(d, v) for (d, v) in hist["gf"][-8:]])

        # ---------- 4. 周中/周末 ----------
        fvals["is_weekend"] = 1.0 if date.weekday() >= 4 else 0.0

        # ---------- 5. 让球 × ELO 交互 ----------
        try:
            ahh = float(row.get("ahh", 0) or 0)
        except (TypeError, ValueError):
            ahh = 0.0
        fvals["ahh_elo_inter"] = ahh * (h["elo"] - a["elo"])

        # ---------- 6. 联赛排名 / 积分差 / 轮次（预计算 rank_map 查表） ----------
        try:
            rm = getattr(self, "_rank_map", None)
            if rm:
                lg, date = row["league"], row["date"]
                hr = rm.get((lg, row["home_team"], date))
                ar = rm.get((lg, row["away_team"], date))
                fvals["home_rank"] = float(hr) if hr is not None else nan
                fvals["away_rank"] = float(ar) if ar is not None else nan
                if hr is not None and ar is not None:
                    fvals["rank_diff"] = float(ar - hr)  # >0 客队排名更靠后
                else:
                    fvals["rank_diff"] = nan
                hg_ = rm.get((lg, row["home_team"], date, "gp"))
                ag_ = rm.get((lg, row["away_team"], date, "gp"))
                fvals["home_gp"] = float(hg_) if hg_ is not None else nan
                fvals["away_gp"] = float(ag_) if ag_ is not None else nan
                if hg_ is not None and ag_ is not None:
                    fvals["gp_diff"] = float(hg_ - ag_)
                else:
                    fvals["gp_diff"] = nan
        except Exception:
            fvals["home_rank"] = fvals["away_rank"] = fvals["rank_diff"] = nan
            fvals["home_gp"] = fvals["away_gp"] = fvals["gp_diff"] = nan

        # ---------- 7. 角球 / 犯规（近5场均值，football-data 五大联赛） ----------
        for side, hist, row_h, row_a in (("home", h, "hc_h", "hf_h"),
                                         ("away", a, "hc_a", "hf_a")):
            hc5 = [v for (_, v) in hist["hc"][-5:] if pd.notna(v)]
            hf5 = [v for (_, v) in hist["hf"][-5:] if pd.notna(v)]
            fvals[f"{side}_hc5"] = float(np.mean(hc5)) if hc5 else nan
            fvals[f"{side}_hf5"] = float(np.mean(hf5)) if hf5 else nan

        # ---------- 8. 大小球盘口（本场 2.5 线，football-data；训练有/预测缺失） ----------
        try:
            ou_o = float(row.get("ou_over")) if pd.notna(row.get("ou_over")) else nan
        except (TypeError, ValueError):
            ou_o = nan
        try:
            ou_u = float(row.get("ou_under")) if pd.notna(row.get("ou_under")) else nan
        except (TypeError, ValueError):
            ou_u = nan
        fvals["ou_over"] = ou_o
        fvals["ou_under"] = ou_u
        if np.isfinite(ou_o) and np.isfinite(ou_u) and ou_o > 1 and ou_u > 1:
            po, pu = 1.0 / ou_o, 1.0 / ou_u
            fvals["ou_implied_over"] = po / (po + pu)
        else:
            fvals["ou_implied_over"] = nan

        # ---------- 8. 近期抢分率 / 净胜球滚动均值（替代主教练胜率，按场次衰减加权） ----------
        fvals.update(recent_form_feats(h, a))

        return fvals

    def _process_row(self, team_hist: dict, h2h_hist: dict, row: pd.Series,
                     update_state: bool = True) -> dict:
        """先算 V3 特征（本场更新前，无泄漏），再走父类（基础特征+状态更新）。"""
        h = team_hist.setdefault(row["home_team"], self._new_hist())
        a = team_hist.setdefault(row["away_team"], self._new_hist())
        v3 = self._v3_feats(h, a, row)
        fvals = super()._process_row(team_hist, h2h_hist, row,
                                     update_state=update_state)
        fvals.update(v3)
        if update_state:
            # 父类不记录 xG/主客场，这里补充（赛后才更新，特征已用赛前数据）
            try:
                xg_h = float(row.get("xg_h")) if pd.notna(row.get("xg_h")) else np.nan
            except (TypeError, ValueError):
                xg_h = np.nan
            try:
                xg_a = float(row.get("xg_a")) if pd.notna(row.get("xg_a")) else np.nan
            except (TypeError, ValueError):
                xg_a = np.nan
            h["xg"].append((row["date"], xg_h, xg_a))
            a["xg"].append((row["date"], xg_a, xg_h))
            h["home"].append((row["date"], 1.0))
            a["home"].append((row["date"], 0.0))
            # 角球/犯规（football-data 附加列）
            for side, hh, row_c, row_f in (("h", h, "hc_h", "hf_h"),
                                           ("a", a, "hc_a", "hf_a")):
                try:
                    vc = float(row.get(row_c)) if pd.notna(row.get(row_c)) else np.nan
                except (TypeError, ValueError):
                    vc = np.nan
                try:
                    vf = float(row.get(row_f)) if pd.notna(row.get(row_f)) else np.nan
                except (TypeError, ValueError):
                    vf = np.nan
                hh["hc"].append((row["date"], vc))
                hh["hf"].append((row["date"], vf))
        return fvals

    def _feature_names(self) -> list[str]:
        names = super()._feature_names()
        for side in ("home", "away"):
            names += [f"{side}_xg_5", f"{side}_xg_10", f"{side}_xga_5",
                      f"{side}_xga_10", f"{side}_xg_diff", f"{side}_xg_eff"]
            names += [f"{side}_hm_gf5", f"{side}_hm_ga5",
                      f"{side}_aw_gf5", f"{side}_aw_ga5"]
            names += [f"{side}_trend_pts8", f"{side}_trend_gf8"]
        names += ["is_weekend", "ahh_elo_inter"]
        names += ["home_rank", "away_rank", "rank_diff"]
        names += ["home_gp", "away_gp", "gp_diff"]
        names += ["home_hc5", "home_hf5", "away_hc5", "away_hf5"]
        names += ["ou_over", "ou_under", "ou_implied_over"]
        names += FORM_NAMES
        return names

    @staticmethod
    def build_single(state: dict, row: pd.Series) -> dict:
        """用已有增量状态计算单场特征（预测时用，不更新状态）。"""
        fb = FeatureBuilderV3.__new__(FeatureBuilderV3)
        fvals = fb._process_row(state["team_hist"], state["h2h_hist"], row,
                                update_state=False)
        out = {}
        for k in fb._feature_names():
            v = fvals.get(k, np.nan)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                out[k] = -1.0
                out[k + "_missing"] = 1.0
            else:
                out[k] = float(v)
                out[k + "_missing"] = 0.0
        return out
