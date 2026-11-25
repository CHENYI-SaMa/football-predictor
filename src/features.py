# -*- coding: utf-8 -*-
"""
特征工程模块
============
为每场比赛构建如下特征向量（对应任务书 4.1 节）：

A. 基础实力 (strength)
   - 主/客队过去 12 个月 / 6 个月 / 3 个月的平均进球、失球、射正(xG 代理)
   - 使用指数加权（半衰期约 90 天），越近的比赛权重越高

B. 近期状态 (form)
   - 最近 3 / 5 / 8 场比赛：场均积分、场均进球、场均失球、赢盘率
   - 指数加权滚动窗口，权重按比赛新旧指数衰减

C. 交锋记录 (h2h)
   - 历史交锋（同主客场地）的场均进球、胜平负占比

D. 赔率特征 (odds)
   - 初始主/平/客赔率，隐含概率（用 Basic 去抽水法反推）、大小球盘口、
     让球盘口、主客赔率对数比

E. 赛程体能 (schedule)
   - 主/客队本场比赛距上一场的休息天数、主队是否一周双赛、
     客队连续客场场次

缺失值处理：数值特征用 -1 填充，并额外生成 *_missing 二值指示特征。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

# 半衰期天数：指数衰减权重 w = 0.5 ** (age_days / half_life)
_HALF_LIFE = 90.0

# 去抽水（Basic method）：p_i' = (1/odds_i) / Σ(1/odds_j)
def implied_prob_debooked(h: float, d: float, a: float) -> tuple[float, float, float]:
    """由三项赔率反推去抽水后的隐含概率（Basic 法）。"""
    inv = np.array([1.0 / h, 1.0 / d, 1.0 / a], dtype=float)
    s = inv.sum()
    if s <= 0 or not np.isfinite(s):
        return 1 / 3, 1 / 3, 1 / 3
    return tuple(inv / s)


def _exp_weights(ages_days: np.ndarray, half_life: float = _HALF_LIFE) -> np.ndarray:
    """按比赛天数年龄计算指数衰减权重（越新权重越高）。"""
    return 0.5 ** (np.asarray(ages_days, dtype=float) / half_life)


class FeatureBuilder:
    """
    顺序构建特征：先按日期排序，对每一行只使用『该比赛之前』的信息，
    保证无未来数据泄漏。使用增量游标（每组球队最后出现位置）实现 O(N)。
    """

    def __init__(self, data: pd.DataFrame):
        # 输入必须已按日期升序
        self.data = data.reset_index(drop=True)
        self.rows = []

    # ---------------------------------------------------------------- 核心循环
    def build(self, state: dict | None = None) -> tuple[pd.DataFrame, dict]:
        """
        返回 (特征 DataFrame, 增量状态 state)。

        state : dict, 可选
            若传入，则在已有历史状态上继续构建（用于 predict 阶段增量扩展），
            否则从空状态开始。

        特征 DataFrame 与 self.data 行一一对应，包含 fthg/ftag 标签。
        """
        df = self.data
        n = len(df)
        # 预分配容器
        feats = {k: np.full(n, np.nan) for k in self._feature_names()}
        # 每队历史记录（按日期追加）：用于滚动统计
        team_hist: dict[str, dict] = {}
        # h2h 记录: (min(home,away), max(home,away)) -> list of dicts (日期, 进球, 结果)
        h2h_hist: dict[tuple, list] = {}
        if state is not None:
            team_hist = state.get("team_hist", {})
            h2h_hist = state.get("h2h_hist", {})

        # 进度条：构建特征（每场比赛处理一组滚动统计）
        for idx in tqdm(range(n), desc="构建特征", unit="场",
                        ncols=90, disable=n < 500):
            row = df.iloc[idx]
            fvals = self._process_row(team_hist, h2h_hist, row)
            for k, v in fvals.items():
                feats[k][idx] = v

        feat_df = pd.DataFrame(feats)
        final = self._finalize(feat_df, df)
        return final, {"team_hist": team_hist, "h2h_hist": h2h_hist}

    # ---------------------------------------------------------------- 单行处理
    def _process_row(self, team_hist: dict, h2h_hist: dict, row: pd.Series,
                     update_state: bool = True) -> dict:
        """对单场比赛计算特征；update_state=False 时不写入历史（纯预测）。"""
        date = row["date"]
        home, away = row["home_team"], row["away_team"]
        fthg, ftag = row["fthg"], row["ftag"]

        h = team_hist.setdefault(home, self._new_hist())
        a = team_hist.setdefault(away, self._new_hist())

        fvals = {}
        # ---------- A. 基础实力（近 365/180/90 天指数加权均值） ----------
        for team, side in ((h, "home"), (a, "away")):
            self._strength_feats(fvals, side, team, date)
        # ---------- B. 近期状态（近 3/5/8 场指数加权） ----------
        for team, side in ((h, "home"), (a, "away")):
            self._form_feats(fvals, side, team)
        # ---------- C. 交锋记录 ----------
        self._h2h_feats(fvals, h2h_hist, home, away, date)
        # ---------- D. 赔率特征 ----------
        self._odds_feats(fvals, row)
        # ---------- D2. 联赛级别 + 净攻防强度（赔率之外的补充信号） ----------
        league_map = {"E0": 0.0, "SP1": 1.0, "D1": 2.0, "I1": 3.0, "F1": 4.0}
        fvals["league_idx"] = league_map.get(row.get("league", np.nan), np.nan)
        for key, col_a, col_b in (("net_att_h", "home_gf_365d", "away_ga_365d"),
                                  ("net_att_a", "away_gf_365d", "home_ga_365d")):
            va, vb = fvals.get(col_a, np.nan), fvals.get(col_b, np.nan)
            if np.isfinite(va) and np.isfinite(vb):
                fvals[key] = va - vb  # 主队进攻-客队防守（净强度）
            else:
                fvals[key] = np.nan
        # ---------- E. 赛程体能 ----------
        self._schedule_feats(fvals, h, a, date)

        # ---------- F. 动态 ELO 评分（独立于赔率的球队实力评级） ----------
        fvals["home_elo"] = h["elo"]
        fvals["away_elo"] = a["elo"]
        fvals["elo_diff"] = h["elo"] - a["elo"]
        fvals["home_elo_rd"] = h["rd"]
        fvals["away_elo_rd"] = a["rd"]
        fvals["elo_rd_diff"] = h["rd"] - a["rd"]

        if update_state:
            # ---------- 更新历史 ----------
            h["gf"].append((date, fthg))
            h["ga"].append((date, ftag))
            h["pts"].append((date, 3 if fthg > ftag else (1 if fthg == ftag else 0)))
            h["sot"].append((date, row.get("hst", np.nan)))
            a["gf"].append((date, ftag))
            a["ga"].append((date, fthg))
            a["pts"].append((date, 3 if ftag > fthg else (1 if ftag == fthg else 0)))
            a["sot"].append((date, row.get("ast", np.nan)))
            # 赢盘率（亚洲让球）：主队赢盘 = fthg + ahh > ftag
            ahh = row.get("ahh", np.nan)
            if np.isfinite(ahh):
                h["cover"].append((date, 1.0 if fthg + ahh > ftag else 0.0))
                a["cover"].append((date, 1.0 if ftag - ahh > fthg else 0.0))
            else:
                h["cover"].append((date, np.nan))
                a["cover"].append((date, np.nan))
            # 连续客场
            a["away_streak"] += 1
            h["away_streak"] = 0
            # h2h
            key = tuple(sorted((home, away)))
            h2h_hist.setdefault(key, []).append(
                {"date": date, "home_win": 1.0 if fthg > ftag else 0.0,
                 "draw": 1.0 if fthg == ftag else 0.0,
                 "away_win": 1.0 if fthg < ftag else 0.0,
                 "goals": fthg + ftag, "home_goals": fthg})
            # ELO 更新（Glicko 贝叶斯更新：评分+不确定性RD，信息量自适应）
            # 大冷门（预期与实际差大）→ 更新幅度大；平淡结果 → 更新幅度小
            home_adv = 60.0
            act_h = 1.0 if fthg > ftag else (0.5 if fthg == ftag else 0.0)
            act_a = 1.0 - act_h
            h0_elo, h0_rd, a0_elo, a0_rd = h["elo"], h["rd"], a["elo"], a["rd"]
            h["elo"], h["rd"] = self._glicko_step(h0_elo, h0_rd, a0_elo, a0_rd,
                                                  act_h, home_adv)
            a["elo"], a["rd"] = self._glicko_step(a0_elo, a0_rd, h0_elo, h0_rd,
                                                  act_a, 0.0)
        return fvals

    @staticmethod
    def _glicko_step(r_a: float, rd_a: float, r_b: float, rd_b: float,
                     score_a: float, home_adv: float) -> tuple[float, float]:
        """Glicko-1 一步更新：返回 (新评分, 新RD)。

        score_a ∈ {1.0, 0.5, 0.0}（A 视角：胜/平/负）。
        预期胜率用对手 RD 修正；更新幅度 = q / (1/RD² + 1/d²) × g(RD) × (score-E)，
        信息量越大（d² 越大）修正越大——即贝叶斯式"用证据修正信念"。
        """
        q = np.log(10.0) / 400.0
        g_rd = 1.0 / np.sqrt(1.0 + 3.0 * q * q * rd_b * rd_b / (np.pi * np.pi))
        e = 1.0 / (1.0 + 10.0 ** (-g_rd * (r_a + home_adv - r_b) / 400.0))
        d2 = 1.0 / (q * q * g_rd * g_rd * e * (1.0 - e))
        r_new = r_a + q / (1.0 / (rd_a * rd_a) + 1.0 / d2) * g_rd * (score_a - e)
        rd_new = np.sqrt(1.0 / (1.0 / (rd_a * rd_a) + 1.0 / d2))
        return r_new, rd_new

    # ---------------------------------------------------------------- 收尾
    def _finalize(self, feat_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """缺失值 -1 填充 + 缺失指示特征 + 附加标签/标识（一次性 concat 避免碎片化）。"""
        cols, miss_cols = {}, {}
        for col in feat_df.columns:
            miss = feat_df[col].isna() | np.isinf(feat_df[col])
            cols[col] = feat_df[col].where(~miss, -1.0)
            if miss.any():
                miss_cols[col + "_missing"] = miss.astype(float)
        out = pd.concat([pd.DataFrame(cols), pd.DataFrame(miss_cols)], axis=1)
        out["fthg"] = df["fthg"].values
        out["ftag"] = df["ftag"].values
        out["date"] = df["date"].values
        out["league"] = df["league"].values
        out["home_team"] = df["home_team"].values
        out["away_team"] = df["away_team"].values
        return out

    @staticmethod
    def build_single(state: dict, row: pd.Series) -> dict:
        """
        用已有增量状态计算单场特征（不更新状态、不写标签），用于 predict.py。
        返回缺失处理后的特征 dict（-1 填充 + *_missing 指示列，保证列齐全）。
        """
        fb = FeatureBuilder.__new__(FeatureBuilder)  # 跳过 __init__（不需要 self.data）
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

    @staticmethod
    def _new_hist() -> dict:
        return {
            "gf": [], "ga": [], "pts": [], "sot": [], "cover": [],
            "last_date": None, "away_streak": 0, "elo": 1500.0, "rd": 350.0,
        }

    def _feature_names(self) -> list[str]:
        names = []
        for side in ("home", "away"):
            for win in (365, 180, 90):
                names += [f"{side}_gf_{win}d", f"{side}_ga_{win}d", f"{side}_sot_{win}d"]
            for k in (3, 5, 8):
                names += [f"{side}_pts_{k}", f"{side}_gf_{k}", f"{side}_ga_{k}", f"{side}_cover_{k}"]
        names += ["h2h_home_win", "h2h_draw", "h2h_goals", "h2h_home_goals"]
        names += ["odds_h", "odds_d", "odds_a", "prob_h", "prob_d", "prob_a",
                  "log_odds_ratio", "ou_line", "ahh"]
        names += ["league_idx", "net_att_h", "net_att_a"]
        names += ["home_elo", "away_elo", "elo_diff"]
        names += ["home_elo_rd", "away_elo_rd", "elo_rd_diff"]
        names += ["home_rest_days", "away_rest_days", "away_consec_away", "home_week_double"]
        return names

    # ---------- A ----------
    def _strength_feats(self, fvals, side, hist, date):
        for win, suffix in ((365, "365d"), (180, "180d"), (90, "90d")):
            # 收集窗口内记录（按日期）
            gf, ga, sot, ages = [], [], [], []
            for (d, v) in hist["gf"]:
                age = (date - d).days
                if age <= win:
                    gf.append(v); ages.append(age)
            for (d, v) in hist["ga"]:
                if (date - d).days <= win:
                    ga.append(v)
            for (d, v) in hist["sot"]:
                if (date - d).days <= win:
                    sot.append(v)
            w = _exp_weights(np.array(ages)) if ages else np.array([])
            fvals[f"{side}_gf_{suffix}"] = np.average(gf, weights=w) if len(gf) else np.nan
            fvals[f"{side}_ga_{suffix}"] = np.average(ga, weights=w) if len(ga) else np.nan
            fvals[f"{side}_sot_{suffix}"] = np.average(sot, weights=w) if len(sot) else np.nan

    # ---------- B ----------
    def _form_feats(self, fvals, side, hist):
        gf = hist["gf"][-8:]
        ga = hist["ga"][-8:]
        pts = hist["pts"][-8:]
        cover = hist["cover"][-8:]
        # 指数权重：最近场权重 = 1，往前每场 ×0.8
        for k in (3, 5, 8):
            w = np.array([0.8 ** (k - 1 - i) for i in range(k)])
            g = np.array([v for _, v in gf[-k:]])
            g_a = np.array([v for _, v in ga[-k:]])
            p = np.array([v for _, v in pts[-k:]])
            c = np.array([v for _, v in cover[-k:]], dtype=float)
            if len(g) == k:
                fvals[f"{side}_pts_{k}"] = np.average(p, weights=w)
                fvals[f"{side}_gf_{k}"] = np.average(g, weights=w)
                fvals[f"{side}_ga_{k}"] = np.average(g_a, weights=w)
                # cover 可能含 NaN（盘口缺失），按位置过滤后重归一化权重
                valid = np.isfinite(c)
                if valid.any():
                    fvals[f"{side}_cover_{k}"] = np.average(c[valid], weights=w[valid])
                else:
                    fvals[f"{side}_cover_{k}"] = np.nan
            else:
                fvals[f"{side}_pts_{k}"] = np.nan
                fvals[f"{side}_gf_{k}"] = np.nan
                fvals[f"{side}_ga_{k}"] = np.nan
                fvals[f"{side}_cover_{k}"] = np.nan

    # ---------- C ----------
    def _h2h_feats(self, fvals, h2h_hist, home, away, date):
        key = tuple(sorted((home, away)))
        # 只取最近 10 次交锋，避免远古交手记录稀释当前实力对比
        recs = h2h_hist.get(key, [])[-10:]
        if not recs:
            return
        # 时间加权：越近的交锋权重越高（decay=0.9^天数）
        ages = np.array([max((date - r["date"]).days, 0) for r in recs], dtype=float)
        w = 0.9 ** ages
        w = w / w.sum()
        hw = float(np.sum(w * np.array([r["home_win"] for r in recs])))
        dr = float(np.sum(w * np.array([r["draw"] for r in recs])))
        gl = float(np.sum(w * np.array([r["goals"] for r in recs])))
        hg = float(np.sum(w * np.array([r["home_goals"] for r in recs])))
        fvals["h2h_home_win"] = hw
        fvals["h2h_draw"] = dr
        fvals["h2h_goals"] = gl
        fvals["h2h_home_goals"] = hg

    # ---------- D ----------
    def _odds_feats(self, fvals, row):
        h, d, a = row.get("b365h"), row.get("b365d"), row.get("b365a")
        if np.isfinite(h) and np.isfinite(d) and np.isfinite(a) and h > 1 and d > 1 and a > 1:
            fvals["odds_h"] = h
            fvals["odds_d"] = d
            fvals["odds_a"] = a
            ph, pd_, pa = implied_prob_debooked(h, d, a)
            fvals["prob_h"] = ph
            fvals["prob_d"] = pd_
            fvals["prob_a"] = pa
            fvals["log_odds_ratio"] = np.log(ph / pa)
        ov = row.get("ou_over")
        if np.isfinite(ov) and ov > 1:
            fvals["ou_line"] = ov
        ahh = row.get("ahh")
        if np.isfinite(ahh):
            fvals["ahh"] = ahh

    # ---------- E ----------
    def _schedule_feats(self, fvals, h, a, date):
        if h["last_date"] is not None:
            fvals["home_rest_days"] = (date - h["last_date"]).days
        if a["last_date"] is not None:
            fvals["away_rest_days"] = (date - a["last_date"]).days
        fvals["away_consec_away"] = a["away_streak"]
        if h["last_date"] is not None:
            # 一周双赛：距上一场 <= 3 天
            fvals["home_week_double"] = 1.0 if (date - h["last_date"]).days <= 3 else 0.0
        # 更新 last_date（在主队视角下，home 队打完本场）
        h["last_date"] = date


def build_features(data: pd.DataFrame, state: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """便捷入口：构建特征 + 标签，返回 (特征表, 增量状态)。"""
    return FeatureBuilder(data).build(state)
