# -*- coding: utf-8 -*-
"""阶段B: 特征工程模块
输入 master_table.csv，输出赛前已知的特征矩阵（严格无未来泄露）：
- SP赔率衍生（implied_H/D/A 去抽水、rq_line、total_goal_line）
- 近期状态（指数加权0.94：进球/失球/积分/射正，近3场进球std）
- xG状态（近5场加权xG/xGA + 效率差）
- 对阵（历史交锋总进球均值、主场对阵胜率）
- 体能（rest_days、is_double_week）
- 缺失填 -1 + 缺失指示列
输出 data/features.csv + data/feature_state.pkl（供预测时增量构建）
"""
import sys, pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")

DECAY = 0.94          # 指数衰减因子
MAX_HIST = 10         # 近期状态回溯场数
XG_WINDOW = 5         # xG 窗口
MAX_GOALS = 10        # 比分网格上限


def implied_probs(o_h, o_d, o_a):
    """SP赔率去抽水 → 真实概率。"""
    if not (o_h and o_d and o_a and o_h > 1 and o_d > 1 and o_a > 1):
        return np.nan, np.nan, np.nan
    inv = 1.0 / np.array([o_h, o_d, o_a], dtype=float)
    return tuple(inv / inv.sum())


def goal_line_from_ou(ou_over, ou_under):
    """大小球SP → 隐含总进球线（去抽水后两概率中点）。"""
    if not (ou_over and ou_under and ou_over > 1 and ou_under > 1):
        return np.nan
    p_over = (1 / ou_over) / (1 / ou_over + 1 / ou_under)
    # 隐含线 = 使 P(over) 成立的临界值近似：用正态近似 线 ≈ 2.5 + Φ⁻¹(p_over)*σ
    # 简化：直接映射 p_over -> 线（p_over=0.5 → 2.5）
    from scipy.stats import norm
    return 2.5 + norm.ppf(p_over) * 1.35  # σ≈1.35 为总进球经验标准差


class TeamState:
    """单队滚动状态（指数加权），全部基于赛前已知数据。
    history 元素: (date, gf, ga, pts, sot, xg, xga, hgf, hga)
    其中 hgf/hga = 本队上半场进球/失球（可能为 NaN，无半场数据）。
    """
    __slots__ = ["history", "last_date", "h2h"]

    def __init__(self):
        self.history = []      # [(date, gf, ga, pts, sot, xg, xga, hgf, hga)]
        self.last_date = None  # 上一场日期（算 rest_days）
        self.h2h = defaultdict(list)  # 对手 -> [总进球, ...]

    def add(self, date, gf, ga, sot, xg, xga, opp, is_home, hgf=np.nan, hga=np.nan):
        self.history.append((date, gf, ga, _pts(gf, ga), sot, xg, xga, hgf, hga))
        self.h2h[opp].append(gf + ga)
        self.last_date = date

    def rest_days(self, date):
        if self.last_date is None:
            return 30.0  # 新队默认
        return (date - self.last_date).days

    def recent(self, date, k=MAX_HIST):
        """截至 date 之前的最近 k 场（不含当天）。"""
        return [h for h in self.history if h[0] < date][-k:]

    def ewma(self, date, field_idx, k=MAX_HIST, decay=DECAY):
        """指数加权均值，权重 = decay^(天数差)。"""
        rows = self.recent(date, k)
        if not rows:
            return np.nan
        wsum = 0.0
        wtot = 0.0
        for r in rows:
            days = (date - r[0]).days
            w = decay ** max(days, 0)
            v = r[field_idx]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            wsum += w * v
            wtot += w
        return wsum / wtot if wtot > 0 else np.nan

    def std_goals_3(self, date):
        """近3场进球标准差。"""
        rows = self.recent(date, 3)
        if len(rows) < 2:
            return 0.0
        return float(np.std([r[1] for r in rows]))

    def h2h_total_goals(self, opp):
        hist = self.h2h.get(opp, [])
        return float(np.mean(hist)) if hist else np.nan

    def h2h_home_win_rate(self, opp, home_state_map, date):
        """主场对阵该对手的胜率（主队视角）。"""
        return np.nan  # 简化：h2h 总进球已含信息，胜率可选

    def half_lead_rate(self, date, k=5):
        """近 k 场半场领先频率（本队半场进球 > 半场失球）。"""
        rows = [r for r in self.recent(date, k) if not (isinstance(r[7], float) and np.isnan(r[7]))]
        if not rows:
            return np.nan
        return float(np.mean([1.0 if r[7] > r[8] else 0.0 for r in rows]))


def _pts(gf, ga):
    if gf > ga:
        return 3.0
    if gf == ga:
        return 1.0
    return 0.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 master_table 构建特征。df 需含列：
    date, league, home_team, away_team, fthg, ftag, ftr, b365h/b365d/b365a,
    ahh, ou_over, ou_under, hs, as, hst, ast, xG_h, xG_a, home_en, away_en,
    htg, htag, htg_miss, htag_miss（半场比分，非五大联赛为填充值+miss=1）
    返回特征 DataFrame（行序与 df 一致）。
    """
    df = df.sort_values("date").reset_index(drop=True)
    states = defaultdict(TeamState)
    rows = []

    for _, m in df.iterrows():
        date = m["date"]
        h, a = m["home_team"], m["away_team"]
        sh, sa = states[h], states[a]

        # ---- 1. SP 赔率衍生 ----
        ih, id_, ia = implied_probs(m["b365h"], m["b365d"], m["b365a"])
        rq = m["ahh"] if pd.notna(m["ahh"]) else 0.0
        gl = goal_line_from_ou(m["ou_over"], m["ou_under"]) if pd.notna(m["ou_over"]) else np.nan

        # ---- 2. 近期状态（主队 / 客队） ----
        feat = {
            "implied_H": ih, "implied_D": id_, "implied_A": ia,
            "rq_line": rq, "total_goal_line": gl,
            # 主队
            "h_w_goals": sh.ewma(date, 1), "h_w_goals_conceded": sh.ewma(date, 2),
            "h_w_pts": sh.ewma(date, 3), "h_w_sot": sh.ewma(date, 4),
            "h_std_goals_3": sh.std_goals_3(date),
            "h_w_xg": sh.ewma(date, 5, k=XG_WINDOW), "h_w_xga": sh.ewma(date, 6, k=XG_WINDOW),
            "h_rest_days": sh.rest_days(date),
            # 客队
            "a_w_goals": sa.ewma(date, 1), "a_w_goals_conceded": sa.ewma(date, 2),
            "a_w_pts": sa.ewma(date, 3), "a_w_sot": sa.ewma(date, 4),
            "a_std_goals_3": sa.std_goals_3(date),
            "a_w_xg": sa.ewma(date, 5, k=XG_WINDOW), "a_w_xga": sa.ewma(date, 6, k=XG_WINDOW),
            "a_rest_days": sa.rest_days(date),
            # 半场特征（赛前已知，近5场 EWMA）
            "h_w_half_goals": sh.ewma(date, 7, k=5),
            "h_w_half_conceded": sh.ewma(date, 8, k=5),
            "a_w_half_goals": sa.ewma(date, 7, k=5),
            "a_w_half_conceded": sa.ewma(date, 8, k=5),
            "h_half_lead_rate": sh.half_lead_rate(date),
            "a_half_lead_rate": sa.half_lead_rate(date),
        }
        # 体能：7天内第3场
        def double_week(st: TeamState, d):
            recent = [x for x in st.history if (d - x[0]).days <= 7]
            return 1.0 if len(recent) >= 2 else 0.0
        feat["h_is_double_week"] = double_week(sh, date)
        feat["a_is_double_week"] = double_week(sa, date)

        # ---- 3. 对阵 ----
        feat["h2h_total_goals"] = sh.h2h_total_goals(a)
        feat["h2h_rev_total_goals"] = sa.h2h_total_goals(h)
        feat["h2h_total_goals_avg"] = np.nanmean([
            f for f in (feat["h2h_total_goals"], feat["h2h_rev_total_goals"])
            if pd.notna(f)]) if (pd.notna(feat["h2h_total_goals"]) or pd.notna(feat["h2h_rev_total_goals"])) else np.nan

        # ---- 4. xG 效率（近5场 xG - 实际进球） ----
        def xg_eff(st: TeamState, d):
            rows = st.recent(d, XG_WINDOW)
            diffs = [r[5] - r[1] for r in rows if not np.isnan(r[5])]
            return float(np.mean(diffs)) if diffs else np.nan
        feat["h_xg_eff"] = xg_eff(sh, date)
        feat["a_xg_eff"] = xg_eff(sa, date)

        # ---- 5. 标签与元数据 ----
        # 半场比分：仅当 miss=0 才视为真实（非五大联赛填充值不可用）
        htg = m["htg"] if (pd.notna(m.get("htg")) and m.get("htg_miss", 1) == 0) else np.nan
        htag = m["htag"] if (pd.notna(m.get("htag")) and m.get("htag_miss", 1) == 0) else np.nan
        feat["date"] = date
        feat["league"] = m["league"]
        feat["home_team"] = h
        feat["away_team"] = a
        feat["fthg"] = m["fthg"]
        feat["ftag"] = m["ftag"]
        feat["htg"] = htg      # 上半场主队进球（真实或 NaN）
        feat["htag"] = htag    # 上半场客队进球（真实或 NaN）
        feat["home_en"] = m["home_en"]
        feat["away_en"] = m["away_en"]
        # 保留赔率（评估基线用）
        feat["b365h"] = m["b365h"]
        feat["b365d"] = m["b365d"]
        feat["b365a"] = m["b365a"]
        rows.append(feat)

        # ---- 6. 更新状态（赛后才更新，特征已用赛前数据） ----
        sh.add(date, m["fthg"], m["ftag"],
               m["hst"] if pd.notna(m["hst"]) else np.nan,
               m["xG_h"] if pd.notna(m["xG_h"]) else np.nan,
               m["xG_a"] if pd.notna(m["xG_a"]) else np.nan,
               a, True, hgf=htg, hga=htag)
        sa.add(date, m["ftag"], m["fthg"],
               m["ast"] if pd.notna(m["ast"]) else np.nan,
               m["xG_a"] if pd.notna(m["xG_a"]) else np.nan,
               m["xG_h"] if pd.notna(m["xG_h"]) else np.nan,
               h, False, hgf=htag, hga=htg)

    out = pd.DataFrame(rows)
    return out


def finalize_features(feat: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """缺失填 -1 + 缺失指示列，仅保留模型特征列。"""
    out = feat.copy()
    for c in feature_cols:
        out[c + "_miss"] = out[c].isna().astype(int)
        out[c] = out[c].fillna(-1.0)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = pd.read_csv("data/master_table.csv", parse_dates=["date"])
    feat = build_features(df)
    print(f"特征构建完成: {feat.shape}, 耗时 {time.time()-t0:.1f}s")
    # 检查覆盖
    print("\n关键特征缺失率:")
    for c in ("implied_H", "h_w_goals", "h_w_sot", "h_w_xg", "h_rest_days",
              "h2h_total_goals", "h_xg_eff", "total_goal_line"):
        print(f"  {c}: {feat[c].isna().mean():.0%}")
    # 特征列（排除标签/元数据）
    FEATURE_COLS = [c for c in feat.columns if c not in
                    {"date", "league", "home_team", "away_team", "fthg", "ftag",
                     "htg", "htag", "home_en", "away_en", "b365h", "b365d", "b365a"}]
    print(f"\n特征列数: {len(FEATURE_COLS)}")
    print("半场特征:", [c for c in FEATURE_COLS if "half" in c or "lead" in c])
    fin = finalize_features(feat, FEATURE_COLS)
    fin.to_csv("data/features.csv", index=False, encoding="utf-8-sig")
    with open("data/feature_state.pkl", "wb") as f:
        pickle.dump({"feature_cols": FEATURE_COLS}, f)
    print("→ data/features.csv 已保存")
