# -*- coding: utf-8 -*-
"""train.py — 足球赛果预测模型完整训练流程（阶段A-E）
====================================================
基于现有数据构建让球胜平负预测模型（方案A：评估维度=竞彩让球胜平负）。

阶段A 数据融合与清洗：
  - A-1 队名映射：team_mapping.csv（中文→英文，已人工核对，121/121）
  - A-2 主训练表：jczq_history 左连接 football_history(半场/射门/欧赔) + understat(xG)
         无法连接用联赛均值填充 + 缺失指示列；按 日期-主客队-联赛 去重
阶段B 特征工程（严格赛前已知，无未来泄露）：
  - SP赔率衍生 implied_H/D/A（去抽水）、rq_line、total_goal_line
  - 近期状态 EWMA（0.94^days 衰减，近10场：进球/失球/积分/射正 + 近3场进球std）
  - xG 状态（近5场加权 xG/xGA + 效率差）
  - 对阵 h2h 总进球均值；体能 rest_days / is_double_week
  - 缺失填 -1 + 缺失指示列
阶段C 模型训练：
  - LightGBM 双输出泊松（objective='poisson' 即 softplus 参数化 λ）
  - Optuna 调优 num_leaves/learning_rate/min_child_samples/feature_fraction
  - isotonic 校准 λ_h、λ_a
阶段D 比分概率：独立泊松网格 + Dixon-Coles(ρ优化) → 让球平移 → 让球胜平负
阶段E 评估：让球 Log Loss / Brier / 命中率 vs 竞彩让球SP基线；特征重要性；保存模型

用法：
  python train.py              # 全流程（含 Optuna 调优）
  python train.py --trials 0   # 跳过 Optuna（固定参数，快速验证）
  python train.py --dry-run    # 仅载入1000场快速验证流程
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAX_GOALS = 12          # 比分网格上限
DECAY = 0.94            # EWMA 指数衰减因子
MAX_HIST = 10           # 近期状态回溯场数
XG_WINDOW = 5           # xG 窗口
BIG5 = {"英超", "西甲", "德甲", "意甲", "法甲"}


# ============================================================
# 阶段A：数据融合与清洗
# ============================================================
# 双源命名归一化：源内队名 → 标准英文名（标准名来自 team_mapping.csv）
FD_NORM = {
    "Ath Bilbao": "Athletic Club", "Ath Madrid": "Atletico Madrid",
    "Celta": "Celta Vigo", "FC Koln": "Cologne", "Ein Frankfurt": "Eintracht Frankfurt",
    "Espanol": "Espanyol", "Hertha": "Hertha Berlin",
    "M'Gladbach": "Monchengladbach", "M'gladbach": "Monchengladbach",
    "Nott'm Forest": "Nottingham Forest", "Paris SG": "PSG",
    "Vallecano": "Rayo Vallecano", "Sociedad": "Real Sociedad",
    "St Etienne": "Saint-Etienne", "St Pauli": "St. Pauli",
}
US_NORM = {
    "AC Milan": "Milan", "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund", "Borussia M.Gladbach": "Monchengladbach",
    "FC Cologne": "Cologne", "FC Heidenheim": "Heidenheim",
    "Hamburger SV": "Hamburg", "Mainz 05": "Mainz",
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Paris Saint Germain": "PSG",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Real Betis": "Betis", "St. Etienne": "Saint-Etienne",
    "St. Pauli": "St. Pauli", "Hertha BSC": "Hertha Berlin",
}
# 阶段A-2 需要的过程列
PROCESS_COLS = ["htg", "htag", "hs", "as", "hst", "ast",
                "ou_over", "ou_under", "xG_h", "xG_a"]


def load_team_mapping(path="team_mapping.csv"):
    """A-1: 读取已人工核对的队名映射表。"""
    m = pd.read_csv(path, encoding="utf-8-sig")
    return dict(zip(m["中文名"], m["英文名"]))


def build_master_table(dry_run=False, out_path="data/master_table.csv"):
    """A-2: 以 jczq_history 为基，左连接欧洲历史与 xG 数据。"""
    cn2en = load_team_mapping()

    # 基础表
    raw = pd.read_csv("data/jczq_history.csv", nrows=1000 if dry_run else None)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date", "fthg", "ftag"]).reset_index(drop=True)
    raw["home_en"] = raw["home_team"].map(cn2en).fillna(raw["home_team"])
    raw["away_en"] = raw["away_team"].map(cn2en).fillna(raw["away_team"])
    raw = raw.drop(columns=[c for c in PROCESS_COLS if c in raw.columns])
    print(f"[A] 基础表: {len(raw)} 场")

    # football_history 侧表
    import glob
    fd_rows = []
    for f in glob.glob("data/football_history/*.csv"):
        try:
            x = pd.read_csv(f)
            if "Date" not in x.columns or "HomeTeam" not in x.columns:
                continue
            x["date"] = pd.to_datetime(x["Date"], dayfirst=True,
                                       errors="coerce").dt.normalize()
            keep = [c for c in ("date", "HomeTeam", "AwayTeam", "FTHG", "FTAG",
                                "HTHG", "HTAG", "HS", "AS", "HST", "AST",
                                "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5")
                    if c in x.columns]
            fd_rows.append(x[keep].copy())
        except Exception:
            pass
    fd = pd.concat(fd_rows, ignore_index=True).dropna(
        subset=["date", "HomeTeam", "AwayTeam"])
    fd["HomeTeam"] = fd["HomeTeam"].map(lambda s: FD_NORM.get(s, s))
    fd["AwayTeam"] = fd["AwayTeam"].map(lambda s: FD_NORM.get(s, s))
    fd = fd.drop_duplicates(subset=["date", "HomeTeam", "AwayTeam"], keep="first")
    fd = fd.rename(columns={"FTHG": "fd_fthg", "FTAG": "fd_ftag", "HTHG": "htg",
                            "HTAG": "htag", "HS": "hs", "AS": "as", "HST": "hst",
                            "AST": "ast", "B365H": "fd_b365h", "B365D": "fd_b365d",
                            "B365A": "fd_b365a", "B365>2.5": "ou_over",
                            "B365<2.5": "ou_under"})
    print(f"[A] football_history 侧表: {len(fd)} 场")

    # understat xG 侧表
    us_rows = []
    for f in glob.glob("data/understat/*.json"):
        try:
            j = json.load(open(f, encoding="utf-8"))
            for dt in j.get("dates", []):
                if not dt.get("isResult", True):
                    continue
                try:
                    g, xg = dt["goals"], dt["xG"]
                    us_rows.append({
                        "date": pd.to_datetime(dt["datetime"]).normalize(),
                        "HomeTeam": US_NORM.get(dt["h"]["title"], dt["h"]["title"]),
                        "AwayTeam": US_NORM.get(dt["a"]["title"], dt["a"]["title"]),
                        "xG_h": float(xg["h"]) if xg.get("h") else np.nan,
                        "xG_a": float(xg["a"]) if xg.get("a") else np.nan,
                    })
                except Exception:
                    pass
        except Exception:
            pass
    us = pd.DataFrame(us_rows).drop_duplicates(
        subset=["date", "HomeTeam", "AwayTeam"], keep="first")
    print(f"[A] understat 侧表: {len(us)} 场")

    # 左连接
    merged = raw.merge(fd, how="left",
                       left_on=["date", "home_en", "away_en"],
                       right_on=["date", "HomeTeam", "AwayTeam"])
    merged = merged.drop(columns=["HomeTeam", "AwayTeam"]).merge(
        us, how="left", left_on=["date", "home_en", "away_en"],
        right_on=["date", "HomeTeam", "AwayTeam"])
    merged = merged.drop(columns=["HomeTeam", "AwayTeam"])

    # 联赛均值填充 + 缺失指示
    for c in PROCESS_COLS:
        if c not in merged.columns:
            merged[c] = np.nan
        merged[c + "_miss"] = merged[c].isna().astype(int)
    lm = merged.groupby("league")[PROCESS_COLS].transform("mean")
    merged[PROCESS_COLS] = merged[PROCESS_COLS].fillna(lm)
    for c in PROCESS_COLS:
        merged[c] = merged[c].fillna(merged[c].mean())

    # 去重
    before = len(merged)
    merged = merged.drop_duplicates(subset=["date", "home_team", "away_team", "league"])
    print(f"[A] 去重: {before} -> {len(merged)}")
    if not dry_run:
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return merged


# ============================================================
# 阶段B：特征工程
# ============================================================
class TeamState:
    """单队滚动状态（指数加权），全部基于赛前已知数据。"""
    __slots__ = ["history", "last_date", "h2h"]

    def __init__(self):
        self.history = []                    # [(date, gf, ga, pts, sot, xg, xga)]
        self.last_date = None
        self.h2h = defaultdict(list)         # 对手 -> [总进球, ...]

    def add(self, date, gf, ga, sot, xg, xga, opp):
        self.history.append((date, gf, ga, _pts(gf, ga), sot, xg, xga))
        self.h2h[opp].append(gf + ga)
        self.last_date = date

    def rest_days(self, date):
        if self.last_date is None:
            return 30.0
        return (date - self.last_date).days

    def recent(self, date, k=MAX_HIST):
        return [h for h in self.history if h[0] < date][-k:]

    def ewma(self, date, field_idx, k=MAX_HIST, decay=DECAY):
        rows = self.recent(date, k)
        if not rows:
            return np.nan
        wsum = wtot = 0.0
        for r in rows:
            w = decay ** max((date - r[0]).days, 0)
            v = r[field_idx]
            if np.isnan(v):
                continue
            wsum += w * v
            wtot += w
        return wsum / wtot if wtot > 0 else np.nan

    def std_goals_3(self, date):
        rows = self.recent(date, 3)
        if len(rows) < 2:
            return 0.0
        return float(np.std([r[1] for r in rows]))

    def h2h_total_goals(self, opp):
        hist = self.h2h.get(opp, [])
        return float(np.mean(hist)) if hist else np.nan


def _pts(gf, ga):
    return 3.0 if gf > ga else (1.0 if gf == ga else 0.0)


def implied_probs(o_h, o_d, o_a):
    if not (o_h and o_d and o_a and o_h > 1 and o_d > 1 and o_a > 1):
        return np.nan, np.nan, np.nan
    inv = 1.0 / np.array([o_h, o_d, o_a], dtype=float)
    return tuple(inv / inv.sum())


def goal_line_from_ou(ou_over, ou_under):
    if not (ou_over and ou_under and ou_over > 1 and ou_under > 1):
        return np.nan
    p_over = (1 / ou_over) / (1 / ou_over + 1 / ou_under)
    from scipy.stats import norm
    return 2.5 + norm.ppf(p_over) * 1.35


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """B: 逐场构建赛前已知特征（与 predict 阶段一致）。"""
    df = df.sort_values("date").reset_index(drop=True)
    states = defaultdict(TeamState)
    rows = []

    def double_week(st, d):
        return 1.0 if len([x for x in st.history if (d - x[0]).days <= 7]) >= 2 else 0.0

    def xg_eff(st, d):
        diffs = [r[5] - r[1] for r in st.recent(d, XG_WINDOW) if not np.isnan(r[5])]
        return float(np.mean(diffs)) if diffs else np.nan

    for _, m in df.iterrows():
        date, h, a = m["date"], m["home_team"], m["away_team"]
        sh, sa = states[h], states[a]
        ih, id_, ia = implied_probs(m["b365h"], m["b365d"], m["b365a"])
        rq = m["ahh"] if pd.notna(m["ahh"]) else 0.0
        gl = goal_line_from_ou(m["ou_over"], m["ou_under"]) if pd.notna(m["ou_over"]) else np.nan
        feat = {
            "implied_H": ih, "implied_D": id_, "implied_A": ia,
            "rq_line": rq, "total_goal_line": gl,
            "h_w_goals": sh.ewma(date, 1), "h_w_goals_conceded": sh.ewma(date, 2),
            "h_w_pts": sh.ewma(date, 3), "h_w_sot": sh.ewma(date, 4),
            "h_std_goals_3": sh.std_goals_3(date),
            "h_w_xg": sh.ewma(date, 5, k=XG_WINDOW), "h_w_xga": sh.ewma(date, 6, k=XG_WINDOW),
            "h_rest_days": sh.rest_days(date),
            "a_w_goals": sa.ewma(date, 1), "a_w_goals_conceded": sa.ewma(date, 2),
            "a_w_pts": sa.ewma(date, 3), "a_w_sot": sa.ewma(date, 4),
            "a_std_goals_3": sa.std_goals_3(date),
            "a_w_xg": sa.ewma(date, 5, k=XG_WINDOW), "a_w_xga": sa.ewma(date, 6, k=XG_WINDOW),
            "a_rest_days": sa.rest_days(date),
            "h_is_double_week": double_week(sh, date),
            "a_is_double_week": double_week(sa, date),
            "h2h_total_goals": sh.h2h_total_goals(a),
            "h2h_rev_total_goals": sa.h2h_total_goals(h),
            "h_xg_eff": xg_eff(sh, date),
            "a_xg_eff": xg_eff(sa, date),
        }
        vals = [v for v in (feat["h2h_total_goals"], feat["h2h_rev_total_goals"])
                if pd.notna(v)]
        feat["h2h_total_goals_avg"] = float(np.mean(vals)) if vals else np.nan
        feat.update({"date": date, "league": m["league"], "home_team": h,
                     "away_team": a, "fthg": m["fthg"], "ftag": m["ftag"],
                     "home_en": m["home_en"], "away_en": m["away_en"],
                     "b365h": m["b365h"], "b365d": m["b365d"], "b365a": m["b365a"]})
        rows.append(feat)
        sh.add(date, m["fthg"], m["ftag"],
               m["hst"] if pd.notna(m["hst"]) else np.nan,
               m["xG_h"] if pd.notna(m["xG_h"]) else np.nan,
               m["xG_a"] if pd.notna(m["xG_a"]) else np.nan, a)
        sa.add(date, m["ftag"], m["fthg"],
               m["ast"] if pd.notna(m["ast"]) else np.nan,
               m["xG_a"] if pd.notna(m["xG_a"]) else np.nan,
               m["xG_h"] if pd.notna(m["xG_h"]) else np.nan, h)
    return pd.DataFrame(rows)


def finalize_features(feat: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """B: 缺失填 -1 + 缺失指示列。"""
    out = feat.copy()
    for c in feature_cols:
        out[c + "_miss"] = out[c].isna().astype(int)
        out[c] = out[c].fillna(-1.0)
    return out


def get_feature_cols(feat: pd.DataFrame) -> list:
    return [c for c in feat.columns if c not in
            {"date", "league", "home_team", "away_team", "fthg", "ftag",
             "home_en", "away_en", "b365h", "b365d", "b365a"}]


# ============================================================
# 阶段D：比分概率生成（Dixon-Coles + 让球平移）
# ============================================================
def build_grid(lam_h, lam_a, rho=0.0):
    """独立泊松矩阵 + Dixon-Coles 低比分修正，归一化。"""
    maxg = MAX_GOALS
    ph = poisson.pmf(np.arange(maxg + 1), lam_h)
    pa = poisson.pmf(np.arange(maxg + 1), lam_a)
    grid = np.outer(ph, pa)
    if rho != 0:
        for i in range(2):
            for j in range(2):
                lh, la = lam_h, lam_a
                if i == 0 and j == 0:
                    tau = 1 - lh * la * rho
                elif i == 0 and j == 1:
                    tau = 1 + lh * rho
                elif i == 1 and j == 0:
                    tau = 1 + la * rho
                elif i == 1 and j == 1:
                    tau = 1 - rho
                else:
                    tau = 1
                grid[i, j] *= tau
    s = grid.sum()
    return grid / s if s > 0 else grid


def range_probs_from_grid(grid, rang):
    """让球胜平负：主队进球 + rang 再与客队比较。"""
    n = grid.shape[0]
    ih, ja = np.indices((n, n))
    shifted = ih + rang
    return (grid[shifted > ja].sum(), grid[shifted == ja].sum(),
            grid[shifted < ja].sum())


def grid_to_probs(grid):
    n = grid.shape[0]
    ih, ja = np.indices((n, n))
    hw, dr, aw = grid[ih > ja].sum(), grid[ih == ja].sum(), grid[ih < ja].sum()
    tg = np.zeros(2 * n)
    for i in range(n):
        for j in range(n):
            tg[i + j] += grid[i, j]
    return hw, dr, aw, tg


# ============================================================
# 阶段C：模型训练
# ============================================================
def train_lambda(X_tr, y_tr, X_va, y_va, params, n_rounds=500):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=n_rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50, verbose=False)])


def calibrate(model, X_tr, y_tr):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.clip(model.predict(X_tr), 1e-6, None), y_tr)
    return iso


def predict_lam(model_h, model_a, cal_h, cal_a, X):
    lh = np.maximum(cal_h.predict(np.clip(model_h.predict(X), 1e-6, None)), 1e-6)
    la = np.maximum(cal_a.predict(np.clip(model_a.predict(X), 1e-6, None)), 1e-6)
    return lh, la


# ============================================================
# 阶段E：评估
# ============================================================
def range_odds_baseline(df_sub):
    """竞彩让球SP 去抽水（b365h/d/a 即让球SP）。"""
    o = df_sub[["b365h", "b365d", "b365a"]].values.astype(float)
    valid = np.isfinite(o).all(axis=1) & (o > 1).all(axis=1)
    p = np.full_like(o, np.nan)
    p[valid] = 1.0 / o[valid]
    p[valid] /= p[valid].sum(axis=1, keepdims=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30, help="Optuna 试验次数，0=跳过")
    ap.add_argument("--dry-run", action="store_true", help="仅载入1000场快速验证")
    args = ap.parse_args()

    # ---- 阶段A ----
    master = build_master_table(dry_run=args.dry_run)

    # ---- 阶段B：优先复用 features_v2 生成的 34 特征版（含半场特征） ----
    import pathlib
    BASE_DIR = pathlib.Path(__file__).resolve().parent
    feat_path = BASE_DIR / "data" / "features.csv"
    if feat_path.exists() and not args.dry_run:
        fin = pd.read_csv(feat_path, parse_dates=["date"])
        with open(BASE_DIR / "data" / "feature_state.pkl", "rb") as f:
            cols = pickle.load(f)["feature_cols"]
        print(f"[B] 复用 features.csv: {fin.shape}, 特征列 {len(cols)}")
    else:
        feat = build_features(master)
        cols = get_feature_cols(feat)
        print(f"[B] 特征: {feat.shape}, 特征列 {len(cols)}")
        fin = finalize_features(feat, cols)
        if args.dry_run:
            fin.to_csv("data/features_dryrun.csv", index=False, encoding="utf-8-sig")
        else:
            fin.to_csv("data/features.csv", index=False, encoding="utf-8-sig")
            with open("data/feature_state.pkl", "wb") as f:
                pickle.dump({"feature_cols": cols}, f)

    X = fin[cols + [c + "_miss" for c in cols]].values
    yh = fin["fthg"].values.astype(float)
    ya = fin["ftag"].values.astype(float)
    rang = fin["rq_line"].values.astype(float)

    # ---- 时序划分（严禁随机） ----
    test_start = fin["date"].max() - pd.Timedelta(days=365)   # 最近1年测试
    if args.dry_run:
        # dry-run 仅 1000 场早期数据：简单 8:1:1 时序划分
        d_max = fin["date"].max()
        val_start = d_max - pd.Timedelta(days=120)
        test_start = d_max - pd.Timedelta(days=30)
    else:
        val_start = pd.Timestamp("2025-06-30")                # 验证起点
    tr = (fin["date"] < val_start).values
    va = ((fin["date"] >= val_start) & (fin["date"] <= test_start)).values
    te = (fin["date"] > test_start).values
    if args.dry_run and (va.sum() == 0 or te.sum() == 0 or tr.sum() == 0):
        # 早期数据稀疏：回退到 8:1:1 比例划分（仅 dry-run 快速验证用）
        n = len(fin)
        tr = np.zeros(n, dtype=bool); tr[: int(n * 0.8)] = True
        va = np.zeros(n, dtype=bool); va[int(n * 0.8): int(n * 0.9)] = True
        te = np.zeros(n, dtype=bool); te[int(n * 0.9):] = True
    print(f"[C] 划分: 训练 {tr.sum()} / 验证 {va.sum()} / 测试 {te.sum()}")

    # ---- Optuna 调优 ----
    params = {"objective": "poisson", "num_leaves": 50, "learning_rate": 0.036,
              "min_child_samples": 88, "feature_fraction": 0.735, "lambda_l2": 1.96,
              "verbosity": -1}
    if args.trials > 0 and not args.dry_run:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            p = {"objective": "poisson",
                 "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                 "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                 "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                 "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
                 "verbosity": -1}
            mh = train_lambda(X[tr], yh[tr], X[va], yh[va], p)
            ma = train_lambda(X[tr], ya[tr], X[va], ya[va], p)
            ph = np.clip(mh.predict(X[va]), 1e-6, None)
            pa = np.clip(ma.predict(X[va]), 1e-6, None)
            return float(np.mean(ph - yh[va] * np.log(ph) + pa - ya[va] * np.log(pa)))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.trials)
        params = {"objective": "poisson", "verbosity": -1, **study.best_params}
        print(f"[C] Optuna 最优 NLL={study.best_value:.4f}")

    # ---- 训练 + 校准 ----
    mh = train_lambda(X[tr], yh[tr], X[va], yh[va], params)
    ma = train_lambda(X[tr], ya[tr], X[va], ya[va], params)
    ch = calibrate(mh, X[tr], yh[tr])
    ca = calibrate(ma, X[tr], ya[tr])

    # ---- ρ 优化（验证集） ----
    lh_v, la_v = predict_lam(mh, ma, ch, ca, X[va])
    yh_v, ya_v, rg_v = yh[va], ya[va], rang[va]
    best_rho, best_ll = 0.0, 1e9
    for rho in [0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]:
        lls = []
        for i in range(va.sum()):
            hw, dr, aw = range_probs_from_grid(build_grid(lh_v[i], la_v[i], rho),
                                               int(round(rg_v[i])))
            p_act = hw if yh_v[i] + rg_v[i] > ya_v[i] else (dr if yh_v[i] + rg_v[i] == ya_v[i] else aw)
            lls.append(-np.log(max(p_act, 1e-12)))
        ll = float(np.mean(lls))
        if ll < best_ll:
            best_ll, best_rho = ll, rho
    print(f"[D] 验证集最优 ρ={best_rho} (让球LL={best_ll:.4f})")

    # ---- 测试集评估 ----
    lh, la = predict_lam(mh, ma, ch, ca, X[te])
    yh_t, ya_t, rg_t = yh[te], ya[te], rang[te]
    n = te.sum()
    pm = np.zeros((n, 3))
    for i in range(n):
        hw, dr, aw = range_probs_from_grid(build_grid(lh[i], la[i], best_rho),
                                           int(round(rg_t[i])))
        pm[i] = [hw, dr, aw]
    yoh = np.zeros((n, 3))
    yoh[yh_t + rg_t > ya_t, 0] = 1
    yoh[yh_t + rg_t == ya_t, 1] = 1
    yoh[yh_t + rg_t < ya_t, 2] = 1
    base = range_odds_baseline(fin[te])
    valid_b = np.isfinite(base).all(axis=1)

    ll_m = float(-np.mean(np.sum(yoh * np.log(np.clip(pm, 1e-12, 1)), axis=1)))
    ll_b = float(-np.mean(np.sum(yoh[valid_b] * np.log(np.clip(base[valid_b], 1e-12, 1)), axis=1)))
    br_m = float(np.mean(np.sum((pm - yoh) ** 2, axis=1)))
    br_b = float(np.mean(np.sum((base[valid_b] - yoh[valid_b]) ** 2, axis=1)))
    acc_m = float(np.mean(np.argmax(pm, 1) == np.argmax(yoh, 1)))
    acc_b = float(np.mean(np.argmax(base[valid_b], 1) == np.argmax(yoh[valid_b], 1)))

    print("\n" + "=" * 62)
    print(f"让球胜平负评估（测试集 {n} 场）")
    print("=" * 62)
    print(f"{'指标':<12}{'模型':<12}{'让球SP基线':<12}{'提升':<10}")
    print(f"{'Log Loss':<12}{ll_m:<12.4f}{ll_b:<12.4f}{ll_b-ll_m:<+10.4f}")
    print(f"{'Brier':<12}{br_m:<12.4f}{br_b:<12.4f}{br_b-br_m:<+10.4f}")
    print(f"{'命中率':<12}{acc_m:<12.3f}{acc_b:<12.3f}{acc_m-acc_b:<+10.3f}")

    # 特征重要性
    if not args.dry_run:
        imp = pd.DataFrame({"feature": cols + [c + "_miss" for c in cols],
                            "importance": mh.feature_importance("gain")})
        imp = imp.sort_values("importance", ascending=False).head(20)
        os.makedirs("outputs", exist_ok=True)
        imp.to_csv("outputs/feature_importance.csv", index=False, encoding="utf-8-sig")
        print("\n特征重要性 Top10:")
        print(imp.head(10).to_string(index=False))

        # ---- 保存 ----
        os.makedirs("models_v2", exist_ok=True)
        mh.save_model("models_v2/lgb_h.txt")
        ma.save_model("models_v2/lgb_a.txt")
        with open("models_v2/calibrators.pkl", "wb") as f:
            pickle.dump({"cal_h": ch, "cal_a": ca, "rho": best_rho}, f)
        with open("models_v2/feature_columns.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(cols + [c + "_miss" for c in cols]))
        with open("models_v2/train_meta.json", "w", encoding="utf-8") as f:
            json.dump({"n_train": int(tr.sum()), "n_val": int(va.sum()),
                       "n_test": int(n), "rho": float(best_rho),
                       "ll_model": float(ll_m), "ll_base": float(ll_b),
                       "brier_model": float(br_m), "brier_base": float(br_b),
                       "acc_model": float(acc_m), "acc_base": float(acc_b),
                       "mode": "range_spf", "params": params},
                      f, ensure_ascii=False, indent=2)
        print("\n→ 模型已保存 models_v2/ (lgb_h.txt / lgb_a.txt / calibrators.pkl / "
              "feature_columns.txt / train_meta.json)")


if __name__ == "__main__":
    main()
