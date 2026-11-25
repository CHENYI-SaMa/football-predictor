# -*- coding: utf-8 -*-
"""eval_enhance.py —— 补齐模型指令缺失项（LR基线/AUC/校准曲线/多模型对比）
=======================================================================
用同一时序分割（同 train_v3）评估：
  1. LR 基线：输入 = 赔率隐含概率（prob_h/d/a）+ 少量特征
  2. LightGBM（现有 models_v3 双泊松 → 赛果概率）
  3. XGBoost / RandomForest（三分类直连，对比）
  4. Stacking（已有 models_v3_spf）
指标：Log Loss / Brier / Acc / AUC(二分类"主胜vs非主胜" 与宏平均) / 校准曲线
输出：outputs/eval_enhance_report.txt + 校准数据
用法: python eval_enhance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.features_v3 import FeatureBuilderV3          # noqa: E402
from src.model import PoissonLightGBM                 # noqa: E402
from src.evaluation import odds_baseline_probs        # noqa: E402
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

HIST_CSV = BASE_DIR / "data" / "jczq_history_xg.csv"


def onehot(y_h, y_a):
    y = np.zeros((len(y_h), 3))
    y[y_h > y_a, 0] = 1
    y[y_h == y_a, 1] = 1
    y[y_h < y_a, 2] = 1
    return y


def build_data():
    raw = pd.read_csv(HIST_CSV, parse_dates=["date"])
    raw = raw.dropna(subset=["fthg", "ftag"]).reset_index(drop=True)
    fb = FeatureBuilderV3(raw)
    feat, _ = fb.build()
    feat["rang"] = raw.sort_values("date").reset_index(drop=True)["ahh"].values.astype(float)
    max_date = feat["date"].max()
    test_start = max_date - pd.Timedelta(days=30)
    val_start = max_date - pd.Timedelta(days=90)
    tr = feat[feat["date"] <= val_start].copy()
    va = feat[(feat["date"] > val_start) & (feat["date"] <= test_start)].copy()
    te = feat[feat["date"] > test_start].copy()
    cols = [c for c in feat.columns if c not in
            {"fthg", "ftag", "date", "league", "home_team", "away_team", "rang"}]
    return tr, va, te, cols


def metrics(name, p, yoh):
    """LogLoss / Brier / Acc / AUC(二分类宏平均 + 主胜vs非主胜)。"""
    from sklearn.metrics import roc_auc_score
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    ll = float(-np.mean(np.sum(yoh * np.log(p), axis=1)))
    brier = float(np.mean(np.sum((p - yoh) ** 2, axis=1)))
    acc = float(np.mean(np.argmax(p, 1) == np.argmax(yoh, 1)))
    # AUC：多分类宏平均（one-vs-rest）+ 主胜 vs 非主胜
    auc_macro = float(roc_auc_score(yoh, p, average="macro", multi_class="ovr"))
    auc_home = float(roc_auc_score(yoh[:, 0], p[:, 0]))
    return {"ll": ll, "brier": brier, "acc": acc,
            "auc_macro": auc_macro, "auc_home": auc_home}


def calibration_bins(p_home, y_home, n_bins=10):
    """主胜概率校准分箱：预测均值 vs 实际频率。"""
    idx = np.argsort(p_home)
    bins = np.array_split(idx, n_bins)
    out = []
    for b in bins:
        if len(b) == 0:
            continue
        out.append((float(np.mean(p_home[b])), float(np.mean(y_home[b])), int(len(b))))
    return out


def main():
    tr, va, te, cols = build_data()
    yh_tr, ya_tr = tr["fthg"].values, tr["ftag"].values
    yh_va, ya_va = va["fthg"].values, va["ftag"].values
    yh_te, ya_te = te["fthg"].values, te["ftag"].values
    y_tr, y_va, y_te = (np.argmax(onehot(yh, ya), 1)
                        for yh, ya in ((yh_tr, ya_tr), (yh_va, ya_va), (yh_te, ya_te)))
    yoh_te = onehot(yh_te, ya_te)

    lines = []
    lines.append(f"评估增强报告（测试集 {len(te)} 场，≥ {te['date'].max().date() - pd.Timedelta(days=30)}）")
    lines.append("=" * 76)

    # ---- 0. 市场 SP 基线（让球SP 去抽水） ----
    m_te = odds_baseline_probs(te["odds_h"].values, te["odds_d"].values, te["odds_a"].values)
    lines.append(f"\n{'模型':<22}{'LogLoss':<9}{'Brier':<8}{'Acc':<8}{'AUC宏':<8}{'AUC主胜'}")
    lines.append("-" * 76)

    def report(name, p):
        m = metrics(name, p, yoh_te)
        lines.append(f"{name:<22}{m['ll']:<9.4f}{m['brier']:<8.4f}"
                     f"{m['acc']:<8.3f}{m['auc_macro']:<8.3f}{m['auc_home']:<8.3f}")
        return m

    report("市场SP基线", m_te)

    # ---- 1. LR 基线（赔率隐含概率 + 少量特征） ----
    lr_feats = ["prob_h", "prob_d", "prob_a", "home_elo", "away_elo", "elo_diff"]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(tr[lr_feats].values, y_tr)
    report("LR(赔率+ELO)", lr.predict_proba(te[lr_feats].values))

    lr2 = LogisticRegression(max_iter=1000)
    lr2.fit(tr[cols].values, y_tr)
    report("LR(全特征)", lr2.predict_proba(te[cols].values))

    # ---- 2. 双泊松（现场重训，避免旧模型文件特征数不匹配） ----
    poisson = PoissonLightGBM()
    poisson.fit(tr[cols].values, tr["fthg"].values, tr["ftag"].values,
                va[cols].values, va["fthg"].values, va["ftag"].values,
                num_boost_round=1500, early_stopping=80)
    grids = poisson.predict_grids(te[cols].values)
    n = grids.shape[1]
    ih, ja = np.indices((n, n))
    p_pois = np.clip(np.stack([grids[:, ih > ja].sum(1),
                               grids[:, ih == ja].sum(1),
                               grids[:, ih < ja].sum(1)], 1), 1e-12, 1.0)
    report("双泊松V3", p_pois)

    # ---- 3. RandomForest（三分类直连） ----
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                min_samples_leaf=30, n_jobs=-1, random_state=42)
    rf.fit(tr[cols].values, y_tr)
    report("RandomForest", rf.predict_proba(te[cols].values))

    # ---- 4. XGBoost（三分类直连） ----
    try:
        import xgboost as xgb
        dx = xgb.DMatrix(tr[cols].values, label=y_tr)
        dve = xgb.DMatrix(va[cols].values, label=y_va)
        bx = xgb.train({"objective": "multi:softprob", "num_class": 3,
                        "eta": 0.05, "max_depth": 5, "subsample": 0.8,
                        "colsample_bytree": 0.8, "lambda": 1.0},
                       dx, num_boost_round=500, evals=[(dve, "val")],
                       early_stopping_rounds=50, verbose_eval=False)
        report("XGBoost", bx.predict(xgb.DMatrix(te[cols].values)))
    except ImportError:
        lines.append("XGBoost 未安装，跳过")

    # ---- 4.5 CatBoost（三分类直连，对比） ----
    try:
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, loss_function="MultiClass",
            random_seed=42, verbose=False)
        cb.fit(tr[cols].values, y_tr, eval_set=(va[cols].values, y_va),
               early_stopping_rounds=50)
        report("CatBoost", cb.predict_proba(te[cols].values))
    except ImportError:
        lines.append("CatBoost 未安装，跳过")

    # ---- 5. Stacking（三分类现场重训 + LR 元学习器） ----
    try:
        from src.evaluation import odds_baseline_probs as obp
        import lightgbm as lgb
        # 三分类直连（重训，避免依赖损坏的模型文件）
        dtr = lgb.Dataset(tr[cols].values, label=y_tr)
        dva = lgb.Dataset(va[cols].values, label=y_va, reference=dtr)
        cls = lgb.train({"objective": "multiclass", "num_class": 3,
                         "verbosity": -1, "num_leaves": 31, "learning_rate": 0.05,
                         "min_child_samples": 30, "feature_fraction": 0.8,
                         "lambda_l2": 1.0},
                        dtr, num_boost_round=500, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        c_tr = np.clip(cls.predict(tr[cols].values), 1e-12, 1.0)
        c_va = np.clip(cls.predict(va[cols].values), 1e-12, 1.0)
        c_te = np.clip(cls.predict(te[cols].values), 1e-12, 1.0)
        m_tr = obp(tr["odds_h"].values, tr["odds_d"].values, tr["odds_a"].values)
        m_va = obp(va["odds_h"].values, va["odds_d"].values, va["odds_a"].values)
        def pois_p(grids_):
            return np.clip(np.stack([grids_[:, ih > ja].sum(1),
                                     grids_[:, ih == ja].sum(1),
                                     grids_[:, ih < ja].sum(1)], 1), 1e-12, 1.0)
        grids_tr = poisson.predict_grids(tr[cols].values)
        grids_va = poisson.predict_grids(va[cols].values)
        p_tr = pois_p(grids_tr); p_va = pois_p(grids_va); p_te = pois_p(grids)
        meta_lr = LogisticRegression(max_iter=1000)
        meta_lr.fit(np.concatenate([p_va, c_va, m_va], 1), y_va)
        s_te = meta_lr.predict_proba(np.concatenate([p_te, c_te, m_te], 1))
        report("Stacking(泊松+分类+市场)", s_te)
    except Exception as e:
        lines.append(f"Stacking 评估失败: {e}")

    # ---- 6. 校准曲线（主胜） ----
    lines.append("\n\n【校准曲线 · 主胜概率】（预测均值 vs 实际频率，测试集）")
    lines.append(f"{'分箱':<6}{'预测均值':<10}{'实际频率':<10}{'样本'}")
    for model_name, p in (("市场SP", m_te), ("双泊松V3", p_pois)):
        lines.append(f"--- {model_name} ---")
        for pm, freq, cnt in calibration_bins(p[:, 0], yoh_te[:, 0]):
            lines.append(f"{'':<6}{pm:<10.3f}{freq:<10.3f}{cnt}")

    # 保存
    report_txt = "\n".join(lines)
    print(report_txt)
    (BASE_DIR / "outputs" / "eval_enhance_report.txt").write_text(report_txt, encoding="utf-8")
    print("\n→ outputs/eval_enhance_report.txt")


if __name__ == "__main__":
    main()
