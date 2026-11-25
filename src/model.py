# -*- coding: utf-8 -*-
"""
模型核心模块（对应任务书 4.2 方案 A）
=====================================
LightGBM 双输出泊松回归模型：

1. 自定义 objective：Poisson NLL
       λ = softplus(raw_score)            # 保证 λ > 0
       Loss = mean( λ_h - y_h·log(λ_h) + λ_a - y_a·log(λ_a) )
   梯度 / Fisher hessian 手推解析式，保证数值稳定：
       g = σ·(1 - y/λ)，  h = σ²/λ        （σ = dλ/dx = sigmoid(x)）
2. 主队 / 客队各训练一棵 LightGBM，共享同一特征矩阵。
3. 训练后在『训练集』上对预测 λ 做 isotonic regression 校准，
   使预测概率与真实频率一致（任务书 4.2 节要求）。
4. ρ 参数（Dixon-Coles）在验证集上网格搜索优化。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .poisson_utils import (MAX_GOALS, derive_probs, dixon_coles_grid,
                            match_probs, optimize_rho)

# ---------------------------------------------------------------------------
# 数值稳定辅助
# ---------------------------------------------------------------------------
def _softplus(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softplus：log(1 + exp(x))。"""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def poisson_nll_objective(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    LightGBM 自定义目标函数（泊松负对数似然，softplus 参数化）。
    LightGBM 4.x 签名：fobj(preds, dataset)，返回 (grad, hess)。
    hess 使用 Fisher 信息保证非负。
    """
    y_true = dataset.get_label()
    lam = _softplus(preds)
    lam = np.clip(lam, 1e-6, None)
    sig = _sigmoid(preds)
    grad = sig * (1.0 - y_true / lam)
    hess = np.maximum(sig ** 2 / lam, 1e-6)
    return grad, hess


def _feval_poisson_nll(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
    """LightGBM feval 评估函数（泊松 NLL，越小越好）。"""
    y_true = dataset.get_label()
    lam = np.clip(_softplus(preds), 1e-12, None)
    nll = float(np.mean(lam - y_true * np.log(lam)))
    return "poisson_nll", nll, False


# ---------------------------------------------------------------------------
# 双输出泊松模型
# ---------------------------------------------------------------------------
class PoissonLightGBM:
    """封装主队/客队两棵 LightGBM + isotonic 校准 + Dixon-Coles ρ。"""

    def __init__(self, base_params: dict | None = None):
        self.base_params = {
            "objective": poisson_nll_objective,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "min_child_samples": 30,
            "verbosity": -1,
            "seed": 42,
        }
        if base_params:
            self.base_params.update(base_params)
        self.model_h: lgb.Booster | None = None
        self.model_a: lgb.Booster | None = None
        self.calib_h: IsotonicRegression | None = None
        self.calib_a: IsotonicRegression | None = None
        self.rho: float = 0.0
        self.temperature: float = 1.0  # 温度标定参数（验证集优化）
        self.feature_names: list[str] = []
        # 训练历史（每轮验证集 Poisson NLL），供画曲线
        self.evals_result: dict = {"home": None, "away": None}

    # ------------------------------------------------------------------ 训练
    def fit(self, X: np.ndarray, y_h: np.ndarray, y_a: np.ndarray,
            X_val=None, y_h_val=None, y_a_val=None,
            num_boost_round: int = 2000, early_stopping: int = 100,
            rho_search_on_val: bool = True) -> dict:
        """
        训练两棵 LightGBM；可选早停；训练后对训练集 λ 做 isotonic 校准。
        ρ 在验证集（若有）或训练集上优化。
        返回训练历史（供绘制 Log Loss 曲线）。
        """
        self.feature_names = [str(c) for c in getattr(X, "columns", range(X.shape[1]))]
        X = np.asarray(X, dtype=float)
        y_h = np.asarray(y_h, dtype=float)
        y_a = np.asarray(y_a, dtype=float)

        train_set = lgb.Dataset(X, label=y_h, feature_name=self.feature_names)
        val_set = None
        if X_val is not None:
            val_set = lgb.Dataset(np.asarray(X_val, dtype=float),
                                  label=np.asarray(y_h_val, dtype=float),
                                  reference=train_set)

        history = {}
        # ---- 主队模型 ----
        evals_h: dict = {}
        self.model_h = lgb.train(
            self.base_params, train_set, num_boost_round=num_boost_round,
            valid_sets=[val_set] if val_set is not None else None,
            valid_names=["val"] if val_set is not None else None,
            feval=_feval_poisson_nll,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                       lgb.record_evaluation(evals_h),
                       lgb.log_evaluation(period=50)],
        )
        # ---- 客队模型 ----
        train_set_a = lgb.Dataset(X, label=y_a, feature_name=self.feature_names)
        val_set_a = None
        if X_val is not None:
            val_set_a = lgb.Dataset(np.asarray(X_val, dtype=float),
                                    label=np.asarray(y_a_val, dtype=float),
                                    reference=train_set_a)
        evals_a: dict = {}
        self.model_a = lgb.train(
            self.base_params, train_set_a, num_boost_round=num_boost_round,
            valid_sets=[val_set_a] if val_set_a is not None else None,
            valid_names=["val"] if val_set_a is not None else None,
            feval=_feval_poisson_nll,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                       lgb.record_evaluation(evals_a),
                       lgb.log_evaluation(period=50)],
        )
        self.evals_result = {"home": evals_h, "away": evals_a}

        # ---- isotonic 校准（在训练集上拟合） ----
        lam_h = self.predict_lambda_h(X)
        lam_a = self.predict_lambda_a(X)
        self.calib_h = IsotonicRegression(out_of_bounds="clip")
        self.calib_h.fit(lam_h, y_h)
        self.calib_a = IsotonicRegression(out_of_bounds="clip")
        self.calib_a.fit(lam_a, y_a)

        # ---- ρ 优化 ----
        if val_set is not None and rho_search_on_val:
            lam_h_v = self.predict_lambda_h(np.asarray(X_val, dtype=float), calibrated=True)
            lam_a_v = self.predict_lambda_a(np.asarray(X_val, dtype=float), calibrated=True)
            self.rho = optimize_rho(lam_h_v, lam_a_v,
                                    np.asarray(y_h_val, dtype=float),
                                    np.asarray(y_a_val, dtype=float))
        else:
            lam_h_c = self.predict_lambda_h(X, calibrated=True)
            lam_a_c = self.predict_lambda_a(X, calibrated=True)
            self.rho = optimize_rho(lam_h_c, lam_a_c, y_h, y_a)

        # ---- 温度标定（验证集上优化 T，最小化赛果 Log Loss） ----
        if val_set is not None:
            self._fit_temperature(np.asarray(X_val, dtype=float),
                                  np.asarray(y_h_val, dtype=float),
                                  np.asarray(y_a_val, dtype=float))

        return history

    # ------------------------------------------------------------------ 温度标定
    def _fit_temperature(self, X_val: np.ndarray, y_h_val: np.ndarray,
                         y_a_val: np.ndarray, n_t: int = 61) -> None:
        """
        在验证集上网格搜索最优温度 T：
            P_T(i,j) ∝ P(i,j)^(1/T)，T>1 使分布更平滑（降低过度自信），
        T<1 使分布更尖锐。目标最小化赛果三分类 Log Loss。
        """
        lam_h = self.predict_lambda_h(X_val, calibrated=True)
        lam_a = self.predict_lambda_a(X_val, calibrated=True)
        grids = np.stack([dixon_coles_grid(lh, la, self.rho)
                          for lh, la in zip(lam_h, lam_a)])  # (N, 11, 11)
        # 真实赛果 one-hot
        y_oh = np.zeros((len(y_h_val), 3))
        y_oh[y_h_val > y_a_val, 0] = 1.0
        y_oh[y_h_val == y_a_val, 1] = 1.0
        y_oh[y_h_val < y_a_val, 2] = 1.0
        ih, ja = np.indices((grids.shape[1], grids.shape[2]))

        best_T, best_ll = 1.0, np.inf
        for T in np.linspace(0.5, 2.0, n_t):
            gT = np.power(np.clip(grids, 1e-12, None), 1.0 / T)
            gT /= gT.sum(axis=(1, 2), keepdims=True)
            ph = gT[:, ih > ja].sum(axis=1)
            pd_ = gT[:, ih == ja].sum(axis=1)
            pa = gT[:, ih < ja].sum(axis=1)
            p = np.clip(np.stack([ph, pd_, pa], axis=1), 1e-12, 1.0)
            ll = float(-np.mean(np.sum(y_oh * np.log(p), axis=1)))
            if ll < best_ll:
                best_ll, best_T = ll, T
        self.temperature = best_T

    # ------------------------------------------------------------------ 预测
    def _raw(self, booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
        return np.asarray(booster.predict(np.asarray(X, dtype=float),
                                          num_iteration=booster.best_iteration
                                          if booster.best_iteration is not None
                                          else booster.current_iteration()))

    def predict_lambda_h(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        lam = _softplus(self._raw(self.model_h, X))
        if calibrated and self.calib_h is not None:
            lam = np.clip(self.calib_h.predict(lam), 0.02, None)
        return lam

    def predict_lambda_a(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        lam = _softplus(self._raw(self.model_a, X))
        if calibrated and self.calib_a is not None:
            lam = np.clip(self.calib_a.predict(lam), 0.02, None)
        return lam

    def predict_lambdas(self, X: np.ndarray, calibrated: bool = True) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_lambda_h(X, calibrated), self.predict_lambda_a(X, calibrated)

    def predict_grids(self, X: np.ndarray, calibrated: bool = True,
                      use_dixon_coles: bool = True,
                      max_goals: int = MAX_GOALS) -> np.ndarray:
        """
        返回每场的联合概率网格 (N, 11, 11)。
        依次应用：λ 预测 → isotonic 校准 → Dixon-Coles 修正 → 温度标定。
        """
        lam_h, lam_a = self.predict_lambdas(X, calibrated)
        grids = np.stack([
            dixon_coles_grid(lh, la, self.rho if use_dixon_coles else 0.0, max_goals)
            for lh, la in zip(lam_h, lam_a)])
        if self.temperature != 1.0:
            grids = np.power(np.clip(grids, 1e-12, None), 1.0 / self.temperature)
            grids /= grids.sum(axis=(1, 2), keepdims=True)
        return grids

    def predict_proba(self, X: np.ndarray, calibrated: bool = True,
                      use_dixon_coles: bool = True, max_goals: int = MAX_GOALS) -> list[dict]:
        """对每一行返回完整预测字典（与 poisson_utils.derive_probs 一致）。"""
        grids = self.predict_grids(X, calibrated, use_dixon_coles, max_goals)
        return [derive_probs(g) for g in grids]

    # ------------------------------------------------------------------ 存取
    def save(self, path: str | Path) -> None:
        """序列化为一个 pickle 文件（含模型、校准器、ρ、特征名）。"""
        path = Path(path)
        payload = {
            "model_h": self.model_h.model_to_string(),
            "model_a": self.model_a.model_to_string(),
            "calib_h": self.calib_h,
            "calib_a": self.calib_a,
            "rho": self.rho,
            "temperature": self.temperature,
            "feature_names": self.feature_names,
            "evals_result": self.evals_result,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "PoissonLightGBM":
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        m = cls()
        m.model_h = lgb.Booster(model_str=payload["model_h"])
        m.model_a = lgb.Booster(model_str=payload["model_a"])
        m.calib_h = payload["calib_h"]
        m.calib_a = payload["calib_a"]
        m.rho = payload.get("rho", 0.0)
        m.temperature = payload.get("temperature", 1.0)
        m.feature_names = payload.get("feature_names", [])
        m.evals_result = payload.get("evals_result", {"home": None, "away": None})
        return m


def calibrate_lambdas(lam: np.ndarray, y: np.ndarray, lam_new: np.ndarray) -> np.ndarray:
    """便捷函数：给定 (λ_train, y_train) 拟合 isotonic，再映射新 λ。"""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(lam, y)
    return np.clip(ir.predict(lam_new), 0.02, None)
