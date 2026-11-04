"""Logistic regression implemented from scratch (L-BFGS via scipy.optimize).

No sklearn in this stack by design — the point of the repo is that a
scorecard developer should know exactly what the estimator does.
Model: logit P(bad) = b0 + sum_j w_j * WOE_j(x). WOE is defined as
ln(goods share / bads share), so a safer bin has higher WOE and a correctly
signed fitted coefficient is negative. The pipeline enforces this
(sign-consistency refit — see pipeline.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .binning import WoeBinner


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionScratch:
    def __init__(self, l2: float = 1.0, max_iter: int = 500):
        self.l2 = l2  # ridge strength on slopes (intercept unpenalised)
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        Xb = np.column_stack([np.ones(len(X)), X])
        pen = np.r_[0.0, np.full(X.shape[1], self.l2)]

        def nll_and_grad(beta):
            z = Xb @ beta
            p = _sigmoid(z)
            eps = 1e-12
            nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            nll += 0.5 * np.sum(pen * beta ** 2) / len(y)
            grad = Xb.T @ (p - y) / len(y) + pen * beta / len(y)
            return nll / len(y), grad

        res = minimize(nll_and_grad, np.zeros(Xb.shape[1]), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        if not res.success and res.fun > 1.0:
            raise RuntimeError(f"logistic fit failed: {res.message}")
        self.coef_ = res.x[1:]
        self.intercept_ = float(res.x[0])
        self.nll_ = float(res.fun)
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.predict_logit(X))


class ScorecardModel:
    """WOE transform + logistic regression, bundled as one fitted object."""

    def __init__(self, binners: dict[str, WoeBinner], features: list[str],
                 lr: LogisticRegressionScratch):
        self.binners = binners
        self.features = features
        self.lr = lr

    def woe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.binners[f].transform(df[f]) for f in self.features])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.woe_matrix(df))

    def coefficient_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": self.features,
            "coefficient": self.lr.coef_,
            "iv": [self.binners[f].iv_ for f in self.features],
        }).assign(intercept=self.lr.intercept_)


def fit_scorecard(train: pd.DataFrame, binners: dict[str, WoeBinner],
                  features: list[str], **lr_kw) -> ScorecardModel:
    model = ScorecardModel(binners, features, LogisticRegressionScratch(**lr_kw))
    X = model.woe_matrix(train)
    model.lr.fit(X, train["bad"].to_numpy())
    return model


def cross_validated_auc(df: pd.DataFrame, binners_factory, features: list[str],
                        numeric_features: list[str], k: int = 5, seed: int = 7) -> float:
    """Stratified k-fold AUC, refitting binning inside each fold (no leakage)."""
    from .binning import fit_binners
    from .validation import auc_score

    rng = np.random.default_rng(seed)
    folds = np.zeros(len(df), dtype=int)
    for _, group in df.groupby("bad"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        folds[np.searchsorted(df.index, idx)] = np.arange(len(idx)) % k
    aucs = []
    for fold in range(k):
        tr, te = df[folds != fold], df[folds == fold]
        if len(te) == 0:
            continue
        binners = binners_factory(tr, features, numeric_features)
        m = fit_scorecard(tr, binners, features)
        aucs.append(auc_score(te["bad"].to_numpy(), m.predict_proba(te)))
    return float(np.mean(aucs))
