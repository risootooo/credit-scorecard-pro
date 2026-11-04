"""Discrimination, calibration and stability metrics — all implemented
from first principles (rank-based AUC, KS from cumulative distributions,
PSI from bin shares)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney rank AUC. Higher p must mean higher P(bad)."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    ranks = stats.rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Max separation between cumulative good and bad distributions."""
    order = np.argsort(p)
    y_sorted = np.asarray(y)[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_bad - cum_good)))


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return 2 * auc_score(y, p) - 1


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))


def roc_curve(y: np.ndarray, p: np.ndarray, n_points: int = 101):
    thresholds = np.quantile(p, np.linspace(0, 1, n_points))
    thresholds = np.unique(np.r_[np.inf, thresholds, -np.inf])
    tpr, fpr = [], []
    y = np.asarray(y)
    for t in thresholds:
        pred = p >= t
        tpr.append(((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1))
        fpr.append(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
    return np.array(fpr), np.array(tpr)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile calibration + Hosmer-Lemeshow statistic (scipy chi2, g-2 dof)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, float)})
    df["decile"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    rows = []
    hl = 0.0
    for label, g in df.groupby("decile", observed=True):
        n = len(g)
        exp_bad = g["p"].sum()
        obs_bad = g["y"].sum()
        exp_good = n - exp_bad
        obs_good = n - obs_bad
        if exp_bad > 0 and exp_good > 0:
            hl += (obs_bad - exp_bad) ** 2 / exp_bad + (obs_good - exp_good) ** 2 / exp_good
        rows.append({
            "decile": str(label), "n": n,
            "mean_predicted_pd": float(g["p"].mean()),
            "observed_bad_rate": float(g["y"].mean()),
            "expected_bads": float(exp_bad), "observed_bads": int(obs_bad),
        })
    g_bins = len(rows)
    p_value = float(stats.chi2.sf(hl, max(g_bins - 2, 1))) if g_bins > 2 else float("nan")
    table = pd.DataFrame(rows)
    table.attrs["hl_statistic"] = float(hl)
    table.attrs["hl_p_value"] = p_value
    return table


def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins: np.ndarray | None = None, n_bins: int = 10) -> float:
    """PSI between two score/PD samples. 0 = identical; <0.1 stable;
    0.1-0.25 moderate shift; >0.25 major shift (industry convention)."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    if bins is None:
        bins = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
        bins[0], bins[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=bins)
    a_counts, _ = np.histogram(actual, bins=bins)
    e_pct = np.clip(e_counts / max(len(expected), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def validation_summary(y: np.ndarray, p: np.ndarray) -> dict:
    cal = calibration_table(y, p)
    return {
        "auc": auc_score(y, p),
        "ks": ks_statistic(y, p),
        "gini": gini(y, p),
        "brier": brier_score(y, p),
        "hl_statistic": cal.attrs["hl_statistic"],
        "hl_p_value": cal.attrs["hl_p_value"],
        "calibration": cal,
    }
