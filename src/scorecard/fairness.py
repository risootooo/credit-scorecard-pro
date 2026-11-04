"""Fairness audit by the dataset's ``sex`` attribute.

Caveats baked into every output: in the German Credit data, sex comes from
``personal_status_sex``, which entangles sex with marital status (e.g. code
A92 is "female divorced/separated/married"); the female group is only ~310
of 1,000 rows, and in a 30% test split the bands get small fast. Group sizes
are printed with every rate so nothing is overclaimed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import auc_score


def group_metrics(df: pd.DataFrame, p: np.ndarray, cutoff: float,
                  group_col: str = "sex") -> pd.DataFrame:
    d = df.copy()
    d["p"] = p
    d["approved"] = d["p"] <= cutoff
    rows = []
    for group, g in d.groupby(group_col):
        rows.append({
            "group": group, "n": len(g),
            "approval_rate": float(g["approved"].mean()),
            "observed_bad_rate": float(g["bad"].mean()),
            "approved_bad_rate": float(g.loc[g["approved"], "bad"].mean())
            if g["approved"].any() else float("nan"),
            "auc_within_group": auc_score(g["bad"].to_numpy(), g["p"].to_numpy())
            if g["bad"].nunique() == 2 else float("nan"),
        })
    out = pd.DataFrame(rows)
    if len(out) == 2:
        hi, lo = out["approval_rate"].max(), out["approval_rate"].min()
        out.attrs["approval_gap_pp"] = float((hi - lo) * 100)
        out.attrs["adverse_impact_ratio"] = float(lo / hi) if hi else float("nan")
    return out


def score_band_decomposition(df: pd.DataFrame, scores: np.ndarray, p: np.ndarray,
                             cutoff_score: float, group_col: str = "sex",
                             bands: tuple = (0, 520, 580, 640, 700, 10_000)) -> pd.DataFrame:
    """Approval rate by group WITHIN score bands — separates 'groups differ
    in score distribution' from 'same score, different treatment'. Because
    approval here is a deterministic function of score, within-band approval
    is identical by construction above/below the cut-off band; the informative
    output is each group's population share per band (exposure decomposition)
    plus within-band bad rates (calibration-by-group evidence)."""
    d = df.copy()
    d["score"] = scores
    d["p"] = p
    labels = [f"{bands[i]}-{bands[i + 1]}" for i in range(len(bands) - 1)]
    d["band"] = pd.cut(d["score"], bins=list(bands), labels=labels)
    rows = []
    for (band, group), g in d.groupby(["band", group_col], observed=True):
        rows.append({
            "score_band": str(band), "group": group, "n": len(g),
            "share_of_group_pct": float(100 * len(g) / (d[group_col] == group).sum()),
            "observed_bad_rate": float(g["bad"].mean()),
            "mean_predicted_pd": float(g["p"].mean()),
        })
    return pd.DataFrame(rows)


def drift_scenario(df: pd.DataFrame, model, scaler, feature: str = "credit_amount",
                   multiplier: float = 1.20) -> dict:
    """Simulated drift: scale one feature's values, re-score, measure migration."""
    from .validation import population_stability_index

    base_scores = scaler.score_from_pbad(model.predict_proba(df))
    drifted = df.copy()
    drifted[feature] = drifted[feature] * multiplier
    drift_scores = scaler.score_from_pbad(model.predict_proba(drifted))
    return {
        "feature": feature, "multiplier": multiplier,
        "mean_score_before": float(np.mean(base_scores)),
        "mean_score_after": float(np.mean(drift_scores)),
        "score_psi": population_stability_index(base_scores, drift_scores),
    }
