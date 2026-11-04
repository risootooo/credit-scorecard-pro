"""Scorecard scaling: log-odds to points.

    score = offset + factor * ln(odds of good:bad)
    factor = PDO / ln(2)                      (points to double the odds)
    offset = base_score - factor * ln(base_odds)

Defaults: 600 points at 50:1 good:bad odds, PDO = 20 — a common retail
convention. Points are allocated per bin so the deployed scorecard is a
simple lookup table:

    points(feature, bin) = -(coef * WOE_bin + intercept / n_features) * factor
                           + offset / n_features

Summing a customer's bin points reproduces the scaled score exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import ScorecardModel


class ScoreScaler:
    def __init__(self, base_score: float = 600.0, base_odds: float = 50.0, pdo: float = 20.0):
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo

    def score_from_pbad(self, p_bad: np.ndarray | float):
        p = np.clip(np.asarray(p_bad, float), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1 - p) / p)

    def pbad_from_score(self, score: np.ndarray | float):
        odds = np.exp((np.asarray(score, float) - self.offset) / self.factor)
        return 1.0 / (1.0 + odds)


def points_table(model: ScorecardModel, scaler: ScoreScaler) -> pd.DataFrame:
    """The deployable scorecard: points for every bin of every feature."""
    n = len(model.features)
    rows = []
    for j, f in enumerate(model.features):
        coef = model.lr.coef_[j]
        for _, bin_row in model.binners[f].table_.iterrows():
            pts = -(coef * bin_row["woe"] + model.lr.intercept_ / n) * scaler.factor \
                  + scaler.offset / n
            rows.append({
                "feature": f, "bin": bin_row["bin"], "woe": bin_row["woe"],
                "bad_rate": bin_row["bad_rate"], "points": round(float(pts), 1),
            })
    return pd.DataFrame(rows)


def score_applicant(model: ScorecardModel, scaler: ScoreScaler,
                    applicant: dict) -> dict:
    """Score one raw applicant dict -> PD, total score, per-feature points.

    ``top_factors`` lists the features costing the applicant the most points
    relative to that feature's best (highest-points) bin — the reason codes
    a lender would have to give for a decline.
    """
    table = points_table(model, scaler)
    detail = []
    total = 0.0
    for f in model.features:
        bin_label = model.binners[f].bin_of(applicant[f])
        row = table[(table["feature"] == f) & (table["bin"] == bin_label)].iloc[0]
        best = table[table["feature"] == f]["points"].max()
        total += row["points"]
        detail.append({
            "feature": f, "value": applicant[f], "bin": bin_label,
            "woe": float(row["woe"]), "points": float(row["points"]),
            "points_lost_vs_best_bin": float(best - row["points"]),
        })
    p_bad = float(scaler.pbad_from_score(total))
    detail_sorted = sorted(detail, key=lambda d: -d["points_lost_vs_best_bin"])
    return {
        "pd": p_bad, "score": float(total),
        "breakdown": detail,
        "top_factors": [d for d in detail_sorted if d["points_lost_vs_best_bin"] > 0][:4],
    }
