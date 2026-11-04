"""Cut-off economics: expected-loss framework.

Per euro lent, approving an applicant with default probability ``p`` earns
the net margin ``MARGIN`` if the loan performs and loses ``LGD`` (loss given
default, as a fraction of exposure) if it defaults:

    EV(p) = (1 - p) * MARGIN - p * LGD

Break-even: p* = MARGIN / (MARGIN + LGD). With the stated assumptions
(MARGIN = 11% of exposure earned over the loan life, LGD = 45%) p* ≈ 0.196 —
far below the naive 0.50 default. Assumptions are explicit and editable;
the profit curve is also computed empirically over thresholds so the
analytic answer can be checked against the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LGD = 0.45
MARGIN = 0.11


def breakeven_pd(lgd: float = LGD, margin: float = MARGIN) -> float:
    return margin / (margin + lgd)


def profit_curve(y: np.ndarray, p: np.ndarray, amounts: np.ndarray,
                 lgd: float = LGD, margin: float = MARGIN,
                 thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Total expected profit (in amount units) if approving all p <= t."""
    y = np.asarray(y)
    p = np.asarray(p, float)
    amounts = np.asarray(amounts, float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    ev = np.where(y == 1, -lgd * amounts, margin * amounts)
    rows = []
    for t in thresholds:
        approved = p <= t
        rows.append({
            "threshold": float(t),
            "approval_rate": float(approved.mean()),
            "approved_bad_rate": float(y[approved].mean()) if approved.any() else 0.0,
            "total_profit": float(ev[approved].sum()),
            "profit_per_applicant": float(ev[approved].sum() / len(y)),
        })
    return pd.DataFrame(rows)


def cutoff_report(y: np.ndarray, p: np.ndarray, amounts: np.ndarray,
                  lgd: float = LGD, margin: float = MARGIN) -> dict:
    p_star = breakeven_pd(lgd, margin)
    curve = profit_curve(y, p, amounts, lgd, margin)
    empirical_best = curve.loc[curve["total_profit"].idxmax()]

    def at(t):
        approved = p <= t
        ev = np.where(y == 1, -lgd * amounts, margin * amounts)
        return {
            "threshold": float(t),
            "approval_rate": float(approved.mean()),
            "approved_bad_rate": float(y[approved].mean()) if approved.any() else 0.0,
            "good_rejected": int(((y == 0) & ~approved).sum()),
            "bad_accepted": int(((y == 1) & approved).sum()),
            "total_profit": float(ev[approved].sum()),
        }

    return {
        "lgd": lgd, "margin": margin, "breakeven_pd": p_star,
        "at_default_0.50": at(0.50),
        "at_cost_optimal": at(p_star),
        "empirical_best": {
            "threshold": float(empirical_best["threshold"]),
            "total_profit": float(empirical_best["total_profit"]),
            "approval_rate": float(empirical_best["approval_rate"]),
        },
        "curve": curve,
    }
