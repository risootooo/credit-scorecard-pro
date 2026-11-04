"""Loader for the UCI Statlog German Credit data (see data/PROVENANCE.md)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "num_dependents", "telephone", "foreign_worker", "class",
]

NUMERIC = [
    "duration_months", "credit_amount", "installment_rate_pct",
    "residence_since", "age", "existing_credits", "num_dependents",
]
CATEGORICAL = [c for c in COLUMNS[:-1] if c not in NUMERIC]

# personal_status_sex codes (original german.doc): A91 male divorced/separated,
# A92 female divorced/separated/married, A93 male single, A94 male married/widowed,
# A95 female single. Sex is entangled with marital status in this attribute —
# the fairness module treats it with explicit caveats.
SEX_MAP = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}


def load_german_credit(path: str | Path | None = None) -> pd.DataFrame:
    """Load german.data; target column ``bad`` = 1 for a bad credit, else 0."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "german.data"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df["bad"] = (df["class"] == 2).astype(int)
    df = df.drop(columns=["class"])
    for col in NUMERIC:
        df[col] = df[col].astype(float)
    df["sex"] = df["personal_status_sex"].map(SEX_MAP)
    return df


def train_test_split(df: pd.DataFrame, test_frac: float = 0.30, seed: int = 42):
    """Stratified split on the target. Returns (train, test) with reset indices."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, group in df.groupby("bad"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        k = int(round(len(idx) * test_frac))
        test_idx.extend(idx[:k])
        train_idx.extend(idx[k:])
    return (df.loc[sorted(train_idx)].reset_index(drop=True),
            df.loc[sorted(test_idx)].reset_index(drop=True))
