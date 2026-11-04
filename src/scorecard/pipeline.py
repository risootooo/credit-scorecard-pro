"""End-to-end pipeline: bin -> fit -> scale -> validate -> economics ->
fairness -> drift. Returns one results dict used by the CLI, the dashboard
and docs/CASES.md, so every number in the repo comes from the same run."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import economics, fairness, validation
from .binning import fit_binners, iv_table
from .data import CATEGORICAL, NUMERIC, load_german_credit, train_test_split
from .model import cross_validated_auc, fit_scorecard
from .scaling import ScoreScaler, points_table

FEATURES = ["checking_status", "duration_months", "credit_history", "purpose",
            "credit_amount", "savings", "employment_since", "installment_rate_pct",
            "personal_status_sex", "other_debtors", "property", "age",
            "other_installment_plans", "housing", "existing_credits", "job"]
IV_FLOOR = 0.02
SEED = 42


def run_pipeline(seed: int = SEED) -> dict:
    df = load_german_credit()
    train, test = train_test_split(df, seed=seed)
    binners = fit_binners(train, FEATURES, NUMERIC)
    ivs = iv_table(binners)
    selected = ivs.loc[ivs["iv"] >= IV_FLOOR, "feature"].tolist()
    if "personal_status_sex" not in selected:
        pass  # it is retained or not purely on IV; fairness uses model PD either way
    model = fit_scorecard(train, binners, selected)
    # Scorecard hygiene: a WOE feature whose fitted coefficient is positive
    # contradicts its own binning (higher WOE must mean lower risk). Drop
    # sign-flipped features and refit — standard development practice.
    coefs = model.coefficient_table().set_index("feature")["coefficient"]
    flipped = [f for f in selected if coefs[f] > 0]
    if flipped:
        selected = [f for f in selected if f not in flipped]
        model = fit_scorecard(train, binners, selected)
    scaler = ScoreScaler()

    p_train = model.predict_proba(train)
    p_test = model.predict_proba(test)
    s_test = scaler.score_from_pbad(p_test)
    s_train = scaler.score_from_pbad(p_train)

    cv_auc = cross_validated_auc(train, lambda tr, f, nf: fit_binners(tr, f, nf),
                                 selected, NUMERIC)
    econ = economics.cutoff_report(test["bad"].to_numpy(), p_test,
                                   test["credit_amount"].to_numpy())
    cutoff = econ["breakeven_pd"]
    cutoff_score = float(scaler.score_from_pbad(cutoff))

    fair = fairness.group_metrics(test, p_test, cutoff)
    bands = fairness.score_band_decomposition(test, s_test, p_test, cutoff_score)
    score_psi = validation.population_stability_index(s_train, s_test)
    drift = fairness.drift_scenario(test, model, scaler)

    return {
        "seed": seed,
        "n_train": len(train), "n_test": len(test),
        "bad_rate_train": float(train["bad"].mean()),
        "bad_rate_test": float(test["bad"].mean()),
        "iv_table": ivs,
        "selected_features": selected,
        "dropped_sign_flipped": flipped,
        "coefficients": model.coefficient_table(),
        "points_table": points_table(model, scaler),
        "model": model, "scaler": scaler, "train": train, "test": test,
        "train_metrics": validation.validation_summary(train["bad"].to_numpy(), p_train),
        "test_metrics": validation.validation_summary(test["bad"].to_numpy(), p_test),
        "cv_auc_train": cv_auc,
        "p_test": p_test, "scores_test": s_test, "p_train": p_train,
        "score_psi_train_test": score_psi,
        "economics": econ, "cutoff_pd": cutoff, "cutoff_score": cutoff_score,
        "fairness_groups": fair,
        "fairness_bands": bands,
        "drift": drift,
    }
