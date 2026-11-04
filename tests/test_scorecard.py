"""Tests for Credit Scorecard Pro. Toy-data tests use hand-computed values;
integration tests run the real pipeline (cached after first build)."""
import numpy as np
import pandas as pd
import pytest

from scorecard.binning import WoeBinner
from scorecard.economics import breakeven_pd, profit_curve
from scorecard.model import LogisticRegressionScratch
from scorecard.scaling import ScoreScaler, points_table, score_applicant
from scorecard.validation import (auc_score, brier_score, ks_statistic,
                                  population_stability_index)


# ---------- WOE / IV on toy data with known values ----------
def _toy():
    # bin A: 80 good / 20 bad; bin B: 20 good / 80 bad -> perfectly readable WOE
    x = pd.Series(["A"] * 100 + ["B"] * 100)
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
    return x, y


def test_woe_sign_and_symmetry():
    x, y = _toy()
    b = WoeBinner().fit(x, y, numeric=False)
    # smoothed WOE_A = ln((80.5/101... )) approx ln(4) = 1.386 up to smoothing
    assert b.woe_map_["A"] == pytest.approx(np.log(4), abs=0.05)
    assert b.woe_map_["B"] == pytest.approx(-np.log(4), abs=0.05)


def test_iv_toy_value_and_nonnegative():
    x, y = _toy()
    b = WoeBinner().fit(x, y, numeric=False)
    assert b.iv_ > 0.8  # (0.6)*ln4 * 2 ≈ 1.66 unsmoothed; smoothing lowers slightly
    assert (b.table_["iv_contrib"] >= -1e-12).all()


def test_iv_zero_for_uninformative_feature():
    x = pd.Series(["A", "B"] * 100)
    y = pd.Series([0, 1, 1, 0] * 50)  # each bin exactly 50% bad -> WOE 0
    b = WoeBinner().fit(x, y, numeric=False)
    assert b.iv_ == pytest.approx(0.0, abs=1e-9)


def test_numeric_binning_monotone_bad_rate():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(50, 10, 2000))
    y = pd.Series((x + rng.normal(0, 8, 2000) > 55).astype(int))
    b = WoeBinner(max_bins=5).fit(x, y, numeric=True)
    rates = b.table_["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= -1e-9)  # refined to monotone increasing


# ---------- Logistic regression (own implementation) ----------
def test_logistic_recovers_planted_coefficient():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000).reshape(-1, 1)
    p = 1 / (1 + np.exp(-(1.5 * x[:, 0] - 0.3)))
    y = (rng.random(4000) < p).astype(int)
    lr = LogisticRegressionScratch(l2=1e-4).fit(x, y)
    assert lr.coef_[0] == pytest.approx(1.5, abs=0.25)
    assert lr.intercept_ == pytest.approx(-0.3, abs=0.2)


def test_logistic_proba_in_unit_interval():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    lr = LogisticRegressionScratch().fit(X, y)
    p = lr.predict_proba(X)
    assert ((p >= 0) & (p <= 1)).all()


# ---------- Metrics ----------
def test_auc_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert auc_score(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_ks_perfect_separation():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_brier_perfect():
    assert brier_score(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0


def test_psi_identical_is_zero():
    x = np.random.default_rng(2).normal(600, 50, 500)
    assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_shifted_is_large():
    rng = np.random.default_rng(2)
    x = rng.normal(600, 50, 1000)
    z = rng.normal(500, 50, 1000)
    assert population_stability_index(x, z) > 0.25


# ---------- Scaling ----------
def test_scaling_anchor_points():
    s = ScoreScaler(base_score=600, base_odds=50, pdo=20)
    assert s.score_from_pbad(1 / 51) == pytest.approx(600.0, abs=1e-6)  # odds 50:1
    # doubling the odds (100:1) adds exactly PDO points
    assert s.score_from_pbad(1 / 101) == pytest.approx(620.0, abs=1e-6)


def test_scaling_monotone_pd_vs_score():
    s = ScoreScaler()
    ps = np.linspace(0.01, 0.9, 20)
    scores = s.score_from_pbad(ps)
    assert np.all(np.diff(scores) < 0)  # higher PD -> lower score
    back = s.pbad_from_score(scores)
    assert np.allclose(back, ps)


def test_breakeven_pd_formula():
    assert breakeven_pd(lgd=0.45, margin=0.11) == pytest.approx(0.11 / 0.56)
    assert breakeven_pd(lgd=1.0, margin=0.0) == 0.0


def test_profit_curve_endpoints():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.05, 0.10, 0.60, 0.90])
    amounts = np.array([100.0] * 4)
    curve = profit_curve(y, p, amounts, thresholds=np.array([0.0, 1.0]))
    assert curve.iloc[0]["approval_rate"] == 0.0
    assert curve.iloc[-1]["approval_rate"] == 1.0


# ---------- Integration on the real dataset ----------
@pytest.fixture(scope="module")
def results():
    from scorecard.pipeline import run_pipeline
    return run_pipeline()


def test_pipeline_auc_in_expected_range(results):
    assert 0.65 < results["test_metrics"]["auc"] < 0.90
    assert results["test_metrics"]["ks"] > 0.25


def test_pipeline_points_sum_to_score(results):
    model, scaler = results["model"], results["scaler"]
    app = results["test"].iloc[0].to_dict()
    s = score_applicant(model, scaler, app)
    assert sum(d["points"] for d in s["breakdown"]) == pytest.approx(s["score"], abs=0.6)
    p_direct = float(model.predict_proba(results["test"].iloc[[0]])[0])
    assert s["pd"] == pytest.approx(p_direct, abs=0.02)


def test_pipeline_scorer_deterministic(results):
    app = results["test"].iloc[5].to_dict()
    s1 = score_applicant(results["model"], results["scaler"], app)
    s2 = score_applicant(results["model"], results["scaler"], app)
    assert s1["score"] == s2["score"] and s1["pd"] == s2["pd"]


def test_pipeline_points_table_covers_selected_features(results):
    table = points_table(results["model"], results["scaler"])
    assert set(table["feature"]) == set(results["selected_features"])
    # within every feature, the bin with the lowest bad rate has the most points
    for f, g in table.groupby("feature"):
        assert g.loc[g["bad_rate"].idxmin(), "points"] == g["points"].max()


def test_pipeline_fairness_groups_present(results):
    groups = set(results["fairness_groups"]["group"])
    assert groups == {"male", "female"}
    assert (results["fairness_groups"]["n"] > 50).all()
