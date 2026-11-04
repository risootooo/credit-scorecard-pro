# Credit Scorecard Pro

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)
![Data](https://img.shields.io/badge/data-UCI%20German%20Credit%20(1%2C000%20loans)-blue)
![Method](https://img.shields.io/badge/method-WOE%2FIV%20%2B%20logistic%20scorecard-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**A bank-grade retail credit scorecard, built the way scorecards are actually built — WOE/IV binning, a logistic model you can read coefficient-by-coefficient, points scaling, calibration, cut-off economics and a fairness audit — on the real UCI German Credit benchmark, with the logistic regression implemented from scratch (no sklearn).**

> Demo report: [`docs/dashboard.html`](docs/dashboard.html) (self-contained, offline) · Case studies: [`docs/CASES.md`](docs/CASES.md) · Deployable points table: [`results/scorecard_points.csv`](results/scorecard_points.csv)

## Headline results (single pipeline run, seed 42, 70/30 stratified split)

| Metric | Value |
|---|---|
| Test AUC / KS / Gini | **0.786 / 0.486 / 0.572** |
| Train AUC · 5-fold CV AUC | 0.830 · 0.787 |
| Brier score (test) | 0.168 |
| Hosmer–Lemeshow p | 0.051 (borderline — no strong miscalibration verdict either way on 300 rows) |
| Score PSI train → test | 0.023 (stable) |
| Cost-optimal PD cut-off | **0.196** (score 528) vs the naive 0.50 |
| Profit at 0.50 → at cost-optimal | **−11,120 DM → +16,248 DM** on the 300-applicant test set (≈ €46,600 per 1,000 applicants) |
| Fairness: approval female vs male | 32.6% (n=92) vs 46.2% (n=208) — gap decomposed in Case B |

Strongest predictors by IV: checking status 0.663, credit history 0.325, duration 0.229, purpose 0.166, savings 0.129. 13 of 16 candidate features passed the IV ≥ 0.02 screen.

*Comparison with an earlier internal cut of this analysis (AUC ≈ 0.745, KS ≈ 0.396, threshold ≈ 0.20, approval 36.5% vs 51.0%): the analytic threshold agrees (0.196 vs 0.20 — same cost formula); discrimination is higher here (different split seed, IV-screened WOE feature set, sign-consistency refit); the fairness gap has the same direction and a similar size on the hold-out (32.6/46.2) vs their full-sample figures. Small samples move these numbers — that is itself one of the findings.*

## The three cases (docs/CASES.md)

- **Case A — the 0.50-threshold mistake.** At a 30% base bad rate and 45% LGD, the textbook 0.50 cut-off approves 76% of applicants and *loses money*. The cost-optimal cut-off is stricter, not looser: it trades 69 more rejected good loans for 33 avoided bads, because one bad costs ~4× one good's margin.
- **Case B — fairness audit.** Raw approval gap 13.5 pp (adverse impact ratio 0.71). Decomposition: approval is a deterministic function of the score, and re-weighting women onto the male score-band distribution lifts female approval to ≈ 48.5% — the gap is a score-distribution gap, not cut-off treatment. The open question is calibration by group, examined band by band (with small-cell caveats printed, not buried).
- **Case C — stability & drift.** A 20% credit-amount inflation is absorbed by WOE binning (score PSI 0.004); a one-grade deterioration in checking status — the top-IV feature — cuts approval 42.0% → 35.0% with no model change (PSI 0.029). Monitor features, not just scores.

## Methodology

- **Binning:** numeric features get quantile bins refined until the bad rate is monotone; categorical features are pooled only when rare (<2%). WOE = ln(distribution of goods / distribution of bads) per bin, Laplace-smoothed; IV = Σ (dist_good − dist_bad) × WOE.
- **Model:** logistic regression on WOE features, implemented from scratch (L-BFGS via `scipy.optimize`, analytic gradient, ridge on slopes). Any feature whose fitted coefficient contradicts its WOE direction is dropped and the model refit — standard scorecard hygiene, applied and logged.
- **Scaling:** `score = offset + factor × ln(odds)`, `factor = PDO / ln 2`, `offset = base_score − factor × ln(base_odds)`; here 600 points at 50:1 odds, PDO 20. Points are allocated per bin (intercept shared equally), so the CSV points table *is* the model — sum the bins, get the score, invert to the PD.
- **Validation:** rank-based AUC, KS, Gini, Brier, decile calibration with Hosmer–Lemeshow, train→test PSI. Binning is refit inside every CV fold (no leakage).
- **Economics:** EV(approve) = (1−p)·margin − p·LGD with margin 11% and LGD 45% of exposure (stated, editable assumptions in `economics.py`) → break-even PD = margin / (margin + LGD) = 0.196. An empirical profit curve is computed alongside; its test-set argmax (0.299) differs from the analytic cut-off, and the repo explains why the analytic one is the defensible choice.

## Quickstart

```bash
pip install -r requirements.txt
python main.py            # full pipeline: metrics, dashboard, CASES.md, 3 sample applicants
python -m pytest tests -q # 20 tests
```

Sample output (actual run):

```
Strong:      PD 0.008 | score 628 | APPROVE
Borderline:  PD 0.232 | score 522 | DECLINE (cut-off 528 — top factors: purpose, credit_history, checking_status)
Weak:        PD 0.978 | score 378 | DECLINE
```

Each scored applicant gets a points breakdown and reason codes (points lost vs the feature's best bin).

## Data provenance

Statlog (German Credit Data), UCI Machine Learning Repository #144 — downloaded unmodified from the official UCI zip on 2026-11-04; full provenance, citation (Hofmann, 1994; doi:10.24432/C5NC77) and the original cost-matrix note are in [`data/PROVENANCE.md`](data/PROVENANCE.md).

## Structure

```
src/scorecard/  binning.py (WOE/IV) · model.py (logistic from scratch) · scaling.py (points)
                validation.py (AUC/KS/calibration/PSI) · economics.py (cut-off) ·
                fairness.py (group audit + drift) · pipeline.py · report.py (dashboard)
docs/           dashboard.html · CASES.md
results/        summary.json · scorecard_points.csv · fairness_bands.csv
tests/          20 tests (toy-data WOE maths with known values, planted-coefficient recovery, scaling identities, PSI, end-to-end)
```

## Limitations — read before believing any of it

- **1,000 rows, 1994, Germany, Deutsche Mark.** This is a development benchmark, not a lending model. Test metrics carry wide uncertainty (the HL test can't even give a clean verdict at n=300).
- The fairness attribute (`personal_status_sex`) entangles sex with marital status, and the female hold-out is 92 people. The audit demonstrates method, not a verdict on any real portfolio.
- WOE binning on the same data that fits the model can flatter in-sample metrics; the CV number (0.787) refits binning per fold and is the honest headline alongside test AUC.
- Amounts drive the € profit figures through stated margin/LGD assumptions — change the assumptions in `economics.py` and every conclusion is recomputed, which is the correct way to disagree with them.

*Educational project. Not financial advice, and never to be used for real credit decisions.*
