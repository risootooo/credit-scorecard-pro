# Case Studies — all numbers computed in the pipeline run (seed 42)

Test set: 300 applicants (30% stratified hold-out of 1,000). Amounts are Deutsche Mark in the source data; € at the fixed 1.95583 DM/€ rate.

## Case A — the 0.50-threshold mistake

A default 0.50 PD cut-off approves **76.0%** of applicants and rejects **29 good loans** in the test set. The cost-optimal cut-off (LGD 45%, margin 11% → break-even PD = 0.196) approves **42.0%**, rejects **98 good loans**, and accepts 14 bads vs 47 at 0.50.

On this portfolio the 0.50 mistake is **over-lending, not over-rejecting**: at a 30% base bad rate with 45% LGD, approving 76% of applicants loses money outright. The cost-optimal cut-off is *stricter*: it rejects 69 more good loans than the default but avoids 33 bad ones — and one avoided bad (45% of exposure lost) outweighs many rejected goods (11% margin forgone), which is the whole point of pricing the asymmetry. Scaled to 1,000 applicants, the cost-optimal cut-off earns ≈ **€46,642 more expected profit** than the 0.50 default (test-set profit -11,120 DM → 16,248 DM). The empirical profit-maximising threshold on the test set is 0.299 (profit 21,863 DM) — close to, but not exactly, the analytic break-even, which is expected sampling noise on 300 rows; the analytic cut-off is the defensible one because it comes from the cost structure, not from fitting the hold-out.

## Case B — fairness audit (sex, via personal_status_sex)

Group metrics at the cost-optimal cut-off (group sizes shown — the female group is small and this attribute entangles sex with marital status, so read with care):

| Group | n | Approval rate | Observed bad rate | AUC |
|---|---|---|---|---|
| female | 92 | 32.6% | 35.9% | 0.759 |
| male | 208 | 46.2% | 27.4% | 0.795 |

Raw approval gap: **13.5 pp** (adverse impact ratio 0.71).

**Decomposition:** approval in this system is a deterministic function of the score, so no within-score differential treatment exists by construction — the gap must come through the score distribution itself. Re-weighting: if female applicants had the *male* score-band distribution (keeping female within-band approval behaviour), female approval would be ≈ **48.5%** instead of 32.6%. Most of the raw gap is therefore a score-distribution gap, not a cut-off artefact. The remaining fairness question is calibration by group — within score bands, do both groups default at the rate the score implies? See the band table in results/fairness_bands.csv (small cells: several bands have <15 women; do not over-read single cells).

## Case C — stability and drift

Score PSI between train and test: **0.023** (< 0.10 = stable by industry convention).

**Scenario 1 — amount inflation (absorbed):** every test applicant's requested credit amount inflated ×1.2: mean score moves 518 → 517 points, score PSI = **0.004**, approval 42.0% → 41.7%. Almost nothing happens — WOE binning absorbs a 20% amount shift because most applicants stay inside their bin. Binning is a robustness feature here.

**Scenario 2 — behavioural deterioration (bites):** every 3rd applicant with a healthy checking status slips one grade (A14→A12, A13→A11) — the strongest-IV feature degrading. Score PSI = **0.029** and approval at the fixed cut-off falls 42.0% → 35.0% with no model change at all. Lesson: monitor PSI per input feature, not just the headline score — amount drift is noise, behaviour drift is signal.
