# Data Provenance — UCI Statlog (German Credit Data)

- **Dataset:** Statlog (German Credit Data), UCI Machine Learning Repository, dataset id 144.
- **Source URL:** https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
  (archive page: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Downloaded:** 2026-11-04 (UTC), file `german.data` extracted unmodified from the official zip
  (the zip also contains `german.data-numeric` and `german.doc`, the original documentation,
  kept here as `german.doc.txt`).
- **Size:** 1,000 instances, 20 attributes (7 numeric, 13 categorical) + class label.
  Class distribution in this file: 700 "good" (class 1), 300 "bad" (class 2) — a 30% bad rate.
- **Original donor:** Professor Dr. Hans Hofmann, Institut für Statistik und Ökonometrie,
  Universität Hamburg.
- **Citation (as requested by UCI):** Hofmann, H. (1994). *Statlog (German Credit Data)*
  [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- **Cost matrix (from the original documentation):** misclassifying a bad credit as good
  costs 5× misclassifying a good credit as bad. This asymmetry motivates the
  cost-optimal cut-off analysis in `docs/CASES.md` (Case A).
- **Known limitations:** collected in Germany in the 1970s–1990s (amounts in Deutsche Mark);
  only 1,000 rows; the `personal_status_sex` attribute entangles sex with marital status
  (see the fairness audit caveats). Used here strictly as a public development benchmark —
  this scorecard must never be used for real lending decisions.
