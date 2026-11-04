"""Self-contained dark HTML dashboard (inline SVG, zero external refs)
and the docs/CASES.md generator — both fed by the single pipeline run."""
from __future__ import annotations

import numpy as np

from .validation import roc_curve

CSS = """
body{background:#0d1117;color:#e6edf3;font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;margin:0;padding:24px}
h1{font-size:26px;margin:0 0 4px} h2{font-size:18px;color:#7ee787;margin-top:28px}
.sub{color:#8b949e;margin-bottom:16px} .cards{display:flex;gap:12px;flex-wrap:wrap}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;min-width:140px}
.card .v{font-size:24px;font-weight:700} .card .k{color:#8b949e;font-size:12px;text-transform:uppercase}
.panel{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-top:16px}
table{border-collapse:collapse;font-size:13px} th,td{border:1px solid #30363d;padding:5px 9px;text-align:right}
th{background:#21262d} td:first-child,th:first-child{text-align:left}
svg{background:#0d1117} .warn{color:#f0b429} footer{color:#8b949e;margin-top:24px;font-size:12px}
"""


def _svg_line(xs, ys, w=520, h=300, color="#58a6ff", diag=False, xlabel="", ylabel=""):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs))
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    if x1 == x0: x1 = x0 + 1
    if y1 == y0: y1 = y0 + 1
    px = 40 + (xs - x0) / (x1 - x0) * (w - 60)
    py = (h - 36) - (ys - y0) / (y1 - y0) * (h - 56)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
    diag_line = ""
    if diag:
        dx0, dy0 = 40 + (0 - x0) / (x1 - x0) * (w - 60), (h - 36) - (0 - y0) / (y1 - y0) * (h - 56)
        dx1, dy1 = 40 + (1 - x0) / (x1 - x0) * (w - 60), (h - 36) - (1 - y0) / (y1 - y0) * (h - 56)
        diag_line = f'<line x1="{dx0:.0f}" y1="{dy0:.0f}" x2="{dx1:.0f}" y2="{dy1:.0f}" stroke="#484f58" stroke-dasharray="4"/>'
    return (f'<svg width="{w}" height="{h}">{diag_line}'
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
            f'<text x="10" y="16" fill="#8b949e" font-size="11">{ylabel}</text>'
            f'<text x="{w - 120}" y="{h - 8}" fill="#8b949e" font-size="11">{xlabel}</text></svg>')


def _svg_hist(scores, cutoff_score, w=520, h=300):
    lo, hi = np.floor(scores.min() / 20) * 20, np.ceil(scores.max() / 20) * 20
    edges = np.arange(lo, hi + 20, 20)
    counts, _ = np.histogram(scores, bins=edges)
    bw = (w - 60) / len(counts)
    bars = []
    for i, c in enumerate(counts):
        mid = 0.5 * (edges[i] + edges[i + 1])
        fill = "#3fb950" if mid >= cutoff_score else "#f85149"
        bh = c / counts.max() * (h - 56)
        bars.append(f'<rect x="{40 + i * bw + 1:.0f}" y="{(h - 36) - bh:.0f}" width="{bw - 2:.0f}" height="{bh:.0f}" fill="{fill}" opacity="0.8"/>')
    cx = 40 + (cutoff_score - lo) / (hi - lo) * (w - 60)
    bars.append(f'<line x1="{cx:.0f}" y1="10" x2="{cx:.0f}" y2="{h - 36}" stroke="#f0b429" stroke-width="2"/>')
    bars.append(f'<text x="{cx + 5:.0f}" y="22" fill="#f0b429" font-size="11">cut-off {cutoff_score:.0f} pts</text>')
    return f'<svg width="{w}" height="{h}">{"".join(bars)}</svg>'


def build_dashboard(results: dict, path: str) -> str:
    test = results["test"]
    y = test["bad"].to_numpy()
    p = results["p_test"]
    scores = results["scores_test"]
    tm = results["test_metrics"]
    fpr, tpr = roc_curve(y, p)
    cal = tm["calibration"]
    pts = results["points_table"].head(24)
    pts_rows = "".join(
        f"<tr><td>{r.feature}</td><td>{r.bin}</td><td>{r.woe:.3f}</td><td>{r.bad_rate:.1%}</td><td>{r.points:.0f}</td></tr>"
        for r in pts.itertuples())
    econ = results["economics"]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Credit Scorecard Pro — Dashboard</title><style>{CSS}</style></head><body>
<h1>Credit Scorecard Pro</h1>
<div class="sub">UCI Statlog German Credit (1,000 loans) · WOE/IV logistic scorecard · test set n={len(test)} · all figures from one pipeline run</div>
<div class="cards">
<div class="card"><div class="v">{tm['auc']:.3f}</div><div class="k">Test AUC</div></div>
<div class="card"><div class="v">{tm['ks']:.3f}</div><div class="k">Test KS</div></div>
<div class="card"><div class="v">{tm['brier']:.3f}</div><div class="k">Brier</div></div>
<div class="card"><div class="v">{results['cutoff_pd']:.3f}</div><div class="k">Cost-optimal PD cut-off</div></div>
<div class="card"><div class="v">{results['cutoff_score']:.0f}</div><div class="k">Cut-off score (pts)</div></div>
<div class="card"><div class="v">{results['score_psi_train_test']:.3f}</div><div class="k">Score PSI train→test</div></div>
</div>
<div class="panel"><h2>ROC curve (test)</h2>{_svg_line(fpr, tpr, diag=True, xlabel="FPR", ylabel="TPR")}</div>
<div class="panel"><h2>Calibration by decile (test)</h2>
{_svg_line(cal['mean_predicted_pd'], cal['observed_bad_rate'], color="#3fb950", diag=True, xlabel="predicted PD", ylabel="observed bad rate")}
<p>Hosmer–Lemeshow χ² = {tm['hl_statistic']:.1f}, p = {tm['hl_p_value']:.3f} (p &gt; 0.05 = no strong evidence of miscalibration).</p></div>
<div class="panel"><h2>Score distribution (test) — green ≥ cut-off</h2>{_svg_hist(scores, results['cutoff_score'])}</div>
<div class="panel"><h2>Economics at the two cut-offs (test, amounts in DM)</h2>
<p>LGD {econ['lgd']:.0%}, margin {econ['margin']:.0%} of exposure. Default 0.50 cut-off: approve {econ['at_default_0.50']['approval_rate']:.1%}, profit {econ['at_default_0.50']['total_profit']:,.0f}. Cost-optimal {results['cutoff_pd']:.3f}: approve {econ['at_cost_optimal']['approval_rate']:.1%}, profit {econ['at_cost_optimal']['total_profit']:,.0f}.</p></div>
<div class="panel"><h2>Scorecard points (first {len(pts)} bins)</h2>
<table><tr><th>Feature</th><th>Bin</th><th>WOE</th><th>Bad rate</th><th>Points</th></tr>{pts_rows}</table></div>
<footer>Development benchmark on 1994 German data (1,000 rows). Educational — not for real lending decisions.</footer>
</body></html>"""
    with open(path, "w") as fh:
        fh.write(html)
    return path
