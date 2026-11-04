"""Weight-of-Evidence (WOE) / Information Value (IV) binning.

Numeric features: quantile bins, then a monotonic refinement that merges
adjacent bins until the bad rate is monotone (direction chosen by the
feature's overall trend). Categorical features: one bin per category, with
rare categories (< ``min_pct`` of the sample) pooled into ``OTHER``.

WOE uses Laplace-style smoothing (+0.5) so empty good/bad cells stay finite.
Definitions (per bin):
    dist_good = goods / total_goods,  dist_bad = bads / total_bads
    WOE = ln(dist_good / dist_bad)          (positive WOE = safer bin)
    IV contribution = (dist_good - dist_bad) * WOE
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTH = 0.5


def _bin_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Input columns: bin, bad. Output adds counts, rates, WOE, IV."""
    total_good = (df["bad"] == 0).sum()
    total_bad = (df["bad"] == 1).sum()
    rows = []
    for label, g in df.groupby("bin", observed=True):
        bads = int(g["bad"].sum())
        count = len(g)
        goods = count - bads
        dist_good = (goods + SMOOTH * 0) / total_good if total_good else 0.0
        dist_bad = bads / total_bad if total_bad else 0.0
        dg = (goods + SMOOTH) / (total_good + SMOOTH * df["bin"].nunique())
        db = (bads + SMOOTH) / (total_bad + SMOOTH * df["bin"].nunique())
        woe = float(np.log(dg / db))
        rows.append({
            "bin": label, "count": count, "goods": goods, "bads": bads,
            "bad_rate": bads / count if count else 0.0,
            "dist_good": dist_good, "dist_bad": dist_bad,
            "woe": woe, "iv_contrib": (dg - db) * woe,
        })
    out = pd.DataFrame(rows)
    out.attrs["iv"] = float(out["iv_contrib"].sum())
    return out


def _monotonic_merge(edges: list[float], x: pd.Series, y: pd.Series) -> list[float]:
    """Merge adjacent bins (by dropping shared edges) until bad rate is monotone."""
    edges = list(edges)
    while len(edges) > 2:
        cats = pd.cut(x, bins=edges, include_lowest=True)
        tmp = pd.DataFrame({"bin": cats, "bad": y})
        rates = tmp.groupby("bin", observed=True)["bad"].mean().to_numpy()
        if len(rates) < 2:
            break
        direction = np.sign(rates[-1] - rates[0]) or 1.0
        diffs = np.diff(rates) * direction
        if (diffs >= -1e-12).all():
            break
        worst = int(np.argmin(diffs))  # most non-monotone adjacent pair
        del edges[worst + 1]
    return edges


class WoeBinner:
    """Fit binning for one feature; transform values to bin labels / WOE."""

    def __init__(self, max_bins: int = 5, min_pct: float = 0.02):
        self.max_bins = max_bins
        self.min_pct = min_pct

    def fit(self, x: pd.Series, y: pd.Series, numeric: bool) -> "WoeBinner":
        self.numeric = numeric
        if numeric:
            qs = np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)))
            edges = qs.tolist()
            edges[0], edges[-1] = -np.inf, np.inf
            edges = _monotonic_merge(edges, x, y)
            self.edges_ = edges
            labels = [self._num_label(edges, i) for i in range(len(edges) - 1)]
            cats = pd.cut(x, bins=edges, include_lowest=True)
            tmp = pd.DataFrame({"bin": pd.Categorical(cats, categories=cats.cat.categories),
                                "bad": y}).assign(bin=cats.astype(str))
            # rebuild with readable labels in order
            mapping = {str(c): lbl for c, lbl in zip(cats.cat.categories, labels)}
            tmp["bin"] = tmp["bin"].map(mapping)
            self.bin_order_ = labels
        else:
            freq = x.value_counts(normalize=True)
            rare = set(freq[freq < self.min_pct].index)
            pooled = x.where(~x.isin(rare), "OTHER")
            tmp = pd.DataFrame({"bin": pooled.astype(str), "bad": y})
            self.rare_ = rare
            self.bin_order_ = sorted(tmp["bin"].unique())
        self.table_ = _bin_stats(tmp)
        self.table_ = self.table_.set_index("bin").reindex(self.bin_order_).reset_index()
        self.table_.attrs["iv"] = float(self.table_["iv_contrib"].sum())
        self.iv_ = self.table_.attrs["iv"]
        self.woe_map_ = dict(zip(self.table_["bin"], self.table_["woe"]))
        return self

    @staticmethod
    def _num_label(edges, i):
        lo, hi = edges[i], edges[i + 1]
        lo_s = "-inf" if lo == -np.inf else f"{lo:g}"
        hi_s = "+inf" if hi == np.inf else f"{hi:g}"
        return f"({lo_s}, {hi_s}]"

    def bin_of(self, value) -> str:
        if self.numeric:
            for i in range(len(self.edges_) - 1):
                if self.edges_[i] < value <= self.edges_[i + 1] or (
                        i == 0 and value <= self.edges_[1]):
                    return self._num_label(self.edges_, i)
            return self._num_label(self.edges_, len(self.edges_) - 2)
        v = str(value)
        return "OTHER" if (v in self.rare_ or v not in self.woe_map_) else v

    def transform(self, x: pd.Series) -> pd.Series:
        bins = x.map(self.bin_of)
        return bins.map(self.woe_map_).astype(float)


def fit_binners(train: pd.DataFrame, features: list[str], numeric_features: list[str],
                target: str = "bad", **kw) -> dict[str, WoeBinner]:
    binners = {}
    for f in features:
        binners[f] = WoeBinner(**kw).fit(train[f], train[target], numeric=f in numeric_features)
    return binners


def iv_table(binners: dict[str, WoeBinner]) -> pd.DataFrame:
    rows = [{"feature": f, "iv": b.iv_,
             "strength": ("strong" if b.iv_ >= 0.30 else "medium" if b.iv_ >= 0.10
                          else "weak" if b.iv_ >= 0.02 else "negligible")}
            for f, b in binners.items()]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
