"""Distribution drift measures: PSI and KL divergence between a reference window and a current window.

Numeric columns are binned on the reference window's deciles, with missing values as their own bin.
Text/category columns use the reference's most frequent values, plus 'other' (unseen or rare) and 'missing'.
Both measures work on the same binned frequencies, so they are directly comparable.

PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i)
KL  = sum_i  cur_i * ln(cur_i / ref_i)            (current relative to reference)
Frequencies are floored at EPS so empty bins do not give infinite values.
Common PSI reading: < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 large shift.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-4


class Binner:
    """Fit bins on a reference column; transform any column to integer bin codes."""

    def __init__(self, ref: pd.Series, n_bins: int = 10, max_cats: int = 20):
        self.numeric = pd.api.types.is_numeric_dtype(ref)
        if self.numeric:
            x = ref.dropna().to_numpy(dtype="float64")
            self.edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(x) else np.array([])
            self.n_value_bins = len(self.edges) + 1
        else:
            vc = ref.dropna().astype(str).value_counts()
            self.cats = list(vc.index[:max_cats])
            self.n_value_bins = len(self.cats) + 1          # last value bin = 'other'
        self.n_bins = self.n_value_bins + 1                  # + missing

    def transform(self, s: pd.Series) -> np.ndarray:
        miss = s.isna().to_numpy()
        if self.numeric:
            codes = np.searchsorted(self.edges, s.to_numpy(dtype="float64", na_value=np.nan), side="right")
        else:
            lookup = {c: i for i, c in enumerate(self.cats)}
            codes = s.astype(str).map(lookup).fillna(len(self.cats)).to_numpy(dtype="int64")
        codes = np.where(miss, self.n_value_bins, codes)
        return codes.astype("int64")

    def freq(self, s: pd.Series) -> np.ndarray:
        c = np.bincount(self.transform(s), minlength=self.n_bins).astype("float64")
        return c / max(c.sum(), 1.0)


def psi_kl(ref_freq: np.ndarray, cur_freq: np.ndarray) -> tuple[float, float]:
    r = np.maximum(ref_freq, EPS); c = np.maximum(cur_freq, EPS)
    return float(np.sum((c - r) * np.log(c / r))), float(np.sum(c * np.log(c / r)))


def drift_table(df: pd.DataFrame, features: list[str], ref_mask: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """PSI and KL of every feature, for every group (e.g. week) against the reference rows.
    Returns long table: feature, group, psi, kl, n."""
    rows = []
    for f in features:
        b = Binner(df.loc[ref_mask, f])
        ref = b.freq(df.loc[ref_mask, f])
        codes = b.transform(df[f])
        for g in np.unique(group):
            m = group == g
            c = np.bincount(codes[m], minlength=b.n_bins).astype("float64")
            n = int(m.sum()); c = c / max(c.sum(), 1.0)
            psi, kl = psi_kl(ref, c)
            rows.append((f, g, psi, kl, n))
    return pd.DataFrame(rows, columns=["feature", "group", "psi", "kl", "n"])
