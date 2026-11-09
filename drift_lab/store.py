"""The shared data-access layer.

Every strategy gets data only through a StreamView(as_of=t). A view exposes
  - features of transactions that have already ARRIVED (time < t), and
  - labels that have already been RELEASED (release_time < t).
Ground truth for scoring is reachable only through Evaluator, which strategies
never receive. One code path, one release rule, for batch, sliding-window and
online learning.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class LabelStream:
    def __init__(self, df: pd.DataFrame, policy, seed: int,
                 time_col: str = "TransactionDT", label_col: str = "isFraud"):
        if not df[time_col].is_monotonic_increasing:
            raise ValueError("df must be sorted by transaction time")
        self.policy, self.seed = policy, seed
        self.time_col, self.label_col = time_col, label_col
        self._time = df[time_col].to_numpy(dtype=np.int64)
        self._y = df[label_col].to_numpy()
        self._X = df.drop(columns=[label_col]).reset_index(drop=True)
        rng = np.random.default_rng(seed)
        self._release = self._time + policy.draw(len(df), rng)
        self._by_release = np.argsort(self._release, kind="stable")
        self._sorted_release = self._release[self._by_release]

    def __len__(self) -> int:
        return len(self._time)

    # -- strategy-facing ---------------------------------------------------
    def view(self, as_of: int) -> "StreamView":
        return StreamView(self, int(as_of))

    # -- evaluation-facing -------------------------------------------------
    def evaluator(self) -> "Evaluator":
        return Evaluator(self)

    # -- internals ---------------------------------------------------------
    def _arrived(self, lo: int, hi: int) -> np.ndarray:
        a = np.searchsorted(self._time, lo, side="left")
        b = np.searchsorted(self._time, hi, side="left")
        return np.arange(a, b)

    def _released_idx(self, as_of: int, since_release: int | None = None) -> np.ndarray:
        # A label is visible at as_of only if release_time < as_of (strict), so a
        # transaction arriving exactly at a checkpoint is never in its own training set.
        hi = np.searchsorted(self._sorted_release, as_of, side="left")
        lo = 0 if since_release is None else np.searchsorted(
            self._sorted_release, since_release, side="left")
        return self._by_release[lo:hi]


class StreamView:
    """What a strategy may see at time `as_of` (seconds)."""

    def __init__(self, store: LabelStream, as_of: int):
        self._s, self.as_of = store, as_of

    def labelled(self, window_days: float | None = None):
        """(X, y) of every transaction whose label is released by as_of, ordered by
        transaction time. window_days keeps only transactions that arrived within
        the last `window_days` days (sliding-window retraining); None = all history
        (batch retraining)."""
        idx = np.sort(self._s._released_idx(self.as_of))
        if window_days is not None:
            idx = idx[self._s._time[idx] >= self.as_of - window_days * 86_400]
        return self._s._X.iloc[idx], self._s._y[idx]

    def new_labels(self, since: int):
        """(X, y) of labels released in [since, as_of), ordered by release time.
        The feed for online learning: each label is delivered once, when it arrives."""
        idx = self._s._released_idx(self.as_of, since_release=since)
        return self._s._X.iloc[idx], self._s._y[idx]

    def arrivals(self, lo: int, hi: int):
        """Features (no labels) of transactions with lo <= time < hi, hi <= as_of."""
        if hi > self.as_of:
            raise ValueError("cannot read arrivals later than as_of")
        return self._s._X.iloc[self._s._arrived(lo, hi)]

    def n_labelled(self) -> int:
        return len(self._s._released_idx(self.as_of))


class Evaluator:
    """Ground truth for scoring. Not passed to strategies."""

    def __init__(self, store: LabelStream):
        self._s = store

    def window(self, lo: int, hi: int):
        idx = self._s._arrived(lo, hi)
        return self._s._X.iloc[idx], self._s._y[idx]
