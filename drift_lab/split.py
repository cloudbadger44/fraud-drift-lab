"""Chronological split, fixed once and shared by every strategy.

Windows are defined on TransactionDT (seconds) boundaries. TransactionDT starts
at 1 day, so day d covers [d*DAY, (d+1)*DAY).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from .delay import DAY


@dataclass(frozen=True)
class SplitConfig:
    train_start_day: int = 1
    val_start_day: int = 61     # train = days 1-60
    test_start_day: int = 91    # validation = days 61-90, test stream = day 91 on
    end_day: int = 183          # exclusive; data ends inside day 182
    checkpoint_days: int = 7    # evaluation checkpoint spacing in the test stream

    @property
    def train(self): return (self.train_start_day * DAY, self.val_start_day * DAY)
    @property
    def val(self): return (self.val_start_day * DAY, self.test_start_day * DAY)
    @property
    def test(self): return (self.test_start_day * DAY, self.end_day * DAY)

    def checkpoints(self) -> list[int]:
        """Checkpoint times t_k (seconds). A model built from labels released by
        t_k is scored on arrivals in [t_k, t_k + checkpoint_days)."""
        step = self.checkpoint_days * DAY
        t0, t1 = self.test
        return [t for t in range(t0, t1, step) if t + step <= t1]

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "SplitConfig":
        with open(path) as f:
            return SplitConfig(**json.load(f))


DEFAULT_SPLIT = SplitConfig()


def assign_window(time: pd.Series, cfg: SplitConfig = DEFAULT_SPLIT) -> pd.Series:
    """Label each transaction 'train' / 'val' / 'test' by its TransactionDT."""
    out = pd.Series("", index=time.index, dtype="object")
    for name, (a, b) in (("train", cfg.train), ("val", cfg.val), ("test", cfg.test)):
        out[(time >= a) & (time < b)] = name
    if (out == "").any():
        raise ValueError("transactions fall outside the configured windows")
    return out


def describe_split(df: pd.DataFrame, cfg: SplitConfig = DEFAULT_SPLIT,
                   time_col: str = "TransactionDT", label_col: str = "isFraud") -> pd.DataFrame:
    w = assign_window(df[time_col], cfg)
    g = df.groupby(w).agg(transactions=(label_col, "size"), fraud=(label_col, "sum"),
                          first_dt=(time_col, "min"), last_dt=(time_col, "max"))
    g["fraud_rate_pct"] = (100 * g["fraud"] / g["transactions"]).round(2)
    return g.loc[["train", "val", "test"]]
