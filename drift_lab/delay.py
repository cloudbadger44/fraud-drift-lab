"""Label delay policies.

A label's release time = transaction time + delay. A policy only draws delays;
it never sees labels, so delay is independent of the fraud outcome (see the
assumptions section of the pipeline document).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DAY = 86_400  # TransactionDT is in seconds


@dataclass(frozen=True)
class NoDelay:
    """Labels are known on arrival. Used only as an oracle upper bound."""
    name: str = "none"

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(n, dtype=np.int64)


@dataclass(frozen=True)
class FixedDelay:
    """Every label is released exactly `days` after the transaction."""
    days: int

    @property
    def name(self) -> str:
        return f"fixed{self.days}"

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.days * DAY, dtype=np.int64)


@dataclass(frozen=True)
class BoundedRandomDelay:
    """Delay drawn uniformly (whole seconds) from (0, max_days days]."""
    max_days: int = 30

    @property
    def name(self) -> str:
        return f"random{self.max_days}"

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(1, self.max_days * DAY + 1, size=n, dtype=np.int64)


def policy_from_name(name: str):
    """'none', 'fixed7', 'fixed14', 'fixed30', 'random30' -> policy object."""
    if name == "none":
        return NoDelay()
    if name.startswith("fixed"):
        return FixedDelay(int(name[5:]))
    if name.startswith("random"):
        return BoundedRandomDelay(int(name[6:]))
    raise ValueError(f"unknown delay policy: {name}")
