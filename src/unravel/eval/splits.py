"""Temporal splits.

The day search mirrors IBM Multi-GNN's ``data_loading.py`` so that boundaries match published work.
Validation is then divided by time into A (tuning) and B (calibration), leaving the test boundary
unchanged.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TARGET = (0.6, 0.2, 0.2)
MINUTES_PER_DAY = 1440
TRAIN, VAL_A, VAL_B, TEST = 0, 1, 2, 3


@dataclass(frozen=True)
class Split:
    """Half-open minute boundaries for train, validation A, validation B and test."""

    origin_minute: int
    train_days: int
    val_days: int
    test_days: int
    val_start: int
    val_b_start: int
    test_start: int
    end: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Split:
        return cls(**json.loads(path.read_text()))

    def assign(self, minutes: np.ndarray) -> np.ndarray:
        """Return partition codes: 0 train, 1 validation A, 2 validation B, 3 test."""
        edges = np.array([self.val_start, self.val_b_start, self.test_start])
        return np.searchsorted(edges, minutes, side="right")


def daily_counts(minutes: np.ndarray) -> tuple[int, np.ndarray]:
    origin = int(minutes.min())
    return origin, np.bincount(((minutes - origin) // MINUTES_PER_DAY).astype(np.int64))


def search_boundaries(counts: np.ndarray) -> tuple[int, int]:
    """Pick day indices ``(i, j)`` minimising the worst relative error against 60/20/20.

    Faithful to IBM's search: ``itertools.combinations`` order, with the first minimum kept.
    """
    best: tuple[int, int] | None = None
    best_score = float("inf")
    total = counts.sum()
    for i, j in itertools.combinations(range(len(counts)), 2):
        parts = (counts[:i].sum(), counts[i:j].sum(), counts[j:].sum())
        score = max(abs(p / total - t) / t for p, t in zip(parts, TARGET, strict=True))
        if score < best_score:
            best_score, best = float(score), (i, j)
    if best is None:
        raise ValueError("need at least two days of data to split")
    return best


def make_split(minutes: np.ndarray, val_b_fraction: float = 0.5) -> Split:
    """Build the split. Validation B holds the later share of validation transactions."""
    origin, counts = daily_counts(minutes)
    i, j = search_boundaries(counts)
    val_start = origin + i * MINUTES_PER_DAY
    test_start = origin + j * MINUTES_PER_DAY
    in_val = np.sort(minutes[(minutes >= val_start) & (minutes < test_start)])
    if len(in_val) == 0:
        raise ValueError("validation window is empty")
    return Split(
        origin_minute=origin,
        train_days=i,
        val_days=j - i,
        test_days=len(counts) - j,
        val_start=val_start,
        val_b_start=int(in_val[int(len(in_val) * (1 - val_b_fraction))]),
        test_start=test_start,
        end=int(minutes.max()) + 1,
    )
