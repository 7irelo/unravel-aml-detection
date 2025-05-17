"""Compare online vectors with offline Parquet: the check that training matches live scoring."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from unravel.features.engine import FEATURE_NAMES
from unravel.streaming.messages import decode_vector


@dataclass
class ParityReport:
    offline_rows: int
    online_unique: int
    online_messages: int
    missing: int
    extra: int
    vectors_compared: int
    vectors_exact: int
    vectors_within_tolerance: int
    conflicting_redeliveries: int
    max_abs_difference: float
    worst_feature: str | None
    tolerance: float

    @property
    def passed(self) -> bool:
        return (
            self.missing == 0
            and self.extra == 0
            and self.conflicting_redeliveries == 0
            and self.vectors_within_tolerance == self.vectors_compared
            and self.vectors_compared == self.offline_rows
        )

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


class PartMatrix:
    """Read rows of Parquet feature parts on demand, holding one part in memory at a time.

    Online messages arrive in nearly the same order as the parts, so this stays efficient.
    """

    def __init__(self, directory: Path, columns: list[str]) -> None:
        self.parts = sorted(directory.glob("part-*.parquet"))
        self.columns = columns
        sizes = [pl.scan_parquet(p).select(pl.len()).collect().item() for p in self.parts]
        self.starts = np.r_[0, np.cumsum(sizes)]
        self.loaded = -1
        self.current = np.zeros((0, len(columns)))

    def __getitem__(self, row: int) -> np.ndarray:
        part = int(np.searchsorted(self.starts, row, side="right")) - 1
        if part != self.loaded:
            frame = pl.read_parquet(self.parts[part], columns=self.columns)
            self.current = frame.select(self.columns).to_numpy()
            self.loaded = part
        return self.current[row - self.starts[part]]


def _digest(payload: bytes) -> bytes:
    return hashlib.blake2b(payload, digest_size=8).digest()


def compare(
    offline_ids: list[str],
    offline_matrix: np.ndarray | PartMatrix,
    online_messages: Iterable[bytes],
    tolerance: float = 1e-9,
) -> ParityReport:
    """Every offline row needs exactly one online vector. Redeliveries must be byte-for-byte equal."""
    position = {tx_id: i for i, tx_id in enumerate(offline_ids)}
    seen: dict[str, bytes] = {}  # digests, so full-scale runs don't hold every payload
    messages = conflicts = extra = exact = within = 0
    worst = 0.0
    worst_index: int | None = None
    for payload in online_messages:
        messages += 1
        tx_id, _, vector = decode_vector(payload)
        digest = _digest(payload)
        earlier = seen.get(tx_id)
        if earlier is not None:
            conflicts += earlier != digest
            continue
        seen[tx_id] = digest
        row = position.get(tx_id)
        if row is None:
            extra += 1
            continue
        online = np.asarray(vector, dtype=np.float64)
        offline = offline_matrix[row]
        both_nan = np.isnan(online) & np.isnan(offline)
        difference = np.abs(online - offline)
        difference[both_nan] = 0.0
        # A NaN on only one side is a definite mismatch.
        difference[np.isnan(difference)] = np.inf
        peak = float(difference.max())
        scale = np.maximum(1.0, np.nan_to_num(np.abs(offline)))
        exact += peak == 0.0
        within += bool(np.all(difference <= tolerance * scale))
        if peak > worst:
            worst, worst_index = peak, int(difference.argmax())
    compared = len(seen) - extra
    return ParityReport(
        offline_rows=len(offline_ids),
        online_unique=len(seen),
        online_messages=messages,
        missing=len(offline_ids) - compared,
        extra=extra,
        vectors_compared=compared,
        vectors_exact=exact,
        vectors_within_tolerance=within,
        conflicting_redeliveries=conflicts,
        max_abs_difference=worst,
        worst_feature=None if worst_index is None else FEATURE_NAMES[worst_index],
        tolerance=tolerance,
    )
