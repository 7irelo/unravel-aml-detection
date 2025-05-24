"""Population Stability Index on features and scores, with fixed bins from the training reference.

PSI thresholds of 0.1 and 0.25 are conventions, and PSI grows with sampling noise at low volume.
``calibrate_alert_level`` sets the level from day-to-day variation within the reference itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FLOOR = 1e-5


@dataclass(frozen=True)
class Reference:
    """Binned distribution of one feature. The final bucket holds nulls and NaNs."""

    name: str
    edges: np.ndarray
    proportions: np.ndarray


def build_reference(name: str, values: np.ndarray, bins: int = 10) -> Reference:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        edges = np.array([-np.inf, np.inf])
    else:
        inner = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
        edges = np.r_[-np.inf, inner, np.inf]
    return Reference(name, edges, _proportions(values, edges))


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins + 1, dtype=np.float64)
    finite_mask = np.isfinite(values)
    index = np.searchsorted(edges, values[finite_mask], side="right") - 1
    counts[:n_bins] = np.bincount(np.clip(index, 0, n_bins - 1), minlength=n_bins)
    counts[n_bins] = (~finite_mask).sum()
    total = counts.sum()
    return counts / total if total else counts


def psi(reference: Reference, values: np.ndarray) -> float:
    actual = np.maximum(_proportions(values, reference.edges), _FLOOR)
    expected = np.maximum(reference.proportions, _FLOOR)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def calibrate_alert_level(
    reference: Reference,
    daily_values: list[np.ndarray],
    minimum: float = 0.1,
    multiple: float = 3.0,
) -> float:
    """Alert when PSI exceeds both a floor and a multiple of a typical day's PSI.

    The median is used rather than the worst day: weekends have a very different mix here, and a
    level set from the worst day would hide real changes such as a payment format disappearing.
    """
    baseline = [psi(reference, day) for day in daily_values if len(day)]
    return max(minimum, multiple * float(np.median(baseline))) if baseline else minimum


class DriftMonitor:
    """Hold references for many features and report which ones have moved."""

    def __init__(self, references: list[Reference], levels: dict[str, float] | None = None) -> None:
        self.references = references
        self.levels = levels or {}

    @classmethod
    def from_matrix(cls, names: list[str], matrix: np.ndarray, bins: int = 10) -> DriftMonitor:
        return cls([build_reference(n, matrix[:, i], bins) for i, n in enumerate(names)])

    def calibrate(self, daily_matrices: list[np.ndarray]) -> None:
        for i, reference in enumerate(self.references):
            self.levels[reference.name] = calibrate_alert_level(
                reference, [m[:, i] for m in daily_matrices]
            )

    def check(self, matrix: np.ndarray) -> dict[str, float]:
        return {r.name: psi(r, matrix[:, i]) for i, r in enumerate(self.references)}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "levels": self.levels,
                    "references": [
                        {
                            "name": r.name,
                            # JSON has no infinity, so the open outer edges are implied.
                            "edges": r.edges[1:-1].tolist(),
                            "proportions": r.proportions.tolist(),
                        }
                        for r in self.references
                    ],
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> DriftMonitor:
        body = json.loads(path.read_text())
        references = [
            Reference(
                r["name"],
                np.r_[-np.inf, np.array(r["edges"], dtype=np.float64), np.inf],
                np.array(r["proportions"], dtype=np.float64),
            )
            for r in body["references"]
        ]
        return cls(references, body["levels"])

    def alerts(self, matrix: np.ndarray, default_level: float = 0.25) -> dict[str, float]:
        return {
            name: value
            for name, value in self.check(matrix).items()
            if value > self.levels.get(name, default_level)
        }
