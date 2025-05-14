"""Turn TreeSHAP contributions into reason codes an analyst can read.

SHAP explains why the *model* scored a transaction highly. It does not establish why the behaviour
happened, and reason codes must not be read as evidence of intent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from unravel.features.registry import describe


@dataclass(frozen=True)
class Reason:
    feature: str
    description: str
    value: float | None
    contribution: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def reason_codes(
    columns: list[str], values: np.ndarray, contributions: np.ndarray, top: int = 5
) -> list[Reason]:
    """Top features pushing this one score up. ``contributions`` excludes the base value."""
    order = np.argsort(-contributions)[:top]
    reasons = []
    for index in order:
        contribution = float(contributions[index])
        if contribution <= 0:
            break
        raw = float(values[index])
        reasons.append(
            Reason(
                feature=columns[index],
                description=describe(columns[index]),
                value=None if np.isnan(raw) else raw,
                contribution=contribution,
            )
        )
    return reasons


def global_importance(columns: list[str], contributions: np.ndarray) -> list[tuple[str, float]]:
    """Mean absolute SHAP per feature, ranked. ``contributions`` excludes the base-value column."""
    importance = np.abs(contributions).mean(axis=0)
    order = np.argsort(-importance)
    return [(columns[i], float(importance[i])) for i in order]
