"""Build evaluation cases from real detector output, keeping truth away from the agent.

``EvalCase`` pairs an ``Alert`` with ground truth, but only the alert is ever passed to the graph.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from unravel.agent.tools import Alert
from unravel.alerts.queue import risk_band
from unravel.models.dataset import ModelData

Stratum = str  # a named pattern, "natural" for other laundering, or "false_positive"


@dataclass(frozen=True)
class EvalCase:
    alert: Alert
    stratum: Stratum
    is_laundering: bool
    truth_pattern: str | None  # None for false positives and natural laundering
    truth_tx_ids: tuple[str, ...]


def build_cases(
    data: ModelData,
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    *,
    per_pattern: int = 10,
    natural: int = 30,
    false_positives: int = 40,
    top_fraction: float = 0.001,
    seed: int = 0,
    model_version: str = "unknown",
) -> tuple[list[EvalCase], dict[str, int]]:
    """Sample high-scoring alerts by stratum and report the counts actually available.

    A positive is taken from the highest-scoring transaction in its pattern instance, which is what
    a capacity-bound queue would surface first.
    """
    rng = random.Random(seed)
    indices = np.flatnonzero(mask)
    order = indices[np.argsort(-raw_scores[indices], kind="mergesort")]
    cutoff = order[: max(1, int(len(indices) * top_fraction))]

    members: dict[int, list[int]] = defaultdict(list)
    for member in np.flatnonzero(data.pattern_instance >= 0).tolist():
        members[int(data.pattern_instance[member])].append(int(member))

    best_of_instance: dict[int, int] = {}
    natural_pool: list[int] = []
    false_pool: list[int] = []
    for candidate in cutoff.tolist():
        i = int(candidate)
        if data.y[i] == 1:
            instance = int(data.pattern_instance[i])
            if instance >= 0:
                best_of_instance.setdefault(instance, i)  # first seen is highest scoring
            else:
                natural_pool.append(i)
        else:
            false_pool.append(i)

    by_pattern: dict[str, list[int]] = defaultdict(list)
    for instance, i in best_of_instance.items():
        pattern = data.pattern[i]
        if pattern is not None:
            by_pattern[pattern].append(instance)

    def make(i: int, stratum: str, truth: tuple[str, ...], pattern: str | None) -> EvalCase:
        tx_id = str(data.tx_id[i])
        probability = float(probabilities[i])
        alert = Alert(
            alert_id=f"A-{tx_id}",
            tx_id=tx_id,
            as_of=int(data.minute[i]),
            probability=probability,
            band=risk_band(probability),
            model_version=model_version,
        )
        return EvalCase(alert, stratum, bool(data.y[i]), pattern, truth)

    cases: list[EvalCase] = []
    for pattern in sorted(by_pattern):
        chosen = sorted(by_pattern[pattern])
        rng.shuffle(chosen)
        for instance in chosen[:per_pattern]:
            i = best_of_instance[instance]
            truth = tuple(str(data.tx_id[m]) for m in members[instance])
            cases.append(make(i, pattern, truth, pattern))
    rng.shuffle(natural_pool)
    for i in natural_pool[:natural]:
        cases.append(make(i, "natural", (str(data.tx_id[i]),), None))
    rng.shuffle(false_pool)
    for i in false_pool[:false_positives]:
        cases.append(make(i, "false_positive", (), None))

    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        counts[case.stratum] += 1
    return cases, dict(counts)
