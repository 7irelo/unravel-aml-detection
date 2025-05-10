"""One evaluation suite, applied identically to every arm on a chosen partition."""

from __future__ import annotations

from typing import Any

import numpy as np

from unravel.eval import metrics as m
from unravel.eval.bootstrap import cluster_ci
from unravel.models.dataset import ModelData

CAPACITIES = (50, 100, 200, 500)


def full_volume_mask(data: ModelData, min_share_of_median: float = 0.05) -> np.ndarray:
    """Exclude days far below normal volume: the generator's laundering-only tail."""
    counts = np.bincount(data.day)
    typical = float(np.median(counts[counts > 0]))
    return (counts >= min_share_of_median * typical)[data.day]


def evaluate_scores(
    data: ModelData,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float | None,
    *,
    n_boot: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate ranking scores on one partition. ``threshold`` comes from validation A."""
    y = data.y[mask].astype(np.int64)
    s = scores[mask]
    frame = data.alert_frame(scores, mask)
    clusters = data.account[mask] * 100 + data.day[mask]
    result: dict[str, Any] = {
        "transactions": int(mask.sum()),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": m.pr_auc(y, s),
        "pr_auc_ci": cluster_ci(
            lambda idx: m.pr_auc(y[idx], s[idx]), clusters, n_boot=n_boot, seed=seed
        ),
        "rate": m.rate_metrics(y, s),
        "capacity": {str(k): m.capacity_metrics(frame, k) for k in CAPACITIES},
        "online_regret_100": m.online_regret(frame, 100),
    }
    if threshold is not None:
        predicted = (s >= threshold).astype(np.int64)
        result["threshold"] = threshold
        result["minority_f1"] = m.f1_at(y, predicted)
        result["minority_f1_ci"] = cluster_ci(
            lambda idx: m.f1_at(y[idx], predicted[idx])["f1"], clusters, n_boot=n_boot, seed=seed
        )
        result["per_pattern"] = m.per_pattern_recall(
            y, predicted, [p for p, keep in zip(data.pattern, mask, strict=True) if keep]
        )
    full_volume = mask & full_volume_mask(data)
    if full_volume.sum() < mask.sum():
        # The headline for capacity metrics: see "The tail" in docs/protocol.md.
        result["full_volume_days"] = evaluate_scores(
            data, scores, full_volume, threshold, n_boot=n_boot, seed=seed
        )
    return result


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return {
        "log_loss": m.log_loss(y, p),
        "brier": m.brier(y, p),
        "brier_skill": m.brier_skill(y, p),
        "adaptive_ece": m.adaptive_ece(y, p),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "reliability": m.reliability_table(y, p),
    }
