"""Cluster bootstrap. Transactions in one pattern are correlated, so an IID bootstrap would give
falsely narrow intervals. Resampling whole clusters (account-days) keeps that correlation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cluster_bootstrap_indices(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one resample: pick clusters with replacement and return all their member indices."""
    order = np.argsort(clusters, kind="mergesort")
    sorted_clusters = clusters[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_clusters)) + 1]
    ends = np.r_[starts[1:], len(clusters)]
    chosen = rng.integers(0, len(starts), size=len(starts))
    lengths = ends[chosen] - starts[chosen]
    # Vectorised concatenation of the chosen slices of ``order``.
    out_offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    positions = np.repeat(starts[chosen] - out_offsets, lengths) + np.arange(lengths.sum())
    return order[positions]


def cluster_ci(
    metric: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile interval. ``metric`` receives resampled indices and returns a number."""
    rng = np.random.default_rng(seed)
    values = np.array([metric(cluster_bootstrap_indices(clusters, rng)) for _ in range(n_boot)])
    values = values[~np.isnan(values)]
    return {
        "low": float(np.quantile(values, alpha / 2)),
        "high": float(np.quantile(values, 1 - alpha / 2)),
        "se": float(values.std(ddof=1)),
        "n_boot": float(len(values)),
    }


def paired_difference_ci(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Interval for ``a - b`` on identical resamples, rather than comparing separate intervals."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        indices = cluster_bootstrap_indices(clusters, rng)
        deltas.append(metric_a(indices) - metric_b(indices))
    values = np.array(deltas)
    values = values[~np.isnan(values)]
    return {
        "mean_difference": float(values.mean()),
        "low": float(np.quantile(values, alpha / 2)),
        "high": float(np.quantile(values, 1 - alpha / 2)),
        "probability_positive": float((values > 0).mean()),
    }
