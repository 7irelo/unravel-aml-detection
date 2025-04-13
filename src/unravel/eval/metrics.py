"""Detection metrics for extreme imbalance, including capacity metrics in three alert units."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-15


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, computed the same way as ``sklearn.metrics.average_precision_score``."""
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    s_sorted = scores[order]
    # Tied scores share one operating point, so only the last index of each tie group counts.
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), len(y) - 1]
    tp = np.cumsum(y_sorted)[last_of_group]
    precision = tp / (last_of_group + 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def f1_at(y: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((predicted == 1) & (y == 1)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising minority F1. Fit on validation A, then applied unchanged to test."""
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    s_sorted = scores[order]
    positives = y.sum()
    if positives == 0:
        return float("inf")
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), len(y) - 1]
    tp = np.cumsum(y_sorted)[last_of_group]
    predicted = last_of_group + 1
    f1 = 2 * tp / (predicted + positives)
    return float(s_sorted[last_of_group[int(np.argmax(f1))]])


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_skill(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> float:
    """Skill against always predicting the base rate. Zero means no better than the prior."""
    rate = float(y.mean()) if base_rate is None else base_rate
    reference = float(np.mean((rate - y) ** 2))
    return 1 - brier(y, p) / reference if reference > 0 else float("nan")


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Reliability on quantile bins, with the top of the range split on a log scale.

    Equal-width bins put nearly every transaction in the first bin at this prevalence, which makes
    standard ECE look excellent regardless of quality.
    """
    if len(y) == 0:
        return []
    # Edges from the score distribution, denser towards the high-risk tail.
    tail = 1 - np.logspace(0, -5, bins + 1)
    edges = np.unique(np.quantile(p, tail))
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    table = []
    for b in range(max(len(edges) - 1, 1)):
        mask = index == b
        n = int(mask.sum())
        if n:
            table.append(
                {
                    "low": float(edges[b]),
                    "high": float(edges[min(b + 1, len(edges) - 1)]),
                    "n": float(n),
                    "mean_predicted": float(p[mask].mean()),
                    "observed": float(y[mask].mean()),
                }
            )
    return table


def adaptive_ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = reliability_table(y, p, bins)
    total = sum(row["n"] for row in table)
    return float(
        sum(row["n"] * abs(row["mean_predicted"] - row["observed"]) for row in table) / total
    )


@dataclass(frozen=True)
class AlertFrame:
    """Everything capacity metrics need, one entry per transaction."""

    y: np.ndarray
    scores: np.ndarray
    day: np.ndarray
    account: np.ndarray  # integer-coded sending account
    usd: np.ndarray
    pattern_instance: np.ndarray  # -1 where not part of a named pattern


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= len(scores):
        return np.arange(len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top], kind="mergesort")]


def capacity_metrics(frame: AlertFrame, k: int) -> dict[str, float | list[dict[str, float]]]:
    """Precision, recall and value at ``k`` alerts per day, for each day and pooled.

    ``recall_ceiling`` is the best any model could reach at this capacity, which recall@k needs
    for context. Pooled numbers weight each day by its alerts.
    """
    per_day: list[dict[str, float]] = []
    hits = alerts = positives = 0
    value_hit = value_total = 0.0
    case_hits = case_alerts = case_positives = 0
    instances_hit: set[int] = set()

    for day in np.unique(frame.day):
        mask = np.flatnonzero(frame.day == day)
        y = frame.y[mask]
        scores = frame.scores[mask]
        top = _top_k_indices(scores, k)
        day_hits = int(y[top].sum())
        day_positives = int(y.sum())
        hits += day_hits
        alerts += len(top)
        positives += day_positives

        usd = np.nan_to_num(frame.usd[mask])
        value_hit += float(usd[top][y[top] == 1].sum())
        value_total += float(usd[y == 1].sum())

        instance = frame.pattern_instance[mask][top]
        instances_hit.update(int(i) for i in instance[instance >= 0])

        # Case unit: one account-day. A case's score is the max over its transactions.
        accounts = frame.account[mask]
        codes, inverse = np.unique(accounts, return_inverse=True)
        case_score = np.full(len(codes), -np.inf)
        np.maximum.at(case_score, inverse, scores)
        case_label = np.zeros(len(codes), dtype=np.int64)
        np.maximum.at(case_label, inverse, y.astype(np.int64))
        top_cases = _top_k_indices(case_score, k)
        day_case_hits = int(case_label[top_cases].sum())
        case_hits += day_case_hits
        case_alerts += len(top_cases)
        case_positives += int(case_label.sum())

        per_day.append(
            {
                "day": float(day),
                "transactions": float(len(mask)),
                "positives": float(day_positives),
                "alerts": float(len(top)),
                "precision": day_hits / len(top) if len(top) else 0.0,
                "recall": day_hits / day_positives if day_positives else float("nan"),
                "recall_ceiling": min(1.0, k / day_positives) if day_positives else float("nan"),
                "case_precision": day_case_hits / len(top_cases) if len(top_cases) else 0.0,
            }
        )

    all_instances = np.unique(frame.pattern_instance[frame.pattern_instance >= 0])
    ceiling = sum(min(k, row["positives"]) for row in per_day) / positives if positives else 0.0
    return {
        "k": float(k),
        "days": float(len(per_day)),
        "precision": hits / alerts if alerts else 0.0,
        "recall": hits / positives if positives else 0.0,
        "recall_ceiling": float(ceiling),
        "case_precision": case_hits / case_alerts if case_alerts else 0.0,
        "case_recall": case_hits / case_positives if case_positives else 0.0,
        "pattern_hit_rate": len(instances_hit) / len(all_instances)
        if len(all_instances)
        else float("nan"),
        "value_captured": value_hit / value_total if value_total else 0.0,
        "per_day": per_day,
    }


def rate_metrics(y: np.ndarray, scores: np.ndarray, rate: float = 0.0002) -> dict[str, float]:
    """Precision at a fixed share of volume, which stays meaningful with few test days."""
    k = max(1, round(rate * len(y)))
    top = _top_k_indices(scores, k)
    hits = int(y[top].sum())
    return {
        "rate": rate,
        "k": float(k),
        "precision": hits / k,
        "recall": hits / int(y.sum()) if y.sum() else float("nan"),
    }


def per_pattern_recall(
    y: np.ndarray, flagged: np.ndarray, patterns: list[str | None]
) -> dict[str, dict[str, float]]:
    """Recall of flagged transactions per pattern. Positives with no pattern count as ``natural``."""
    out: dict[str, dict[str, float]] = {}
    names = np.array([p if p is not None else "natural" for p in patterns], dtype=object)
    positive = y == 1
    for name in sorted(set(names[positive].tolist())):
        mask = positive & (names == name)
        out[str(name)] = {"n": float(mask.sum()), "recall": float(flagged[mask].mean())}
    return out


def online_regret(
    frame: AlertFrame,
    k: int,
    warmup_days: int = 1,
    quantile_window: int = 3,
    evaluate_from_day: int | None = None,
) -> dict[str, float]:
    """Compare a live policy against the oracle top-k, which needs the finished day to compute.

    The live policy uses a threshold set from earlier days' score distributions, then a queue that
    keeps the ``k`` highest scores up to the point of each day's cut-off. It is evaluated only on
    days that have enough history.
    """
    days = np.unique(frame.day)
    oracle_hits = live_hits = live_alerts = oracle_alerts = 0
    thresholds: list[float] = []
    evaluated = 0
    for index, day in enumerate(days):
        mask = frame.day == day
        scores = frame.scores[mask]
        y = frame.y[mask]
        counted = evaluate_from_day is None or day >= evaluate_from_day
        if index >= warmup_days and thresholds and counted:
            threshold = float(np.mean(thresholds[-quantile_window:]))
            live = np.flatnonzero(scores >= threshold)
            if len(live) > k:  # the analyst queue is full: arrivals past capacity don't get worked
                live = live[:k]
            live_hits += int(y[live].sum())
            live_alerts += len(live)
            top = _top_k_indices(scores, k)
            oracle_hits += int(y[top].sum())
            oracle_alerts += len(top)
            evaluated += 1
        top = _top_k_indices(scores, k)
        thresholds.append(float(scores[top].min()) if len(top) else float("inf"))
    return {
        "days_evaluated": float(evaluated),
        "oracle_precision": oracle_hits / oracle_alerts if oracle_alerts else float("nan"),
        "live_precision": live_hits / live_alerts if live_alerts else float("nan"),
        "live_alerts_per_day": live_alerts / evaluated if evaluated else float("nan"),
        "true_positives_lost": float(oracle_hits - live_hits),
    }
