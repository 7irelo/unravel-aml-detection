"""Floors the ML models must beat: random, rank by amount, and handwritten monitoring rules.

The rules resemble legacy transaction monitoring. Their thresholds are chosen by judgement, not fit
to labels, so they use no training information beyond the feature definitions.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def random_scores(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def amount_scores(features: pl.DataFrame) -> np.ndarray:
    return features["log_usd"].to_numpy()


RULE_COLUMNS = [
    "log_usd",
    "src_out_cnt_24h",
    "src_out_distinct_24h",
    "dst_in_distinct_24h",
    "src_forward_ratio_24h",
    "src_out_round_ratio_24h",
    "two_paths_24h",
    "cycle_min_hops_96h",
    "src_mins_since_in",
    "self_transfer",
]


def rule_scores(features: pl.DataFrame) -> np.ndarray:
    """Weighted sum of simple red flags, plus a small amount tiebreak."""
    f = features.select(RULE_COLUMNS).fill_nan(None).fill_null(strategy="zero")
    active = (f["self_transfer"] == 0).to_numpy()
    flags = {
        # Rapid movement: most of what came in has already gone out, and funds arrived recently.
        "pass_through": (
            ((f["src_forward_ratio_24h"] > 0.7) & (f["src_forward_ratio_24h"] < 1.3)).to_numpy()
            & (f["src_mins_since_in"] < 720).to_numpy()
            & (f["src_mins_since_in"] > 0).to_numpy(),
            3.0,
        ),
        "fan_out": ((f["src_out_distinct_24h"] >= 5).to_numpy(), 2.0),
        "fan_in": ((f["dst_in_distinct_24h"] >= 5).to_numpy(), 2.0),
        "scatter_gather": ((f["two_paths_24h"] >= 2).to_numpy(), 3.0),
        "cycle": ((f["cycle_min_hops_96h"] > 0).to_numpy(), 4.0),
        "round_amounts": ((f["src_out_round_ratio_24h"] > 0.5).to_numpy(), 1.0),
        "large": ((f["log_usd"] > np.log1p(10_000)).to_numpy(), 1.0),
    }
    score = np.zeros(features.height)
    for hit, weight in flags.values():
        score += hit * weight
    return active * (score + f["log_usd"].to_numpy() / 100.0)
