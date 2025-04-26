"""Static USD rates implied by cross-currency rows, fitted on the training window only."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def fit_fx(frame: pl.DataFrame) -> dict[str, float]:
    """Estimate each currency's USD rate from ``amount_paid`` and ``amount_received`` pairs.

    Rates spread outward from USD by breadth-first search, preferring the pair with the most rows.
    Currencies that never connect to USD are left out, and engine features for them become NaN.
    """
    pairs = (
        frame.filter(
            (pl.col("payment_currency") != pl.col("receiving_currency"))
            & (pl.col("amount_paid") > 0)
            & (pl.col("amount_received") > 0)
        )
        .with_columns(
            (pl.col("amount_paid") / pl.col("amount_received")).alias("paid_per_received")
        )
        .group_by("payment_currency", "receiving_currency")
        .agg(pl.col("paid_per_received").median().alias("ratio"), pl.len().alias("n"))
        .sort("n", descending=True)
    )
    edges: dict[str, list[tuple[str, float]]] = {}
    for paid, received, ratio, _ in pairs.iter_rows():
        # paid * rate(paid) == received * rate(received)
        edges.setdefault(paid, []).append((received, ratio))
        edges.setdefault(received, []).append((paid, 1.0 / ratio))
    rates: dict[str, float] = {"US Dollar": 1.0}
    frontier = ["US Dollar"]
    while frontier:
        next_frontier: list[str] = []
        for currency in frontier:
            for other, paid_per_other in edges.get(currency, []):
                if other not in rates:
                    rates[other] = rates[currency] * paid_per_other
                    next_frontier.append(other)
        frontier = next_frontier
    return dict(sorted(rates.items()))


def save_fx(rates: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rates, indent=2) + "\n")


def load_fx(path: Path) -> dict[str, float]:
    rates: dict[str, float] = json.loads(path.read_text())
    return rates
