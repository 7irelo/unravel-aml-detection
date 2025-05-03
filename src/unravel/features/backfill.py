"""Offline feature generation. It feeds the shared engine with events in canonical order."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from unravel.config import DatasetPaths
from unravel.data.schema import TX_COLUMNS
from unravel.features.engine import FEATURE_NAMES, FeatureEngine, TxEvent
from unravel.features.fx import load_fx


def iter_events(transactions: pl.DataFrame) -> Iterator[TxEvent]:
    """Yield canonical events. ``tx.raw`` and backfill both use this order."""
    for row in transactions.select(TX_COLUMNS).iter_rows():
        yield TxEvent(*row)


def backfill(
    paths: DatasetPaths, *, chunk_size: int = 50_000, log_every: int = 250_000
) -> dict[str, float]:
    """Compute features for every transaction and write them as Parquet parts."""
    transactions = pl.read_parquet(paths.transactions).select(TX_COLUMNS)
    engine = FeatureEngine(load_fx(paths.fx))
    out = paths.features
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    started = time.perf_counter()
    ids: list[str] = []
    rows: list[list[float]] = []
    part = 0

    def flush() -> None:
        nonlocal part, ids, rows
        if not rows:
            return
        matrix = np.asarray(rows, dtype=np.float64)
        frame = pl.DataFrame({"tx_id": ids}).with_columns(
            [pl.Series(name, matrix[:, i]) for i, name in enumerate(FEATURE_NAMES)]
        )
        frame.write_parquet(out / f"part-{part:05d}.parquet", compression="zstd")
        part += 1
        ids, rows = [], []

    for count, event in enumerate(iter_events(transactions), start=1):
        rows.append(engine.process(event))
        ids.append(event.tx_id)
        if len(rows) >= chunk_size:
            flush()
        if count % log_every == 0:
            rate = count / (time.perf_counter() - started)
            print(f"{count:,} events, {rate:,.0f}/s", flush=True)
    flush()

    elapsed = time.perf_counter() - started
    stats = {
        "events": float(engine.events_seen),
        "seconds": elapsed,
        "events_per_second": engine.events_seen / elapsed if elapsed else 0.0,
        "truncated_share": engine.truncated_events / max(engine.events_seen, 1),
        "accounts": float(len(engine.accounts)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def load_features(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Load Parquet feature parts, in canonical order, optionally restricted to some columns."""
    selected = None if columns is None else ["tx_id", *columns]
    return pl.scan_parquet(path / "part-*.parquet").select(selected or pl.all()).collect()
