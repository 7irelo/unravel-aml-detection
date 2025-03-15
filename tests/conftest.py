from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from unravel.config import DatasetPaths
from unravel.data.fixtures import generate_fixture
from unravel.data.ingest import ingest
from unravel.eval.splits import make_split
from unravel.features.engine import TxEvent
from unravel.features.fx import fit_fx, save_fx

FX = {"US Dollar": 1.0, "Euro": 1.1, "Yuan": 0.15, "Rupee": 0.012}


def make_event(
    minute: int,
    src: str,
    dst: str,
    amount: float = 100.0,
    *,
    tx_id: str | None = None,
    fmt: str = "ACH",
    currency: str = "US Dollar",
    bank: str = "001",
) -> TxEvent:
    return TxEvent(
        tx_id=tx_id or f"T{minute}-{src}-{dst}-{amount}",
        order_key=minute,
        minute=minute,
        src=src,
        dst=dst,
        from_bank=bank,
        to_bank=bank,
        amount_received=amount,
        receiving_currency=currency,
        amount_paid=amount,
        payment_currency=currency,
        payment_format=fmt,
    )


@pytest.fixture(scope="session")
def fixture_paths(tmp_path_factory: pytest.TempPathFactory) -> DatasetPaths:
    """A small end-to-end dataset, ingested the same way as the real one."""
    root: Path = tmp_path_factory.mktemp("unravel")
    paths = DatasetPaths("fixture", root=root)
    generate_fixture(paths, background_per_day=300, accounts=200, instances_per_pattern=4)
    ingest(paths)
    tx = pl.read_parquet(paths.transactions)
    split = make_split(tx["minute"].to_numpy())
    split.save(paths.split)
    save_fx(fit_fx(tx.filter(pl.col("minute") < split.val_start)), paths.fx)
    return paths
