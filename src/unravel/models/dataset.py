"""Assemble aligned arrays for modelling. Labels join here and nowhere earlier in the pipeline.

Features stay on disk until needed. ``FeatureReader`` reads only the requested columns and rows, part
by part, so training on a small sample of negatives never loads the whole matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from unravel.config import DatasetPaths
from unravel.eval.metrics import AlertFrame
from unravel.eval.splits import MINUTES_PER_DAY, TEST, VAL_A, VAL_B, Split


class FeatureReader:
    """Random access to Parquet feature parts, in canonical transaction order."""

    def __init__(self, directory: Path) -> None:
        self.parts = sorted(directory.glob("part-*.parquet"))
        if not self.parts:
            raise FileNotFoundError(f"no feature parts in {directory}; run the backfill")
        self.sizes = [pl.scan_parquet(p).select(pl.len()).collect().item() for p in self.parts]
        self.height = int(sum(self.sizes))

    def tx_ids(self) -> pl.Series:
        return pl.concat(
            [pl.read_parquet(p, columns=["tx_id"])["tx_id"] for p in self.parts], rechunk=True
        )

    def matrix(self, columns: list[str], mask: np.ndarray | None = None) -> np.ndarray:
        """Return float32 rows where ``mask`` is true, in order. Missing values stay NaN."""
        chunks: list[np.ndarray] = []
        start = 0
        for part, size in zip(self.parts, self.sizes, strict=True):
            piece = None if mask is None else mask[start : start + size]
            start += size
            if piece is not None and not piece.any():
                continue
            frame = pl.read_parquet(part, columns=columns)
            if piece is not None:
                frame = frame.filter(pl.Series(piece))
            chunks.append(frame.select(columns).to_numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, len(columns)), dtype=np.float32)
        return np.concatenate(chunks)

    def frame(self, columns: list[str]) -> pl.DataFrame:
        """Load whole columns. Use for a handful of columns only."""
        return pl.concat([pl.read_parquet(p, columns=columns) for p in self.parts], rechunk=True)


@dataclass(frozen=True)
class ModelData:
    """Everything aligned to canonical transaction order."""

    tx_id: np.ndarray
    features: FeatureReader
    y: np.ndarray
    partition: np.ndarray
    minute: np.ndarray
    day: np.ndarray
    account: np.ndarray
    usd: np.ndarray
    pattern: list[str | None]
    pattern_instance: np.ndarray
    split: Split

    def mask(self, *codes: int) -> np.ndarray:
        return np.isin(self.partition, codes)

    def alert_frame(self, scores: np.ndarray, mask: np.ndarray) -> AlertFrame:
        return AlertFrame(
            y=self.y[mask],
            scores=scores[mask],
            day=self.day[mask],
            account=self.account[mask],
            usd=self.usd[mask],
            pattern_instance=self.pattern_instance[mask],
        )


def load_model_data(paths: DatasetPaths) -> ModelData:
    split = Split.load(paths.split)
    tx = pl.read_parquet(paths.transactions, columns=["tx_id", "minute", "src"])
    labels = pl.read_parquet(paths.labels)
    features = FeatureReader(paths.features)
    if not features.tx_ids().equals(tx["tx_id"]):
        raise ValueError("features and transactions are not aligned; rerun the backfill")
    joined = tx.join(labels, on="tx_id", how="left", maintain_order="left")
    minute = joined["minute"].to_numpy()
    usd_source = features.frame(["log_usd"])["log_usd"].to_numpy()
    return ModelData(
        tx_id=joined["tx_id"].to_numpy(),
        features=features,
        y=joined["is_laundering"].to_numpy().astype(np.int8),
        partition=split.assign(minute),
        minute=minute,
        day=(minute - split.origin_minute) // MINUTES_PER_DAY,
        account=joined["src"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64),
        usd=np.expm1(usd_source),
        pattern=joined["pattern"].to_list(),
        pattern_instance=joined["pattern_instance"].fill_null(-1).to_numpy().astype(np.int64),
        split=split,
    )


__all__ = ["TEST", "VAL_A", "VAL_B", "FeatureReader", "ModelData", "load_model_data"]
