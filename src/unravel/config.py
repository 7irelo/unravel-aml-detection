"""Filesystem layout and dataset naming. Everything derived lives under ``data/`` or ``artifacts/``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetPaths:
    """Paths for one dataset, e.g. ``HI-Small`` or a generated ``fixture``."""

    name: str
    root: Path = ROOT

    @property
    def raw_trans(self) -> Path:
        return self.root / "data" / "raw" / f"{self.name}_Trans.csv"

    @property
    def raw_patterns(self) -> Path:
        return self.root / "data" / "raw" / f"{self.name}_Patterns.txt"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed" / self.name

    @property
    def transactions(self) -> Path:
        return self.processed / "transactions.parquet"

    @property
    def labels(self) -> Path:
        return self.processed / "labels.parquet"

    @property
    def fx(self) -> Path:
        return self.processed / "fx.json"

    @property
    def split(self) -> Path:
        return self.processed / "split.json"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features" / self.name

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts" / self.name

    @property
    def reports(self) -> Path:
        return self.root / "reports" / self.name
