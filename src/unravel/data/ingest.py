"""Convert raw AMLworld CSV and pattern files to canonical Parquet, with labels held separately."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from unravel.config import DatasetPaths
from unravel.data.schema import LABEL_COLUMNS, RAW_COLUMNS, TX_COLUMNS, make_tx_id

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\- ]+?)(?::|\s*$)")
_END = "END LAUNDERING ATTEMPT"
_KEY_COLUMNS = RAW_COLUMNS[:-1]


@dataclass(frozen=True)
class PatternRow:
    key: str
    pattern: str
    instance: int


def parse_patterns(path: Path) -> list[PatternRow]:
    """Parse a ``*_Patterns.txt`` file. Each row keeps its pattern type and instance number."""
    rows: list[PatternRow] = []
    pattern: str | None = None
    instance = -1
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        begin = _BEGIN.match(line)
        if begin:
            pattern = begin.group(1).strip()
            instance += 1
        elif line.startswith(_END):
            pattern = None
        elif pattern is not None:
            # The trailing field is the label; the key is everything before it.
            rows.append(PatternRow(line.rsplit(",", 1)[0], pattern, instance))
    return rows


def _read_raw(path: Path) -> pl.DataFrame:
    # The header repeats "Account", and bank codes have leading zeros, so read everything as text.
    return pl.read_csv(path, has_header=True, new_columns=RAW_COLUMNS, infer_schema_length=0)


def ingest(paths: DatasetPaths) -> dict[str, int]:
    """Write ``transactions.parquet`` (sorted in canonical order) and ``labels.parquet``."""
    raw = _read_raw(paths.raw_trans)
    keys = raw.select(pl.concat_str(_KEY_COLUMNS, separator=",").alias("raw_key"))["raw_key"]
    tx_ids: list[pl.Series] = []
    order_keys: list[pl.Series] = []
    for offset in range(0, len(keys), 500_000):
        batch = [
            make_tx_id(offset + i, key)
            for i, key in enumerate(keys.slice(offset, 500_000).to_list())
        ]
        tx_ids.append(pl.Series([a for a, _ in batch], dtype=pl.String))
        order_keys.append(pl.Series([b for _, b in batch], dtype=pl.Int64))
    frame = raw.with_columns(
        raw_key=keys,
        tx_id=pl.concat(tx_ids, rechunk=True),
        order_key=pl.concat(order_keys, rechunk=True),
        minute=pl.col("timestamp")
        .str.strptime(pl.Datetime("ms"), "%Y/%m/%d %H:%M")
        .dt.epoch("s")
        .floordiv(60)
        .cast(pl.Int64),
        src=pl.concat_str(["from_bank", "from_account"], separator=":"),
        dst=pl.concat_str(["to_bank", "to_account"], separator=":"),
        amount_received=pl.col("amount_received").cast(pl.Float64),
        amount_paid=pl.col("amount_paid").cast(pl.Float64),
        is_laundering=pl.col("is_laundering").cast(pl.Int8),
    ).sort(["minute", "order_key"])

    pattern_rows = parse_patterns(paths.raw_patterns) if paths.raw_patterns.exists() else []
    patterns = pl.DataFrame(
        {
            "raw_key": [r.key for r in pattern_rows],
            "pattern": [r.pattern for r in pattern_rows],
            "pattern_instance": [r.instance for r in pattern_rows],
        },
        schema={"raw_key": pl.String, "pattern": pl.String, "pattern_instance": pl.Int32},
    ).unique(subset="raw_key", keep="first", maintain_order=True)
    is_positive = pl.col("is_laundering") == 1
    labels = (
        frame.select("tx_id", "is_laundering", "raw_key")
        .join(patterns, on="raw_key", how="left", maintain_order="left")
        .with_columns(
            pl.when(is_positive).then(pl.col("pattern")).alias("pattern"),
            pl.when(is_positive).then(pl.col("pattern_instance")).alias("pattern_instance"),
        )
        .select(LABEL_COLUMNS)
    )

    paths.processed.mkdir(parents=True, exist_ok=True)
    frame.select(TX_COLUMNS).write_parquet(paths.transactions)
    labels.write_parquet(paths.labels)
    return {
        "rows": frame.height,
        "positives": int(frame["is_laundering"].cast(pl.Int64).sum()),
        "pattern_rows": len(pattern_rows),
        "pattern_rows_matched": int(labels["pattern"].is_not_null().sum()),
    }


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def write_manifest(paths: DatasetPaths, manifest: Path) -> dict[str, dict[str, object]]:
    """Record the raw file hashes, so results can be tied to their exact inputs."""
    existing: dict[str, dict[str, object]] = (
        json.loads(manifest.read_text()) if manifest.exists() else {}
    )
    for path in (paths.raw_trans, paths.raw_patterns):
        if path.exists():
            existing[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return existing
