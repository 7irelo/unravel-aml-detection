"""Canonical transaction schema and the deterministic transaction ID.

The raw data has no transaction ID and may contain duplicate rows, so an ID is derived from the row
index plus content. ``order_key`` comes from the same hash and breaks ties inside a minute:
timestamps have minute resolution, and file order inside a minute could carry generator structure.
Neither column may ever be a model feature.
"""

from __future__ import annotations

import hashlib

RAW_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

#: Columns that may be published on ``tx.raw``. Labels are stored separately and never join this set.
TX_COLUMNS = [
    "tx_id",
    "order_key",
    "minute",
    "src",
    "dst",
    "from_bank",
    "to_bank",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
]

LABEL_COLUMNS = ["tx_id", "is_laundering", "pattern", "pattern_instance"]

PATTERNS = [
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "BIPARTITE",
    "STACK",
    "RANDOM",
]


def make_tx_id(row_index: int, raw_key: str) -> tuple[str, int]:
    """Return ``(tx_id, order_key)`` for a raw row. Stable across platforms and library versions."""
    digest = hashlib.blake2b(f"{row_index}|{raw_key}".encode(), digest_size=8).hexdigest()
    return f"T{digest}", int(digest[:15], 16)
