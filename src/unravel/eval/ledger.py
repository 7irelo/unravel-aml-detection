"""Test-set discipline: every look at test data leaves a permanent record."""

from __future__ import annotations

import json
from pathlib import Path


def record_test_evaluation(
    ledger: Path, *, dataset: str, arm: str, protocol: str, model_hash: str, reason: str
) -> None:
    """Append one entry. The reason is required, and the write-up reports the total count."""
    if not reason.strip():
        raise ValueError("a test evaluation needs a stated reason")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "dataset": dataset,
        "arm": arm,
        "protocol": protocol,
        "model_hash": model_hash,
        "reason": reason,
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger(ledger: Path) -> list[dict[str, str]]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
