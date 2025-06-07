"""Read-only access to transactions for the investigator.

The store has no way to reach labels or patterns. The in-memory store is built from the transaction
table only; the Postgres store connects with a role that has no grants on the ``ground_truth``
schema. Every query takes ``as_of`` and cannot return anything later, so an agent investigating an
alert sees only what was knowable when the alert fired.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class Tx:
    tx_id: str
    minute: int
    src: str
    dst: str
    amount_paid: float
    payment_currency: str
    amount_received: float
    receiving_currency: str
    payment_format: str
    usd: float

    def as_dict(self) -> dict[str, object]:
        return {
            "tx_id": self.tx_id,
            "minute": self.minute,
            "src": self.src,
            "dst": self.dst,
            "amount_paid": self.amount_paid,
            "payment_currency": self.payment_currency,
            "amount_received": self.amount_received,
            "receiving_currency": self.receiving_currency,
            "payment_format": self.payment_format,
            "usd": round(self.usd, 2),
        }


class TxStore(Protocol):
    def get(self, tx_id: str) -> Tx | None: ...

    def account_transactions(self, account: str, start: int, end: int) -> list[Tx]: ...

    def get_many(self, tx_ids: Iterable[str]) -> list[Tx]: ...


class MemoryStore:
    """Indexed in-process store, used by the offline evaluation, tests and the demo."""

    def __init__(self, transactions: Sequence[Tx]) -> None:
        self.by_id = {t.tx_id: t for t in transactions}
        index: dict[str, list[Tx]] = defaultdict(list)
        for t in sorted(transactions, key=lambda x: x.minute):
            index[t.src].append(t)
            if t.dst != t.src:
                index[t.dst].append(t)
        self.index = dict(index)
        self.minutes = {a: [t.minute for t in rows] for a, rows in self.index.items()}

    @classmethod
    def from_parquet(cls, path: Path, fx: dict[str, float]) -> MemoryStore:
        frame = pl.read_parquet(path).select(
            "tx_id",
            "minute",
            "src",
            "dst",
            "amount_paid",
            "payment_currency",
            "amount_received",
            "receiving_currency",
            "payment_format",
        )
        rows = [Tx(*row, row[4] * fx.get(row[5], 0.0)) for row in frame.iter_rows()]  # type: ignore[call-arg]
        return cls(rows)

    def get(self, tx_id: str) -> Tx | None:
        return self.by_id.get(tx_id)

    def get_many(self, tx_ids: Iterable[str]) -> list[Tx]:
        return [self.by_id[i] for i in tx_ids if i in self.by_id]

    def account_transactions(self, account: str, start: int, end: int) -> list[Tx]:
        """Transactions touching ``account`` with ``start <= minute <= end``, oldest first."""
        rows = self.index.get(account)
        if not rows:
            return []
        minutes = self.minutes[account]
        lo = bisect_right(minutes, start - 1)
        hi = bisect_right(minutes, end)
        return rows[lo:hi]


def neighbourhood(
    store: TxStore,
    seeds: Iterable[str],
    start: int,
    end: int,
    *,
    hops: int = 2,
    max_nodes: int = 300,
    max_edges: int = 3000,
    max_degree: int = 200,
) -> tuple[list[Tx], bool]:
    """Bounded breadth-first expansion. Returns edges plus a flag if any limit was reached.

    Busy hub accounts would otherwise swallow the budget, so a node with more than ``max_degree``
    edges is included but not expanded through.
    """
    visited: set[str] = set()
    frontier = list(dict.fromkeys(seeds))
    edges: dict[str, Tx] = {}
    truncated = False
    for _ in range(hops):
        next_frontier: list[str] = []
        for account in frontier:
            if account in visited:
                continue
            if len(visited) >= max_nodes:
                truncated = True
                break
            visited.add(account)
            rows = store.account_transactions(account, start, end)
            if len(rows) > max_degree:
                truncated = True
                rows = rows[-max_degree:]
                expand = False
            else:
                expand = True
            for tx in rows:
                if len(edges) >= max_edges:
                    truncated = True
                    break
                edges[tx.tx_id] = tx
                if expand:
                    next_frontier.extend(a for a in (tx.src, tx.dst) if a not in visited)
        frontier = list(dict.fromkeys(next_frontier))
        if not frontier:
            break
    return sorted(edges.values(), key=lambda t: (t.minute, t.tx_id)), truncated
