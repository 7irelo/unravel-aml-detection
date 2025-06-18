"""Postgres-backed ``TxStore``. It connects as ``unravel_agent``, which cannot read labels."""

from __future__ import annotations

from collections.abc import Iterable

import psycopg

from unravel.agent.store import Tx

_COLUMNS = (
    "tx_id, minute, src, dst, amount_paid, payment_currency, "
    "amount_received, receiving_currency, payment_format, usd"
)


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.connection = psycopg.connect(dsn, autocommit=True)

    def get(self, tx_id: str) -> Tx | None:
        row = self.connection.execute(
            f"SELECT {_COLUMNS} FROM transactions WHERE tx_id = %s",
            (tx_id,),
        ).fetchone()
        return Tx(*row) if row else None

    def get_many(self, tx_ids: Iterable[str]) -> list[Tx]:
        rows = self.connection.execute(
            f"SELECT {_COLUMNS} FROM transactions WHERE tx_id = ANY(%s)",
            (list(tx_ids),),
        ).fetchall()
        return [Tx(*row) for row in rows]

    def account_transactions(self, account: str, start: int, end: int) -> list[Tx]:
        rows = self.connection.execute(
            f"""
            SELECT {_COLUMNS} FROM transactions
            WHERE src = %(a)s AND minute BETWEEN %(s)s AND %(e)s
            UNION
            SELECT {_COLUMNS} FROM transactions
            WHERE dst = %(a)s AND minute BETWEEN %(s)s AND %(e)s
            ORDER BY minute, tx_id
            """,
            {"a": account, "s": start, "e": end},
        ).fetchall()
        return [Tx(*row) for row in rows]
