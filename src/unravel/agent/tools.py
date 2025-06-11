"""The investigator's tools: read-only, typed, bounded, and unable to look past the alert time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from unravel.agent.detectors import detect
from unravel.agent.store import Tx, TxStore, neighbourhood

MAX_PAGE = 100
MINUTES_PER_DAY = 1440
DEFAULT_LOOKBACK_DAYS = 7


@dataclass(frozen=True)
class Alert:
    """What the detector hands over. It has no ground truth in it."""

    alert_id: str
    tx_id: str
    as_of: int  # the alert's minute: tools can't see past it
    probability: float
    band: str
    model_version: str
    reasons: list[dict[str, Any]] = field(default_factory=list)


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_alert",
        "description": "Get the alert: score, risk band, model version and SHAP reason codes.",
        "strict": True,
        "input_schema": _object({}, []),
    },
    {
        "name": "get_account_profile",
        "description": "Summarise one account's activity up to the alert time: volumes, peers, "
        "formats, currencies and time span in the data.",
        "strict": True,
        "input_schema": _object({"account": {"type": "string"}}, ["account"]),
    },
    {
        "name": "get_transactions",
        "description": "List one account's transactions, oldest first, within a lookback window. "
        "Returns at most 100 rows, with a cursor when more exist.",
        "strict": True,
        "input_schema": _object(
            {
                "account": {"type": "string"},
                "lookback_days": {"type": "integer"},
                "cursor": {"type": "integer"},
            },
            ["account", "lookback_days", "cursor"],
        ),
    },
    {
        "name": "get_neighbourhood",
        "description": "Summarise the subgraph around an account: up to 3 hops, with edge IDs "
        "and a truncation flag if limits were reached.",
        "strict": True,
        "input_schema": _object(
            {
                "account": {"type": "string"},
                "hops": {"type": "integer"},
                "lookback_days": {"type": "integer"},
            },
            ["account", "hops", "lookback_days"],
        ),
    },
    {
        "name": "trace_flows",
        "description": "Follow money from a transaction, downstream or upstream, taking the "
        "time-respecting hops that carry the most value.",
        "strict": True,
        "input_schema": _object(
            {
                "tx_id": {"type": "string"},
                "direction": {"type": "string", "enum": ["downstream", "upstream"]},
                "hops": {"type": "integer"},
            },
            ["tx_id", "direction", "hops"],
        ),
    },
    {
        "name": "detect_patterns",
        "description": "Run deterministic detectors for the 8 laundering typologies on the "
        "alert's neighbourhood. Returns candidates with member transaction IDs and a strength. "
        "Candidates are leads, not conclusions.",
        "strict": True,
        "input_schema": _object({"lookback_days": {"type": "integer"}}, ["lookback_days"]),
    },
    {
        "name": "aggregate",
        "description": "Compute exact totals over transaction IDs: USD total, count, first and "
        "last minute, and distinct counterparties. Use this for every number you state.",
        "strict": True,
        "input_schema": _object(
            {"tx_ids": {"type": "array", "items": {"type": "string"}}}, ["tx_ids"]
        ),
    },
]


class Toolbox:
    """Tools for one alert. It records every transaction ID shown, for verifier check L2."""

    def __init__(self, store: TxStore, alert: Alert, retrieved: set[str] | None = None) -> None:
        self.store = store
        self.alert = alert
        self.retrieved: set[str] = set() if retrieved is None else retrieved

    def _visible(self, rows: list[Tx]) -> list[Tx]:
        return [t for t in rows if t.minute <= self.alert.as_of]

    def _mark(self, rows: list[Tx]) -> list[dict[str, object]]:
        self.retrieved.update(t.tx_id for t in rows)
        return [t.as_dict() for t in rows]

    def _window(self, lookback_days: int) -> tuple[int, int]:
        days = max(1, min(lookback_days, 30))
        return self.alert.as_of - days * MINUTES_PER_DAY, self.alert.as_of

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return JSON. Errors come back as data, so the agent can recover."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(handler(**arguments))
        except (KeyError, TypeError, ValueError) as error:
            return json.dumps({"error": f"{type(error).__name__}: {error}"})

    def _tool_get_alert(self) -> dict[str, Any]:
        seed = self.store.get(self.alert.tx_id)
        return {
            "alert_id": self.alert.alert_id,
            "as_of_minute": self.alert.as_of,
            "probability": round(self.alert.probability, 5),
            "band": self.alert.band,
            "model_version": self.alert.model_version,
            "reason_codes": self.alert.reasons,
            "transaction": self._mark([seed])[0] if seed else None,
        }

    def _tool_get_account_profile(self, account: str) -> dict[str, Any]:
        rows = self.store.account_transactions(account, 0, self.alert.as_of)
        if not rows:
            return {"account": account, "transactions": 0}
        sent = [t for t in rows if t.src == account and t.dst != account]
        received = [t for t in rows if t.dst == account and t.src != account]
        return {
            "account": account,
            "transactions": len(rows),
            "first_minute": rows[0].minute,
            "last_minute": rows[-1].minute,
            "sent": {"count": len(sent), "usd": round(sum(t.usd for t in sent), 2)},
            "received": {"count": len(received), "usd": round(sum(t.usd for t in received), 2)},
            "self_transfers": len(rows) - len(sent) - len(received),
            "distinct_payees": len({t.dst for t in sent}),
            "distinct_payers": len({t.src for t in received}),
            "formats": sorted({t.payment_format for t in rows}),
            "currencies": sorted({t.payment_currency for t in rows}),
        }

    def _tool_get_transactions(
        self, account: str, lookback_days: int, cursor: int = 0
    ) -> dict[str, Any]:
        start, end = self._window(lookback_days)
        rows = self.store.account_transactions(account, start, end)
        page = rows[cursor : cursor + MAX_PAGE]
        following = cursor + MAX_PAGE
        return {
            "total": len(rows),
            "transactions": self._mark(page),
            "next_cursor": following if following < len(rows) else None,
        }

    def _tool_get_neighbourhood(
        self, account: str, hops: int, lookback_days: int
    ) -> dict[str, Any]:
        start, end = self._window(lookback_days)
        edges, truncated = neighbourhood(
            self.store, [account], start, end, hops=max(1, min(hops, 3))
        )
        shown = edges[-MAX_PAGE:] if len(edges) > MAX_PAGE else edges
        return {
            "accounts": len({t.src for t in edges} | {t.dst for t in edges}),
            "edges_total": len(edges),
            "edges_shown": self._mark(shown),
            "truncated": truncated or len(edges) > MAX_PAGE,
        }

    def _tool_trace_flows(self, tx_id: str, direction: str, hops: int) -> dict[str, Any]:
        current = self.store.get(tx_id)
        if current is None or current.minute > self.alert.as_of:
            return {"error": f"{tx_id} is not visible"}
        path = [current]
        for _ in range(max(1, min(hops, 8))):
            if direction == "downstream":
                rows = self.store.account_transactions(
                    current.dst, current.minute, self.alert.as_of
                )
                options = [t for t in rows if t.src == current.dst and t.dst != current.dst]
            else:
                rows = self.store.account_transactions(
                    current.src, current.minute - 7 * MINUTES_PER_DAY, current.minute
                )
                options = [t for t in rows if t.dst == current.src and t.src != current.src]
            options = [t for t in options if t.tx_id not in {p.tx_id for p in path}]
            if not options:
                break
            current = max(options, key=lambda t: t.usd)
            path.append(current)
        return {"direction": direction, "path": self._mark(path)}

    def _tool_detect_patterns(self, lookback_days: int) -> dict[str, Any]:
        seed = self.store.get(self.alert.tx_id)
        if seed is None:
            return {"error": "alert transaction not found"}
        start, end = self._window(lookback_days)
        edges, truncated = neighbourhood(self.store, [seed.src, seed.dst], start, end, hops=2)
        candidates = detect(edges, seed)[:5]
        cited = {i for c in candidates for i in c.tx_ids}
        self.retrieved.update(cited)
        self.retrieved.add(seed.tx_id)
        return {
            "candidates": [c.as_dict() for c in candidates],
            "neighbourhood_edges": len(edges),
            "truncated": truncated,
        }

    def _tool_aggregate(self, tx_ids: list[str]) -> dict[str, Any]:
        rows = self._visible(self.store.get_many(tx_ids))
        self.retrieved.update(t.tx_id for t in rows)
        known = {t.tx_id for t in rows}
        counts: dict[str, int] = {}
        for t in rows:
            counts[t.src] = counts.get(t.src, 0) + 1
            counts[t.dst] = counts.get(t.dst, 0) + 1
        accounts = set(counts)
        centre = max(counts, key=lambda a: counts[a], default=None)
        return {
            "total_usd": round(sum(t.usd for t in rows), 2),
            "count": len(rows),
            "first_minute": min((t.minute for t in rows), default=None),
            "last_minute": max((t.minute for t in rows), default=None),
            "accounts": len(accounts),
            "distinct_counterparties": len(accounts - {centre}) if centre else 0,
            "unknown_or_later_ids": [i for i in tx_ids if i not in known],
        }
