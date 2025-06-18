"""Case persistence and the agent worker. Separated from HTTP so that both can be tested directly."""

from __future__ import annotations

import json
import time
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from unravel.agent.graph import build_graph, check_decision, investigate
from unravel.agent.llm import AnthropicClient, ModelClient, SpendGuard
from unravel.agent.pg_store import PostgresStore
from unravel.agent.scripted import ScriptedClient
from unravel.agent.tools import Alert
from unravel.api.settings import Settings


def make_client(settings: Settings) -> ModelClient:
    if settings.llm == "anthropic":
        return AnthropicClient(guard=SpendGuard(settings.llm_budget_usd))
    if settings.llm == "scripted":
        return ScriptedClient()
    raise ValueError(f"unknown UNRAVEL_LLM {settings.llm!r}")


def connect(dsn: str) -> psycopg.Connection[dict[str, Any]]:
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


def checkpointer(dsn: str) -> PostgresSaver:
    connection = psycopg.connect(dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    saver = PostgresSaver(connection)
    saver.setup()
    return saver


def claim_next_alert(db: psycopg.Connection[dict[str, Any]]) -> dict[str, Any] | None:
    """Atomically take the highest-priority new alert. Safe with several workers."""
    return db.execute(
        """
        UPDATE alerts SET state = 'investigating'
        WHERE alert_id = (
            SELECT alert_id FROM alerts WHERE state = 'new'
            ORDER BY raw_score DESC LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        RETURNING *
        """
    ).fetchone()


def investigate_alert(
    row: dict[str, Any],
    db: psycopg.Connection[dict[str, Any]],
    graph: Any,
    model_name: str,
) -> dict[str, Any]:
    alert = Alert(
        alert_id=row["alert_id"],
        tx_id=row["tx_id"],
        as_of=row["minute"],
        probability=row["probability"],
        band=row["band"],
        model_version=row["model_version"],
        reasons=row["reasons"],
    )
    output = investigate(graph, alert)
    pause, values = output["pause"], output["values"]
    db.execute(
        """
        INSERT INTO cases (alert_id, status, draft, findings, incomplete_reason, model, usage)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (alert_id) DO UPDATE SET
            status = EXCLUDED.status, draft = EXCLUDED.draft, findings = EXCLUDED.findings,
            incomplete_reason = EXCLUDED.incomplete_reason, updated_at = now()
        """,
        (
            alert.alert_id,
            pause["status"],
            Jsonb(values.get("draft")),
            Jsonb(values.get("findings", [])),
            values.get("incomplete_reason"),
            model_name,
            Jsonb(values.get("usage", {})),
        ),
    )
    db.execute("UPDATE alerts SET state = 'drafted' WHERE alert_id = %s", (alert.alert_id,))
    return pause


def record_decision(
    db: psycopg.Connection[dict[str, Any]],
    graph: Any,
    alert_id: str,
    *,
    analyst: str,
    action: str,
    reason: str | None,
    edited_draft: dict[str, Any] | None = None,
) -> str:
    """Resume the paused graph with the analyst's decision and write the audit event."""
    case = db.execute("SELECT * FROM cases WHERE alert_id = %s", (alert_id,)).fetchone()
    if case is None:
        raise KeyError(alert_id)
    if case["status"] in ("approved", "rejected"):
        raise ValueError(f"case {alert_id} is already {case['status']}")
    decision = {"action": action, "reason": reason, "edited_draft": edited_draft}
    check_decision(decision, case["draft"])
    final = graph.invoke(
        Command(resume=decision),
        {"configurable": {"thread_id": alert_id}},
    )
    db.execute(
        "UPDATE cases SET status = %s, draft = %s, updated_at = now() WHERE alert_id = %s",
        (final["status"], Jsonb(final.get("draft")), alert_id),
    )
    db.execute(
        """
        INSERT INTO review_events (alert_id, analyst, action, reason, draft_before, draft_after)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (alert_id, analyst, action, reason, Jsonb(case["draft"]), Jsonb(final.get("draft"))),
    )
    status: str = final["status"]
    return status


def run_worker(settings: Settings, *, poll_seconds: float = 2.0, once: bool = False) -> int:
    """Draft cases from the alert queue until stopped, or until it's empty when ``once`` is set."""
    db = connect(settings.database_url)
    store = PostgresStore(settings.agent_database_url)
    client = make_client(settings)
    graph = build_graph(store, client, checkpointer=checkpointer(settings.database_url))
    done = 0
    while True:
        row = claim_next_alert(db)
        if row is None:
            if once:
                return done
            time.sleep(poll_seconds)
            continue
        pause = investigate_alert(row, db, graph, client.name)
        done += 1
        print(json.dumps({"alert": row["alert_id"], "status": pause["status"]}), flush=True)
