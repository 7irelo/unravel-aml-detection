"""Analyst review API. Reading a case and recording a decision: there is no filing endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from unravel.agent.graph import build_graph
from unravel.agent.pg_store import PostgresStore
from unravel.api.cases import checkpointer, connect, make_client, record_decision
from unravel.api.settings import Settings


class Decision(BaseModel):
    analyst: str
    action: Literal["approve", "reject"]
    reason: str | None = None
    edited_draft: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _rejection_needs_reason(self) -> Decision:
        if self.action == "reject" and not (self.reason or "").strip():
            raise ValueError("a rejection needs a reason")
        return self


settings = Settings()
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    state["db"] = connect(settings.database_url)
    state["store"] = PostgresStore(settings.agent_database_url)
    state["graph"] = build_graph(
        state["store"],
        make_client(settings),
        checkpointer=checkpointer(settings.database_url),
    )
    yield
    state["db"].close()


app = FastAPI(title="Unravel review API", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    state["db"].execute("SELECT 1")
    return {"status": "ok"}


@app.get("/cases")
def list_cases(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """The analyst's queue, highest score first."""
    return (
        state["db"]
        .execute(
            """
        SELECT c.alert_id, c.status, a.band, a.probability, a.raw_score, a.tx_id, a.minute,
               c.draft ->> 'recommendation' AS recommendation,
               c.draft ->> 'pattern' AS pattern, c.incomplete_reason, c.model
        FROM cases c JOIN alerts a USING (alert_id)
        WHERE (%(s)s::text IS NULL OR c.status = %(s)s)
        ORDER BY a.raw_score DESC LIMIT %(l)s
        """,
            {"s": status, "l": min(limit, 500)},
        )
        .fetchall()
    )


@app.get("/cases/{alert_id}")
def get_case(alert_id: str) -> dict[str, Any]:
    case = (
        state["db"]
        .execute(
            """
        SELECT c.*, a.tx_id, a.minute, a.probability, a.band, a.reasons, a.model_version
        FROM cases c JOIN alerts a USING (alert_id) WHERE c.alert_id = %s
        """,
            (alert_id,),
        )
        .fetchone()
    )
    if case is None:
        raise HTTPException(404, "case not found")
    draft = case.get("draft") or {}
    cited = {
        tx_id
        for claim in [*draft.get("claims", []), *draft.get("contrary_evidence", [])]
        for tx_id in claim.get("tx_ids", [])
    }
    case["cited_transactions"] = [t.as_dict() for t in state["store"].get_many(sorted(cited))]
    case["review_events"] = (
        state["db"]
        .execute(
            "SELECT analyst, action, reason, created_at FROM review_events "
            "WHERE alert_id = %s ORDER BY id",
            (alert_id,),
        )
        .fetchall()
    )
    return case


@app.post("/cases/{alert_id}/decision")
def decide(alert_id: str, decision: Decision) -> dict[str, str]:
    try:
        status = record_decision(
            state["db"],
            state["graph"],
            alert_id,
            analyst=decision.analyst,
            action=decision.action,
            reason=decision.reason,
            edited_draft=decision.edited_draft,
        )
    except KeyError as error:
        raise HTTPException(404, "case not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
    return {"alert_id": alert_id, "status": status}
