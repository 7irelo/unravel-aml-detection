"""The investigator as a LangGraph state machine.

    load_alert -> gather <-> tools -> run_detectors -> draft -> verify -> human_review -> record
                                                         ^         |
                                                         +-revise--+ (at most ``MAX_REVISIONS``)

Failure never looks like success. A run that exhausts its budget, is refused, or can't pass
verification reaches the analyst marked ``incomplete``, with the reason attached.

No node can file anything. The only way out of the graph is ``record``, which stores the analyst's
decision. "Approved" means an analyst accepted the draft; what happens next is outside this system.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from unravel.agent.case import CaseFile
from unravel.agent.llm import (
    BudgetExceededError,
    ModelClient,
    ModelRefusedError,
    Usage,
    strict_json_schema,
)
from unravel.agent.prompts import DRAFT_INSTRUCTION, REVISE_INSTRUCTION, SYSTEM
from unravel.agent.store import TxStore
from unravel.agent.tools import TOOL_SCHEMAS, Alert, Toolbox
from unravel.agent.verifier import verify

MAX_TOOL_ROUNDS = 12
MAX_REVISIONS = 2
LOOKBACK_DAYS = 7
CASE_SCHEMA = strict_json_schema(CaseFile.model_json_schema())


class InvestigationState(TypedDict, total=False):
    alert: dict[str, Any]
    messages: list[dict[str, Any]]
    retrieved: list[str]
    tool_rounds: int
    revisions: int
    draft: dict[str, Any] | None
    findings: list[dict[str, str]]
    status: str  # pending_review | incomplete | approved | rejected
    incomplete_reason: str | None
    usage: dict[str, int]
    decision: dict[str, Any] | None


def _add_usage(state: InvestigationState, usage: Usage) -> dict[str, int]:
    total = Usage(**state.get("usage", {}))
    total.add(usage)
    return total.__dict__.copy()


def check_decision(decision: dict[str, Any], draft: dict[str, Any] | None) -> None:
    """Validate an analyst decision. Call this before resuming the graph.

    A decision the graph rejects after resuming stays stored in its checkpoint, so the case could
    never be decided. The graph also checks, as a backstop.
    """
    action = decision.get("action")
    if action not in ("approve", "reject"):
        raise ValueError(f"invalid analyst action {action!r}")
    if action == "reject" and not str(decision.get("reason") or "").strip():
        raise ValueError("a rejection needs a reason")
    if action == "approve" and not (draft or decision.get("edited_draft")):
        raise ValueError("there is no draft to approve")
    if decision.get("edited_draft") is not None:
        CaseFile.model_validate(decision["edited_draft"])


def build_graph(
    store: TxStore,
    client: ModelClient,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    effort: str = "high",
) -> Any:
    """Compile the graph. Every node's state is JSON-safe, so checkpoints survive restarts."""

    def _toolbox(state: InvestigationState) -> Toolbox:
        return Toolbox(store, Alert(**state["alert"]), set(state.get("retrieved", [])))

    def load_alert(state: InvestigationState) -> dict[str, Any]:
        alert = state["alert"]
        opening = (
            f"Investigate alert {alert['alert_id']} on transaction {alert['tx_id']}. "
            "Gather the evidence you need."
        )
        return {
            "messages": [{"role": "user", "content": opening}],
            "retrieved": [],
            "tool_rounds": 0,
            "revisions": 0,
            "draft": None,
            "findings": [],
            "status": "running",
            "incomplete_reason": None,
            "usage": {},
        }

    def gather(state: InvestigationState) -> dict[str, Any]:
        try:
            reply = client.respond(
                system=SYSTEM, messages=state["messages"], tools=TOOL_SCHEMAS, effort=effort
            )
        except (BudgetExceededError, ModelRefusedError) as error:
            return {"status": "incomplete", "incomplete_reason": f"{type(error).__name__}: {error}"}
        return {
            "messages": [*state["messages"], {"role": "assistant", "content": reply.content}],
            "usage": _add_usage(state, reply.usage),
            "tool_rounds": state["tool_rounds"] + 1,
        }

    def tools(state: InvestigationState) -> dict[str, Any]:
        toolbox = _toolbox(state)
        results = []
        for block in state["messages"][-1]["content"]:
            if block.get("type") == "tool_use":
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": toolbox.call(block["name"], block["input"]),
                    }
                )
        return {
            # All results go back in one message, which keeps parallel tool use working.
            "messages": [*state["messages"], {"role": "user", "content": results}],
            "retrieved": sorted(toolbox.retrieved),
        }

    def after_gather(state: InvestigationState) -> str:
        if state.get("status") == "incomplete":
            return "human_review"
        wants_tools = any(
            b.get("type") == "tool_use" for b in state["messages"][-1].get("content", [])
        )
        if wants_tools:
            return "tools"
        return "run_detectors"

    def after_tools(state: InvestigationState) -> str:
        return "gather" if state["tool_rounds"] < MAX_TOOL_ROUNDS else "run_detectors"

    def run_detectors(state: InvestigationState) -> dict[str, Any]:
        """Detection always runs, whatever the model chose to do, so its facts are on the record."""
        toolbox = _toolbox(state)
        result = toolbox.call("detect_patterns", {"lookback_days": LOOKBACK_DAYS})
        note = (
            "Deterministic detector results (leads to check, not conclusions):\n"
            f"{result}\n\n{DRAFT_INSTRUCTION}"
        )
        return {
            "messages": [*state["messages"], {"role": "user", "content": note}],
            "retrieved": sorted(toolbox.retrieved),
        }

    def draft(state: InvestigationState) -> dict[str, Any]:
        try:
            reply = client.respond(
                system=SYSTEM,
                messages=state["messages"],
                tools=TOOL_SCHEMAS,
                effort=effort,
                output_schema=CASE_SCHEMA,
            )
        except (BudgetExceededError, ModelRefusedError) as error:
            return {"status": "incomplete", "incomplete_reason": f"{type(error).__name__}: {error}"}
        messages = [*state["messages"], {"role": "assistant", "content": reply.content}]
        usage = _add_usage(state, reply.usage)
        if reply.tool_uses():
            # The model asked for more data while revising. Answer, then ask again.
            toolbox = _toolbox(state)
            results = [
                {
                    "type": "tool_result",
                    "tool_use_id": b["id"],
                    "content": toolbox.call(b["name"], b["input"]),
                }
                for b in reply.tool_uses()
            ]
            return {
                "messages": [*messages, {"role": "user", "content": results}],
                "usage": usage,
                "retrieved": sorted(toolbox.retrieved),
                "tool_rounds": state["tool_rounds"] + 1,
                "draft": None,
                "findings": [
                    {"level": "RETRY", "claim_id": "draft", "message": "needed more data"}
                ],
            }
        try:
            case = CaseFile.model_validate_json(reply.text())
        except ValidationError as error:
            return {
                "messages": messages,
                "usage": usage,
                "draft": None,
                "findings": [{"level": "SCHEMA", "claim_id": "draft", "message": str(error)[:800]}],
            }
        # The alert ID comes from the system, not from whatever the draft says.
        case.alert_id = state["alert"]["alert_id"]
        return {"messages": messages, "usage": usage, "draft": case.model_dump(), "findings": []}

    def check(state: InvestigationState) -> dict[str, Any]:
        if state.get("draft") is None:
            return {}
        case = CaseFile.model_validate(state["draft"])
        findings = verify(
            case,
            store,
            retrieved_ids=state.get("retrieved", []),
            as_of=state["alert"]["as_of"],
        )
        return {"findings": [f.as_dict() for f in findings]}

    def after_verify(state: InvestigationState) -> str:
        if state.get("status") == "incomplete":
            return "human_review"
        if state.get("findings"):
            return "revise" if state["revisions"] < MAX_REVISIONS else "flag_incomplete"
        return "human_review"

    def revise(state: InvestigationState) -> dict[str, Any]:
        listing = "\n".join(
            f"- [{f['level']}] {f['claim_id']}: {f['message']}" for f in state["findings"]
        )
        note = REVISE_INSTRUCTION.format(findings=listing)
        messages = list(state["messages"])
        if messages[-1]["role"] == "user":
            # Two consecutive user turns aren't allowed, so combine with pending tool results.
            content = messages[-1]["content"]
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            messages[-1] = {"role": "user", "content": [*blocks, {"type": "text", "text": note}]}
        else:
            messages.append({"role": "user", "content": note})
        return {"messages": messages, "revisions": state["revisions"] + 1}

    def flag_incomplete(state: InvestigationState) -> dict[str, Any]:
        return {
            "status": "incomplete",
            "incomplete_reason": f"failed verification after {state['revisions']} revisions",
        }

    def human_review(state: InvestigationState) -> dict[str, Any]:
        status = "incomplete" if state.get("status") == "incomplete" else "pending_review"
        decision = interrupt(
            {
                "alert_id": state["alert"]["alert_id"],
                "status": status,
                "incomplete_reason": state.get("incomplete_reason"),
                "draft": state.get("draft"),
                "findings": state.get("findings", []),
            }
        )
        return {"decision": decision, "status": status}

    def record(state: InvestigationState) -> dict[str, Any]:
        decision = state.get("decision") or {}
        check_decision(decision, state.get("draft"))
        action = decision["action"]
        if decision.get("edited_draft") is not None:
            edited = CaseFile.model_validate(decision["edited_draft"])
            return {
                "draft": edited.model_dump(),
                "status": "approved" if action == "approve" else "rejected",
            }
        return {"status": "approved" if action == "approve" else "rejected"}

    graph: StateGraph[Any] = StateGraph(InvestigationState)
    graph.add_node("load_alert", load_alert)
    graph.add_node("gather", gather)
    graph.add_node("tools", tools)
    graph.add_node("run_detectors", run_detectors)
    graph.add_node("draft", draft)
    graph.add_node("verify", check)
    graph.add_node("revise", revise)
    graph.add_node("flag_incomplete", flag_incomplete)
    graph.add_node("human_review", human_review)
    graph.add_node("record", record)

    graph.add_edge(START, "load_alert")
    graph.add_edge("load_alert", "gather")
    graph.add_conditional_edges("gather", after_gather)
    graph.add_conditional_edges("tools", after_tools)
    graph.add_edge("run_detectors", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges("verify", after_verify)
    graph.add_edge("revise", "draft")
    graph.add_edge("flag_incomplete", "human_review")
    graph.add_edge("human_review", "record")
    graph.add_edge("record", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def investigate(graph: Any, alert: Alert) -> dict[str, Any]:
    """Run to the human-review pause, and return the state an analyst would see."""
    config = {"configurable": {"thread_id": alert.alert_id}, "recursion_limit": 100}
    state = {"alert": alert.__dict__.copy()}
    graph.invoke(state, config)
    snapshot = graph.get_state(config)
    pause = snapshot.tasks[0].interrupts[0].value if snapshot.tasks else {}
    return {"pause": pause, "values": snapshot.values}


def to_json(state: dict[str, Any]) -> str:
    return json.dumps(state, default=str)
