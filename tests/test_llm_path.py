"""The real LLM code path, driven by a fake SDK. It runs everything except the network.

``AnthropicClient`` wraps a stand-in that replays what the API returns, so request building, the
tool loop, structured drafts, the verifier and the judge all run for real. It shows the wiring works.
It says nothing about how good a real model's narratives are.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from unravel.agent.graph import build_graph, investigate
from unravel.agent.llm import AnthropicClient, SpendGuard
from unravel.agent.scripted import ScriptedClient
from unravel.agent.store import MemoryStore, Tx
from unravel.agent.tools import Alert
from unravel.agent_eval.dataset import EvalCase
from unravel.agent_eval.judge import DIMENSIONS, citation_support_rate, run_judge
from unravel.agent_eval.score import run_agent, summarise


def _tx(tx_id: str, minute: int, src: str, dst: str) -> Tx:
    return Tx(tx_id, minute, src, dst, 1000.0, "US Dollar", 1000.0, "US Dollar", "ACH", 1000.0)


CYCLE = [_tx("c1", 10, "a", "b"), _tx("c2", 20, "b", "c"), _tx("c3", 30, "c", "a")]


class _Block:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return self.data


class FakeSdk:
    """Answers like the API, using the scripted client's decisions, plus a thinking block."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.script = ScriptedClient()
        outer = self

        class Messages:
            def create(self, **kwargs: Any) -> Any:
                return outer.reply(kwargs)

        self.beta = SimpleNamespace(messages=Messages())

    def reply(self, kwargs: dict[str, Any]) -> Any:
        # Snapshot now: this list is what the API would receive on this exact request.
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        output = kwargs["output_config"].get("format")
        if kwargs["tools"] == []:
            body = {
                **dict.fromkeys(DIMENSIONS, 3),
                "claims": [{"claim_id": "c1", "supported": True, "reason": "matches"}],
            }
            blocks = [{"type": "text", "text": json.dumps(body)}]
        else:
            result = self.script.respond(
                system="",
                messages=kwargs["messages"],
                tools=kwargs["tools"],
                effort="high",
                output_schema=output and output["schema"],
            )
            blocks = [{"type": "thinking", "thinking": "", "signature": "sig"}, *result.content]
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return SimpleNamespace(
            content=[_Block(b) for b in blocks],
            stop_reason=stop,
            stop_details=None,
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=100,
                cache_read_input_tokens=500,
                cache_creation_input_tokens=0,
            ),
        )


def _alert() -> Alert:
    return Alert("A-c3", "c3", 30, 0.9, "critical", "test-model")


def test_full_investigation_through_the_real_client() -> None:
    sdk = FakeSdk()
    client = AnthropicClient(client=sdk, guard=SpendGuard(10.0))
    output = investigate(build_graph(MemoryStore(CYCLE), client), _alert())
    assert output["pause"]["status"] == "pending_review"
    assert output["values"]["draft"]["pattern"] == "CYCLE"
    assert output["values"]["findings"] == []
    assert client.guard is not None
    assert client.guard.usage.requests == len(sdk.requests)


def test_history_is_append_only_and_keeps_thinking_blocks() -> None:
    """Replayed thinking blocks are rejected if earlier turns change, so each request must extend."""
    sdk = FakeSdk()
    investigate(build_graph(MemoryStore(CYCLE), AnthropicClient(client=sdk)), _alert())
    assert len(sdk.requests) >= 3
    for earlier, later in zip(sdk.requests, sdk.requests[1:], strict=False):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]
    replayed = [
        block
        for message in sdk.requests[-1]["messages"]
        if message["role"] == "assistant"
        for block in message["content"]
        if block["type"] == "thinking"
    ]
    assert replayed
    assert all(block["signature"] == "sig" for block in replayed)


def test_only_the_draft_request_uses_structured_output() -> None:
    sdk = FakeSdk()
    investigate(build_graph(MemoryStore(CYCLE), AnthropicClient(client=sdk)), _alert())
    with_schema = [r for r in sdk.requests if "format" in r["output_config"]]
    assert len(with_schema) == 1
    assert with_schema[0]["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_evaluation_and_judge_run_through_the_real_client() -> None:
    sdk = FakeSdk()
    store = MemoryStore(CYCLE)
    client = AnthropicClient(client=sdk, guard=SpendGuard(10.0))
    case = EvalCase(_alert(), "CYCLE", True, "CYCLE", ("c1", "c2", "c3"))
    results = run_agent([case], store, client)
    summary = summarise(results, client.name)
    assert summary["pattern"]["accuracy"] == 1.0
    assert summary["cost_per_case_usd"] > 0
    assert results[0].draft is not None
    score = run_judge(results[0].draft, store, client)
    assert citation_support_rate([score]) == 1.0
    judge_request = sdk.requests[-1]
    assert judge_request["tools"] == []
