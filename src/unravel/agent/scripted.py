"""A deterministic stand-in for the LLM. It is NOT an AI and its output is not an LLM result.

It follows a fixed investigation script and writes a template narrative from detector output. It
exists so the graph, tools, verifier and review flow can run in CI and in the demo with no API key,
and it doubles as a template baseline in evaluations, reported under its own name.
"""

from __future__ import annotations

import json
from typing import Any

from unravel.agent.llm import ModelReply, Usage

_STRONG = 0.45  # below this a detector candidate isn't treated as grounds for escalation


class ScriptedClient:
    name = "scripted-template"

    def respond(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        effort: str,
        output_schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> ModelReply:
        if output_schema is not None:
            return self._draft(messages)
        return self._gather(messages)

    # -- gathering ---------------------------------------------------------------------------------

    def _gather(self, messages: list[dict[str, Any]]) -> ModelReply:
        calls = sum(
            1
            for m in messages
            if m["role"] == "assistant"
            for b in m["content"]
            if b.get("type") == "tool_use"
        )
        plan: list[dict[str, Any]] = [{"name": "get_alert", "input": {}}]
        results = _tool_results(messages)
        alert = next((r for r in results if "reason_codes" in r), None)
        if alert and alert.get("transaction"):
            tx = alert["transaction"]
            plan += [
                {"name": "get_account_profile", "input": {"account": tx["src"]}},
                {"name": "get_account_profile", "input": {"account": tx["dst"]}},
                {"name": "aggregate", "input": {"tx_ids": [tx["tx_id"]]}},
            ]
        if calls >= len(plan) and alert:
            return ModelReply(
                [{"type": "text", "text": "I have enough evidence."}], "end_turn", Usage(requests=1)
            )
        step = plan[min(calls, len(plan) - 1)]
        return ModelReply(
            [{"type": "tool_use", "id": f"call_{calls}", **step}], "tool_use", Usage(requests=1)
        )

    # -- drafting ----------------------------------------------------------------------------------

    def _draft(self, messages: list[dict[str, Any]]) -> ModelReply:
        results = _tool_results(messages)
        alert = next(r for r in results if "reason_codes" in r)
        tx = alert["transaction"]
        detector = _detector_result(messages)
        candidates = detector.get("candidates", [])
        top = candidates[0] if candidates else None
        claims = [
            {
                "id": "c1",
                "text": f"Alerted payment from {tx['src']} to {tx['dst']}.",
                "kind": "fact",
                "tx_ids": [tx["tx_id"]],
                "asserted": {"total_usd": tx["usd"], "count": 1},
                "supports": [],
            }
        ]
        contrary: list[dict[str, Any]] = []
        if top and top["strength"] >= _STRONG:
            pattern = top["pattern"]
            recommendation = "escalate"
            claims.append(
                {
                    "id": "c2",
                    "text": top["detail"],
                    "kind": "pattern",
                    "tx_ids": top["tx_ids"],
                    "asserted": {"count": len(top["tx_ids"])},
                    "supports": [],
                }
            )
            why = f"Detected structure: {top['detail']}."
        else:
            pattern = "NONE"
            recommendation = "insufficient_evidence" if top else "close"
            why = (
                f"The strongest structure found was weak: {top['detail']}."
                if top
                else "No recognised laundering structure was found around this payment."
            )
            contrary.append(
                {
                    "id": "c3",
                    "text": "Detectors found no strong laundering structure.",
                    "kind": "fact",
                    "tx_ids": [tx["tx_id"]],
                    "asserted": None,
                    "supports": [],
                }
            )
        case = {
            "alert_id": alert["alert_id"],
            "recommendation": recommendation,
            "pattern": pattern,
            "subjects": [tx["src"], tx["dst"]],
            "claims": claims,
            "contrary_evidence": contrary,
            "narrative": {
                "who": f"Sending account {tx['src']}; receiving account {tx['dst']}.",
                "what": f"A payment of ${tx['usd']:,.2f} by {tx['payment_format']}.",
                "when": f"Minute {tx['minute']} of the monitoring timeline.",
                "where": f"Paid in {tx['payment_currency']}, received in {tx['receiving_currency']}.",
                "why_suspicious": why,
            },
            "open_questions": ["Is this activity consistent with the customers' known business?"],
        }
        return ModelReply(
            [{"type": "text", "text": json.dumps(case)}], "end_turn", Usage(requests=1)
        )


def _tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] != "user" or not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block.get("type") == "tool_result":
                results.append(json.loads(block["content"]))
    return results


def _detector_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        content = message["content"]
        if message["role"] == "user" and isinstance(content, str) and "detector results" in content:
            start = content.index("{")
            result, _ = json.JSONDecoder().raw_decode(content[start:])
            detector: dict[str, Any] = result
            return detector
    return {}
