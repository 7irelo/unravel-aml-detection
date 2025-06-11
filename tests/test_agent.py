"""Agent tools, detectors, verifier and graph. No network and no LLM: the client is scripted."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from langgraph.types import Command

from unravel.agent.case import CaseFile, Claim, Narrative
from unravel.agent.detectors import detect, rule_baseline_label
from unravel.agent.graph import CASE_SCHEMA, build_graph, check_decision, investigate
from unravel.agent.llm import (
    AnthropicClient,
    BudgetExceededError,
    ModelRefusedError,
    ModelReply,
    SpendGuard,
    Usage,
    strict_json_schema,
)
from unravel.agent.scripted import ScriptedClient
from unravel.agent.store import MemoryStore, Tx, neighbourhood
from unravel.agent.tools import TOOL_SCHEMAS, Alert, Toolbox
from unravel.agent.verifier import verify
from unravel.agent_eval.dataset import EvalCase
from unravel.agent_eval.judge import (
    DIMENSIONS,
    JudgeScore,
    agreement,
    apply_bias_correction,
    citation_support_rate,
    export_gold_template,
    fit_bias_correction,
    judge_input,
    quadratic_weighted_kappa,
    read_gold,
    run_judge,
)
from unravel.agent_eval.mutations import MUTATIONS
from unravel.agent_eval.score import (
    macro_f1,
    pattern_metrics,
    rule_baseline,
    run_agent,
    summarise,
    verifier_mutation_test,
)


def tx(
    tx_id: str,
    minute: int,
    src: str,
    dst: str,
    usd: float = 1000.0,
    fmt: str = "ACH",
) -> Tx:
    return Tx(tx_id, minute, src, dst, usd, "US Dollar", usd, "US Dollar", fmt, usd)


def cycle() -> list[Tx]:
    return [
        tx("c1", 10, "a", "b"),
        tx("c2", 20, "b", "c"),
        tx("c3", 30, "c", "d"),
        tx("c4", 40, "d", "a"),
    ]


def alert_on(tx_id: str, as_of: int) -> Alert:
    return Alert(f"A-{tx_id}", tx_id, as_of, 0.4, "high", "test-model")


# -- store and tools ------------------------------------------------------------------------------


def test_tool_schemas_are_strict_and_closed() -> None:
    for tool in TOOL_SCHEMAS:
        assert tool["strict"] is True
        assert tool["input_schema"]["additionalProperties"] is False
        assert set(tool["input_schema"]["required"]) == set(tool["input_schema"]["properties"])


def test_no_tool_can_file_or_write() -> None:
    banned = {"file", "submit", "send", "write", "update", "delete", "approve", "post", "create"}
    assert not [t["name"] for t in TOOL_SCHEMAS if banned & set(t["name"].split("_"))]


def test_tools_never_show_data_after_the_alert() -> None:
    store = MemoryStore([*cycle(), tx("future", 99, "a", "z")])
    box = Toolbox(store, alert_on("c2", as_of=25))
    listed = json.loads(
        box.call("get_transactions", {"account": "a", "lookback_days": 7, "cursor": 0})
    )
    assert [t["tx_id"] for t in listed["transactions"]] == ["c1"]
    profile = json.loads(box.call("get_account_profile", {"account": "a"}))
    assert profile["transactions"] == 1
    near = json.loads(
        box.call("get_neighbourhood", {"account": "a", "hops": 3, "lookback_days": 7})
    )
    assert "future" not in {t["tx_id"] for t in near["edges_shown"]}
    agg = json.loads(box.call("aggregate", {"tx_ids": ["c1", "future"]}))
    assert agg["count"] == 1
    assert agg["unknown_or_later_ids"] == ["future"]
    flows = json.loads(
        box.call("trace_flows", {"tx_id": "future", "direction": "upstream", "hops": 2})
    )
    assert "error" in flows
    down = json.loads(
        box.call("trace_flows", {"tx_id": "c1", "direction": "downstream", "hops": 8})
    )
    assert [t["tx_id"] for t in down["path"]] == ["c1", "c2"]
    assert "future" not in box.retrieved


def test_tool_errors_return_as_data() -> None:
    box = Toolbox(MemoryStore(cycle()), alert_on("c1", 50))
    assert "error" in json.loads(box.call("nope", {}))
    assert "error" in json.loads(box.call("get_account_profile", {"wrong": 1}))


def test_nothing_in_a_tool_response_exposes_labels() -> None:
    store = MemoryStore(cycle())
    assert not {"is_laundering", "pattern", "label"} & set(store.get("c1").as_dict())  # type: ignore[union-attr]


def test_neighbourhood_limits_are_reported() -> None:
    rows = [tx(f"h{i}", i, "hub", f"p{i}") for i in range(50)]
    edges, truncated = neighbourhood(MemoryStore(rows), ["hub"], 0, 100, hops=2, max_degree=10)
    assert truncated
    assert len(edges) == 10


# -- detectors ------------------------------------------------------------------------------------


def _label(rows: list[Tx], seed: str) -> str:
    return rule_baseline_label(rows, next(t for t in rows if t.tx_id == seed))


def test_detects_cycle() -> None:
    assert _label(cycle(), "c4") == "CYCLE"


def test_detects_fan_out_and_fan_in() -> None:
    fan = [tx(f"f{i}", i, "hub", f"p{i}") for i in range(8)]
    assert _label(fan, "f7") == "FAN-OUT"
    gather = [tx(f"g{i}", i, f"p{i}", "hub") for i in range(8)]
    assert _label(gather, "g7") == "FAN-IN"


def test_detects_scatter_gather() -> None:
    rows = [tx(f"a{i}", i, "s", f"m{i}") for i in range(4)]
    rows += [tx(f"b{i}", 10 + i, f"m{i}", "t") for i in range(4)]
    assert _label(rows, "b3") == "SCATTER-GATHER"


def test_detects_gather_scatter() -> None:
    rows = [tx(f"a{i}", i, f"x{i}", "hub") for i in range(3)]
    rows += [tx(f"b{i}", 10 + i, "hub", f"y{i}") for i in range(3)]
    assert _label(rows, "b1") == "GATHER-SCATTER"


def test_detects_bipartite() -> None:
    rows = [tx(f"e{i}{j}", i * 5 + j, f"l{i}", f"r{j}") for i in range(3) for j in range(3)]
    assert _label(rows, "e11") == "BIPARTITE"


def test_detects_stack() -> None:
    rows = [tx(f"e{i}{j}", i + j, f"l{i}", f"m{j}") for i in range(2) for j in range(2)]
    rows += [tx(f"f{i}{j}", 10 + i + j, f"m{i}", f"r{j}") for i in range(2) for j in range(2)]
    assert _label(rows, "e00") == "STACK"


def test_detects_chain_as_random() -> None:
    rows = [tx(f"h{i}", i, f"n{i}", f"n{i + 1}") for i in range(5)]
    assert _label(rows, "h2") == "RANDOM"


def test_isolated_payment_matches_nothing() -> None:
    rows = [tx("only", 0, "a", "b")]
    assert detect(rows, rows[0]) == []
    assert _label(rows, "only") == "NONE"


def test_cycle_must_respect_time_order() -> None:
    rows = [tx("late", 50, "b", "c"), tx("early", 5, "c", "a"), tx("seed", 30, "a", "b")]
    # Funds can't return through a hop that happened before the money moved along the previous one.
    assert "CYCLE" not in {c.pattern for c in detect(rows, rows[2])}


def test_self_transfers_are_not_structure() -> None:
    rows = [tx(f"s{i}", i, "a", "a") for i in range(10)]
    assert _label(rows, "s0") == "NONE"


# -- verifier -------------------------------------------------------------------------------------


def good_case() -> CaseFile:
    return CaseFile(
        alert_id="A",
        recommendation="escalate",
        pattern="CYCLE",
        subjects=["a"],
        claims=[
            Claim(
                id="c1",
                text="Funds returned to a in four hops.",
                kind="pattern",
                tx_ids=["c1", "c2", "c3", "c4"],
                asserted={"total_usd": 4000.0, "count": 4, "first_minute": 10, "last_minute": 40},
            ),
            Claim(id="c2", text="Layering.", kind="inference", supports=["c1"]),
        ],
        contrary_evidence=[Claim(id="c3", text="Round", kind="fact", tx_ids=["c1"])],
        narrative=Narrative(
            who="a, b, c, d",
            what="$4,000.00 moved",
            when="m10 to m40",
            where="USD",
            why_suspicious="loop",
        ),
    )


IDS = ["c1", "c2", "c3", "c4"]


def _verify(case: CaseFile, ids: list[str] = IDS, as_of: int = 40) -> set[str]:
    return {f.level for f in verify(case, MemoryStore(cycle()), retrieved_ids=ids, as_of=as_of)}


def test_clean_case_passes() -> None:
    assert _verify(good_case()) == set()


def test_l1_fabricated_id() -> None:
    case = good_case()
    case.claims[0].tx_ids.append("Tmade-up")
    assert "L1" in _verify(case)


def test_l2_id_not_retrieved_in_this_run() -> None:
    assert "L2" in _verify(good_case(), ids=["c1"])


def test_l3_wrong_numbers() -> None:
    for field, value in (
        ("total_usd", 9000.0),
        ("count", 2),
        ("first_minute", 1),
        ("last_minute", 2),
    ):
        case = good_case()
        setattr(case.claims[0].asserted, field, value)
        assert "L3" in _verify(case), field


def test_l3_tolerance_allows_rounding_only() -> None:
    case = good_case()
    case.claims[0].asserted.total_usd = 4000.4  # type: ignore[union-attr]
    assert _verify(case) == set()


def test_l4_prose_figure_with_no_backing() -> None:
    case = good_case()
    case.narrative.what = "US$1,234,567.00 moved"
    assert "L4" in _verify(case)


def test_l5_citation_after_alert() -> None:
    assert "L5" in _verify(good_case(), as_of=25)


def test_l6_cited_edges_do_not_form_the_claimed_shape() -> None:
    case = good_case()
    case.claims[0].tx_ids = ["c1", "c2"]
    case.claims[0].asserted = None
    assert "L6" in _verify(case)
    case = good_case()
    case.claims = [c for c in case.claims if c.kind != "pattern"]
    case.claims.append(Claim(id="c9", text="x", kind="fact", tx_ids=["c1"]))
    case.claims[0] = Claim(id="c2", text="x", kind="inference", supports=["c9"])
    assert "L6" in _verify(case)


def test_inference_needs_support_and_facts_need_citations() -> None:
    with pytest.raises(ValueError, match="must name"):
        Claim(id="i", text="x", kind="inference")
    with pytest.raises(ValueError, match="must cite"):
        Claim(id="f", text="x", kind="fact")
    case = good_case()
    case.claims[1].supports = ["missing"]
    assert "L1" in _verify(case)


# -- model client ---------------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return self.data


class _FakeResponse:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.content = [
            _FakeBlock({"type": "thinking", "thinking": "", "signature": "s"}),
            _FakeBlock({"type": "text", "text": "hi"}),
        ]
        self.stop_reason = stop_reason
        self.stop_details = type("D", (), {"category": "other"})()
        self.usage = type(
            "U",
            (),
            {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )()


class _FakeSdk:
    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self.stop_reason = stop_reason
        outer = self

        class Messages:
            def create(self, **kwargs: Any) -> _FakeResponse:
                outer.calls.append(kwargs)
                return _FakeResponse(outer.stop_reason)

        self.beta = type("B", (), {"messages": Messages()})()


def test_request_follows_current_api_rules() -> None:
    sdk = _FakeSdk()
    reply = AnthropicClient(client=sdk).respond(
        system="s", messages=[], tools=TOOL_SCHEMAS, effort="high", output_schema=CASE_SCHEMA
    )
    sent = sdk.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert not {"temperature", "top_p", "top_k"} & set(sent)
    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"]["effort"] == "high"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["fallbacks"] == "default"
    assert sent["betas"] == ["server-side-fallback-2026-07-01"]
    # Thinking blocks come back unchanged, so history can be replayed.
    assert reply.content[0]["type"] == "thinking"
    assert reply.text() == "hi"


def test_refusal_raises_rather_than_reading_content() -> None:
    with pytest.raises(ModelRefusedError):
        AnthropicClient(client=_FakeSdk("refusal")).respond(
            system="s", messages=[], tools=[], effort="low"
        )


def test_spend_guard_stops_further_requests() -> None:
    guard = SpendGuard(limit_usd=10.0)
    client = AnthropicClient(client=_FakeSdk(), guard=guard)
    client.respond(system="s", messages=[], tools=[], effort="low")  # costs $24 at list price
    assert guard.usage.cost() == pytest.approx(24.0)
    with pytest.raises(BudgetExceededError):
        client.respond(system="s", messages=[], tools=[], effort="low")


def test_case_schema_is_closed_at_every_level() -> None:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node.get("properties", {}))
            assert "default" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(strict_json_schema(CaseFile.model_json_schema()))


# -- graph ----------------------------------------------------------------------------------------


def _background() -> list[Tx]:
    return [*cycle(), tx("quiet", 15, "p", "q")]


def test_graph_reaches_review_with_a_verified_draft_and_files_nothing() -> None:
    store = MemoryStore(_background())
    graph = build_graph(store, ScriptedClient())
    out = investigate(graph, alert_on("c4", 40))
    assert out["pause"]["status"] == "pending_review"
    assert out["pause"]["findings"] == []
    assert out["values"]["draft"]["pattern"] == "CYCLE"
    assert out["values"]["status"] != "approved"  # nothing is decided until a human decides


def test_quiet_alert_is_not_escalated() -> None:
    graph = build_graph(MemoryStore(_background()), ScriptedClient())
    out = investigate(graph, alert_on("quiet", 15))
    assert out["values"]["draft"]["recommendation"] != "escalate"
    assert out["values"]["draft"]["contrary_evidence"]


@pytest.mark.parametrize(("action", "expected"), [("approve", "approved"), ("reject", "rejected")])
def test_analyst_decision_is_what_ends_the_case(action: str, expected: str) -> None:
    graph = build_graph(MemoryStore(_background()), ScriptedClient())
    alert = alert_on("c4", 40)
    investigate(graph, alert)
    config = {"configurable": {"thread_id": alert.alert_id}}
    final = graph.invoke(Command(resume={"action": action, "reason": "reviewed"}), config)
    assert final["status"] == expected


def test_rejection_requires_a_reason_and_invalid_actions_fail() -> None:
    for decision in ({"action": "reject", "reason": " "}, {"action": "file_it"}):
        graph = build_graph(MemoryStore(_background()), ScriptedClient())
        alert = alert_on("c4", 40)
        investigate(graph, alert)
        with pytest.raises(ValueError, match=r"reason|invalid"):
            graph.invoke(Command(resume=decision), {"configurable": {"thread_id": alert.alert_id}})


class _BadDrafter(ScriptedClient):
    """Always invents a citation, so verification can never pass."""

    def __init__(self) -> None:
        self.drafts = 0

    def respond(self, **kwargs: Any) -> ModelReply:
        reply = super().respond(**kwargs)
        if kwargs.get("output_schema") is None:
            return reply
        self.drafts += 1
        case = json.loads(reply.text())
        case["claims"][0]["tx_ids"].append("Tinvented")
        return ModelReply([{"type": "text", "text": json.dumps(case)}], "end_turn", Usage())


def test_failing_drafts_get_bounded_revisions_then_are_flagged() -> None:
    client = _BadDrafter()
    out = investigate(build_graph(MemoryStore(_background()), client), alert_on("c4", 40))
    assert client.drafts == 3  # the first attempt, plus two revisions
    assert out["pause"]["status"] == "incomplete"
    assert "verification" in out["pause"]["incomplete_reason"]
    assert any(f["level"] == "L1" for f in out["pause"]["findings"])


class _Malformed(ScriptedClient):
    def respond(self, **kwargs: Any) -> ModelReply:
        if kwargs.get("output_schema") is None:
            return super().respond(**kwargs)
        return ModelReply([{"type": "text", "text": "{not json"}], "end_turn", Usage())


def test_malformed_output_cannot_reach_an_analyst_as_a_draft() -> None:
    out = investigate(build_graph(MemoryStore(_background()), _Malformed()), alert_on("c4", 40))
    assert out["pause"]["status"] == "incomplete"
    assert out["pause"]["draft"] is None


class _Refuses(ScriptedClient):
    def respond(self, **kwargs: Any) -> ModelReply:
        raise ModelRefusedError("general_harms")


def test_refusal_reaches_a_human_with_the_reason() -> None:
    out = investigate(build_graph(MemoryStore(_background()), _Refuses()), alert_on("c4", 40))
    assert out["pause"]["status"] == "incomplete"
    assert "general_harms" in out["pause"]["incomplete_reason"]


def test_a_case_with_no_draft_cannot_be_approved_only_rejected() -> None:
    graph = build_graph(MemoryStore(_background()), _Refuses())
    alert = alert_on("c4", 40)
    investigate(graph, alert)
    config = {"configurable": {"thread_id": alert.alert_id}}
    # Validated before resuming, so a bad request can't leave the case stuck.
    with pytest.raises(ValueError, match="no draft"):
        check_decision({"action": "approve"}, None)
    final = graph.invoke(Command(resume={"action": "reject", "reason": "no draft"}), config)
    assert final["status"] == "rejected"


def test_invalid_edited_draft_is_refused_before_resuming() -> None:
    with pytest.raises(ValueError, match="validation error"):
        check_decision({"action": "approve", "edited_draft": {"pattern": "CYCLE"}}, None)


class _Endless(ScriptedClient):
    def respond(self, **kwargs: Any) -> ModelReply:
        if kwargs.get("output_schema") is not None:
            return super().respond(**kwargs)
        return ModelReply(
            [{"type": "tool_use", "id": "x", "name": "get_alert", "input": {}}], "tool_use", Usage()
        )


def test_tool_loop_is_bounded() -> None:
    out = investigate(build_graph(MemoryStore(_background()), _Endless()), alert_on("c4", 40))
    assert out["values"]["tool_rounds"] == 12


# -- evaluation -----------------------------------------------------------------------------------


def _eval_cases() -> list[EvalCase]:
    return [
        EvalCase(alert_on("c4", 40), "CYCLE", True, "CYCLE", tuple(IDS)),
        EvalCase(alert_on("quiet", 15), "false_positive", False, None, ()),
    ]


def test_scores_and_summary_on_scripted_runs() -> None:
    store = MemoryStore(_background())
    results = run_agent(_eval_cases(), store, ScriptedClient())
    summary = summarise(results, "scripted-template")
    assert summary["pattern"]["accuracy"] == 1.0
    assert summary["evidence_recall"] == 1.0
    assert summary["false_positive_decline_rate"] == 1.0
    assert summary["missed_laundering_rate"] == 0.0
    assert summary["citation_validity_rate"] == 1.0
    assert rule_baseline(_eval_cases(), store)["accuracy"] == 1.0


def test_verifier_catches_each_mutation_it_should() -> None:
    # A realistic store: most transactions are ones this investigation never looked at.
    unrelated = [tx(f"u{i}", 5, f"x{i}", f"y{i}", 50.0 + i) for i in range(5000)]
    store = MemoryStore([*_background(), *unrelated])
    results = run_agent(_eval_cases(), store, ScriptedClient())
    outcome = verifier_mutation_test(results, store)
    for name, row in outcome.items():
        if row["applied"] and row["verifier_expected_to_catch"]:
            assert row["detection_rate"] == 1.0, name
    assert outcome["removed_contrary_evidence"]["detection_rate"] == 0.0


def test_every_mutator_leaves_the_original_untouched() -> None:
    store = MemoryStore(_background())
    case = good_case()
    before = case.model_dump()
    for mutate, _ in MUTATIONS.values():
        mutate(case, store, random.Random(0))
    assert case.model_dump() == before


def test_macro_f1_and_confusion() -> None:
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert pattern_metrics(["a"], ["b"])["confusion"] == {"a": {"b": 1}}


def test_kappa_known_values() -> None:
    assert quadratic_weighted_kappa([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert quadratic_weighted_kappa([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Near misses cost less than large misses.
    near = quadratic_weighted_kappa([1, 2, 3, 4, 1, 4], [1, 2, 3, 3, 2, 4])
    far = quadratic_weighted_kappa([1, 2, 3, 4, 1, 4], [4, 2, 3, 1, 2, 4])
    assert near > far


def test_agreement_reports_bias_and_correction_reduces_it() -> None:
    human = [dict.fromkeys(DIMENSIONS, v) for v in (1, 2, 3, 2, 1, 3)]
    judge = [dict.fromkeys(DIMENSIONS, min(4, v + 1)) for v in (1, 2, 3, 2, 1, 3)]
    assert agreement(human, judge)[0].mean_bias == pytest.approx(1.0)
    correction = fit_bias_correction(human, judge)
    assert apply_bias_correction(judge[0], correction) == human[0]


class _FakeJudge(ScriptedClient):
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def respond(self, **kwargs: Any) -> ModelReply:
        self.seen.append(kwargs)
        body = {
            **dict.fromkeys(DIMENSIONS, 3),
            "claims": [
                {"claim_id": "c1", "supported": True, "reason": "ok"},
                {"claim_id": "c2", "supported": False, "reason": "no"},
            ],
        }
        return ModelReply([{"type": "text", "text": json.dumps(body)}], "end_turn", Usage())


def test_judge_sees_only_the_narrative_and_evidence() -> None:
    store = MemoryStore(_background())
    client = _FakeJudge()
    score = run_judge(good_case(), store, client)
    assert isinstance(score, JudgeScore)
    assert citation_support_rate([score]) == 0.5
    sent = client.seen[0]
    assert sent["tools"] == []
    body = sent["messages"][0]["content"]
    for forbidden in ("is_laundering", "truth", "stratum", "retrieved", "thinking"):
        assert forbidden not in body
    assert "cited_transactions" in json.loads(judge_input(good_case(), store))


def test_gold_sheet_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = MemoryStore(_background())
    path = tmp_path / "gold.csv"
    export_gold_template([("case-1", good_case())], store, path)
    assert read_gold(path) == {}  # unrated rows aren't treated as ratings
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("case_id,case_json,completeness")
