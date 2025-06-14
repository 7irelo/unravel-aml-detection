"""Agent metrics against ground truth, plus the rule baseline they must beat."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from unravel.agent.case import CaseFile
from unravel.agent.detectors import rule_baseline_label
from unravel.agent.graph import build_graph, investigate
from unravel.agent.llm import ModelClient, Usage
from unravel.agent.store import TxStore, neighbourhood
from unravel.agent.verifier import verify
from unravel.agent_eval.dataset import EvalCase
from unravel.agent_eval.mutations import MUTATIONS

MINUTES_PER_DAY = 1440
_NON_PATTERN = {"natural", "false_positive"}


@dataclass
class CaseResult:
    case: EvalCase
    status: str
    draft: CaseFile | None
    findings: list[dict[str, str]]
    revisions: int
    usage: Usage
    retrieved: list[str]
    incomplete_reason: str | None


def run_agent(
    cases: Iterable[EvalCase], store: TxStore, client: ModelClient, *, effort: str = "high"
) -> list[CaseResult]:
    """Investigate each alert up to the human-review pause. No analyst decision is simulated."""
    graph = build_graph(store, client, effort=effort)
    results = []
    for case in cases:
        output = investigate(graph, case.alert)
        values = output["values"]
        draft = CaseFile.model_validate(values["draft"]) if values.get("draft") else None
        results.append(
            CaseResult(
                case,
                output["pause"].get("status", values.get("status", "unknown")),
                draft,
                values.get("findings", []),
                values.get("revisions", 0),
                Usage(**values.get("usage", {})),
                values.get("retrieved", []),
                values.get("incomplete_reason"),
            )
        )
    return results


def macro_f1(truth: list[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(truth))
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def _confusion(truth: list[str], predicted: Sequence[str]) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for t, p in zip(truth, predicted, strict=True):
        table[t][p] += 1
    return {t: dict(row) for t, row in sorted(table.items())}


def pattern_metrics(truth: list[str], predicted: Sequence[str]) -> dict[str, Any]:
    n = len(truth)
    correct = sum(t == p for t, p in zip(truth, predicted, strict=True))
    return {
        "n": n,
        "accuracy": correct / n if n else float("nan"),
        "macro_f1": macro_f1(truth, predicted),
        "confusion": _confusion(truth, predicted),
    }


def rule_baseline(cases: list[EvalCase], store: TxStore, lookback_days: int = 7) -> dict[str, Any]:
    """Pattern accuracy using detectors alone, and how often they recover the true transactions."""
    truth: list[str] = []
    predicted: list[str] = []
    for case in cases:
        if case.truth_pattern is None:
            continue
        seed = store.get(case.alert.tx_id)
        if seed is None:
            continue
        start = case.alert.as_of - lookback_days * MINUTES_PER_DAY
        edges, _ = neighbourhood(store, [seed.src, seed.dst], start, case.alert.as_of, hops=2)
        truth.append(case.truth_pattern)
        predicted.append(rule_baseline_label(edges, seed))
    return pattern_metrics(truth, predicted)


def summarise(results: list[CaseResult], model: str) -> dict[str, Any]:
    """Metrics that need no judge. Everything is reported per case, then pooled."""
    total = len(results)
    drafted = [r for r in results if r.draft is not None]
    verified = [r for r in drafted if not r.findings]

    truth = [r.case.truth_pattern for r in results if r.case.truth_pattern and r.draft]
    predicted = [r.draft.pattern for r in results if r.case.truth_pattern and r.draft is not None]

    evidence_recall = []
    for r in results:
        if r.case.truth_tx_ids and r.draft is not None:
            cited = {i for c in r.draft.all_claims() for i in c.tx_ids}
            evidence_recall.append(len(cited & set(r.case.truth_tx_ids)) / len(r.case.truth_tx_ids))

    escalated = [
        r.case.is_laundering for r in results if r.draft and r.draft.recommendation == "escalate"
    ]
    laundering = [r for r in results if r.case.is_laundering]
    false_positives = [r for r in results if not r.case.is_laundering]
    missed = [r for r in laundering if r.draft is not None and r.draft.recommendation != "escalate"]
    declined = [
        r for r in false_positives if r.draft is not None and r.draft.recommendation != "escalate"
    ]

    usage = Usage()
    for r in results:
        usage.add(r.usage)

    levels: Counter[str] = Counter(f["level"] for r in results for f in r.findings)
    return {
        "model": model,
        "cases": total,
        "strata": dict(Counter(r.case.stratum for r in results)),
        "status": dict(Counter(r.status for r in results)),
        "drafted": len(drafted),
        "citation_validity_rate": len(verified) / len(drafted) if drafted else float("nan"),
        "remaining_finding_levels": dict(levels),
        "mean_revisions": float(np.mean([r.revisions for r in results])) if results else 0.0,
        "pattern": pattern_metrics(truth, predicted),
        "evidence_recall": float(np.mean(evidence_recall)) if evidence_recall else float("nan"),
        "escalate_precision": sum(escalated) / len(escalated) if escalated else float("nan"),
        "missed_laundering_rate": len(missed) / len(laundering) if laundering else float("nan"),
        "false_positive_decline_rate": len(declined) / len(false_positives)
        if false_positives
        else float("nan"),
        "usage": usage.__dict__.copy(),
        "cost_per_case_usd": usage.cost(model) / total if total else 0.0,
    }


def verifier_mutation_test(
    results: list[CaseResult], store: TxStore, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Corrupt clean drafts and measure how often the deterministic verifier notices each error."""
    rng = random.Random(seed)
    outcome: dict[str, dict[str, float]] = {}
    clean = [r for r in results if r.draft is not None and not r.findings]
    for name, (mutate, expected) in MUTATIONS.items():
        applied = caught = 0
        for result in clean:
            assert result.draft is not None
            mutated = mutate(result.draft, store, rng)
            if mutated is None:
                continue
            applied += 1
            findings = verify(
                mutated,
                store,
                retrieved_ids=result.retrieved,
                as_of=result.case.alert.as_of,
            )
            caught += bool(findings)
        outcome[name] = {
            "applied": float(applied),
            "caught": float(caught),
            "detection_rate": caught / applied if applied else float("nan"),
            "verifier_expected_to_catch": float(expected),
        }
    return outcome
