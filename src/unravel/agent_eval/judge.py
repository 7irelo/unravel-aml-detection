"""LLM judge for what code can't check: narrative quality, and whether citations support claims.

"Calibrated" means measured against human ratings and seeded errors, not merely prompted well. The
judge sees only the narrative and the cited evidence, never the agent's reasoning or tool history.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from unravel.agent.case import CaseFile
from unravel.agent.llm import ModelClient, strict_json_schema
from unravel.agent.prompts import JUDGE_SYSTEM
from unravel.agent.store import TxStore

DIMENSIONS = (
    "completeness",
    "factual_accuracy",
    "typology_reasoning",
    "clarity",
    "proportionate_language",
    "actionability",
)
RUBRIC = """Score each dimension from 1 to 4.

completeness: 1 = who, what, when or where is missing. 2 = all present but thin. 3 = all present with \
specifics. 4 = also covers the sequence of events and what ties them together.
factual_accuracy: 1 = a statement contradicts the evidence. 2 = unsupported statements of substance. \
3 = all consistent with the evidence, with minor imprecision. 4 = exact throughout.
typology_reasoning: 1 = none, or the wrong typology. 2 = names one with little reasoning. 3 = links \
structure to typology. 4 = also addresses alternatives and contrary evidence.
clarity: 1 = hard to follow. 2 = understandable with effort. 3 = clear. 4 = an unfamiliar reader could \
act on it unaided.
proportionate_language: 1 = asserts crime or intent. 2 = loaded language. 3 = neutral with occasional \
overreach. 4 = fully factual and appropriately hedged.
actionability: 1 = no basis for a decision. 2 = a recommendation without enough support. 3 = a supported \
recommendation. 4 = also identifies what would change the conclusion.

For each claim, also say whether its cited transactions support it. A claim is unsupported if the \
transactions don't show what it says."""


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    supported: bool
    reason: str


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completeness: Literal[1, 2, 3, 4]
    factual_accuracy: Literal[1, 2, 3, 4]
    typology_reasoning: Literal[1, 2, 3, 4]
    clarity: Literal[1, 2, 3, 4]
    proportionate_language: Literal[1, 2, 3, 4]
    actionability: Literal[1, 2, 3, 4]
    claims: list[ClaimVerdict]


def judge_input(case: CaseFile, store: TxStore) -> str:
    """Only what the judge may see: the narrative, claims and cited transactions."""
    cited = sorted({i for c in case.all_claims() for i in c.tx_ids})
    evidence = [t.as_dict() for t in store.get_many(cited)]
    return json.dumps(
        {
            "recommendation": case.recommendation,
            "pattern": case.pattern,
            "narrative": case.narrative.model_dump(),
            "claims": [c.model_dump() for c in case.claims],
            "contrary_evidence": [c.model_dump() for c in case.contrary_evidence],
            "open_questions": case.open_questions,
            "cited_transactions": evidence,
        },
        indent=1,
    )


def run_judge(
    case: CaseFile, store: TxStore, client: ModelClient, effort: str = "high"
) -> JudgeScore:
    reply = client.respond(
        system=f"{JUDGE_SYSTEM}\n\n{RUBRIC}",
        messages=[{"role": "user", "content": judge_input(case, store)}],
        tools=[],
        effort=effort,
        output_schema=strict_json_schema(JudgeScore.model_json_schema()),
    )
    return JudgeScore.model_validate_json(reply.text())


def citation_support_rate(scores: list[JudgeScore]) -> float:
    verdicts = [v.supported for s in scores for v in s.claims]
    return sum(verdicts) / len(verdicts) if verdicts else float("nan")


# -- calibration against human ratings ------------------------------------------------------------


def quadratic_weighted_kappa(a: list[int], b: list[int], levels: int = 4) -> float:
    """Agreement beyond chance, with larger disagreements penalised more."""
    if len(a) != len(b) or not a:
        raise ValueError("ratings must be non-empty and the same length")
    observed = np.zeros((levels, levels))
    for x, y in zip(a, b, strict=True):
        observed[x - 1, y - 1] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(a)
    weights = np.array(
        [[(i - j) ** 2 / (levels - 1) ** 2 for j in range(levels)] for i in range(levels)]
    )
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator else float("nan")


@dataclass(frozen=True)
class Agreement:
    dimension: str
    n: int
    kappa: float
    mean_bias: float  # positive means the judge scores higher than the human


def agreement(human: list[dict[str, int]], judge: list[dict[str, int]]) -> list[Agreement]:
    out = []
    for dimension in DIMENSIONS:
        h = [row[dimension] for row in human]
        j = [row[dimension] for row in judge]
        out.append(
            Agreement(
                dimension,
                len(h),
                quadratic_weighted_kappa(h, j),
                float(np.mean(np.array(j) - np.array(h))),
            )
        )
    return out


def fit_bias_correction(
    human: list[dict[str, int]], judge: list[dict[str, int]]
) -> dict[str, float]:
    """A simple per-dimension offset, fit on the development split only."""
    return {
        d: float(np.mean([h[d] - j[d] for h, j in zip(human, judge, strict=True)]))
        for d in DIMENSIONS
    }


def apply_bias_correction(row: dict[str, int], correction: dict[str, float]) -> dict[str, int]:
    return {d: int(np.clip(round(row[d] + correction[d]), 1, 4)) for d in DIMENSIONS}


def export_gold_template(cases: list[tuple[str, CaseFile]], store: TxStore, path: Path) -> None:
    """Write a blind rating sheet. Nothing here identifies the model or the ground truth."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "case_json", *DIMENSIONS, "notes"])
        for case_id, case in cases:
            writer.writerow([case_id, judge_input(case, store), *[""] * len(DIMENSIONS), ""])


def read_gold(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["case_id"]: {d: int(row[d]) for d in DIMENSIONS}
        for row in rows
        if all(row[d].strip() for d in DIMENSIONS)
    }
