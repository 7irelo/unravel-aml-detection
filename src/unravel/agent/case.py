"""The case file schema. Every factual statement is a structured claim that can be checked."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Recommendation = Literal["escalate", "close", "insufficient_evidence"]
PatternLabel = Literal[
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "BIPARTITE",
    "STACK",
    "RANDOM",
    "OTHER_SUSPICIOUS",
    "NONE",
]
ClaimKind = Literal["fact", "aggregate", "pattern", "inference"]
ReviewStatus = Literal["pending_review", "incomplete", "approved", "rejected"]


class Asserted(BaseModel):
    """Machine-checkable values a claim makes about its cited transactions."""

    model_config = ConfigDict(extra="forbid")

    total_usd: float | None = None
    count: int | None = None
    first_minute: int | None = None
    last_minute: int | None = None
    distinct_counterparties: int | None = None


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    kind: ClaimKind
    tx_ids: list[str] = Field(default_factory=list)
    asserted: Asserted | None = None
    supports: list[str] = Field(default_factory=list)  # for inference: the claims it rests on

    @model_validator(mode="after")
    def _evidence_required(self) -> Claim:
        if self.kind == "inference":
            if not self.supports:
                raise ValueError(f"inference {self.id} must name the claims it rests on")
        elif not self.tx_ids:
            raise ValueError(f"{self.kind} claim {self.id} must cite transactions")
        return self


class Narrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    who: str
    what: str
    when: str
    where: str
    why_suspicious: str

    def full_text(self) -> str:
        return "\n".join((self.who, self.what, self.when, self.where, self.why_suspicious))


class CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str
    recommendation: Recommendation
    pattern: PatternLabel
    subjects: list[str]
    claims: list[Claim]
    contrary_evidence: list[Claim] = Field(default_factory=list)
    narrative: Narrative
    open_questions: list[str] = Field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        return [*self.claims, *self.contrary_evidence]
