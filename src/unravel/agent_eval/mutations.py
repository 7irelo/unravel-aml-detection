"""Seed known errors into case files, to measure what the verifier and the judge catch."""

from __future__ import annotations

import random
from collections.abc import Callable

from unravel.agent.case import CaseFile
from unravel.agent.store import TxStore

Mutator = Callable[[CaseFile, TxStore, random.Random], CaseFile | None]

_CONFUSABLE = {
    "FAN-OUT": "FAN-IN",
    "FAN-IN": "FAN-OUT",
    "CYCLE": "FAN-OUT",
    "SCATTER-GATHER": "CYCLE",
    "GATHER-SCATTER": "CYCLE",
    "BIPARTITE": "CYCLE",
    "STACK": "CYCLE",
    "RANDOM": "BIPARTITE",
}


def _cited_claim(case: CaseFile) -> int | None:
    for i, claim in enumerate(case.claims):
        if claim.tx_ids:
            return i
    return None


def fabricated_id(case: CaseFile, store: TxStore, rng: random.Random) -> CaseFile | None:
    index = _cited_claim(case)
    if index is None:
        return None
    out = case.model_copy(deep=True)
    out.claims[index].tx_ids[0] = f"T{rng.randrange(16**15):015x}"
    return out


def inflated_total(case: CaseFile, store: TxStore, rng: random.Random) -> CaseFile | None:
    out = case.model_copy(deep=True)
    for claim in out.claims:
        if claim.asserted is not None and claim.asserted.total_usd is not None:
            claim.asserted.total_usd = claim.asserted.total_usd * 1.5 + 100
            return out
    return None


def wrong_count(case: CaseFile, store: TxStore, rng: random.Random) -> CaseFile | None:
    out = case.model_copy(deep=True)
    for claim in out.claims:
        if claim.asserted is not None and claim.asserted.count is not None:
            claim.asserted.count += 3
            return out
    return None


def unsupported_dollar_figure(
    case: CaseFile, store: TxStore, rng: random.Random
) -> CaseFile | None:
    out = case.model_copy(deep=True)
    out.narrative.what += " A further $9,876,543.21 moved through the same route."
    return out


def wrong_typology(case: CaseFile, store: TxStore, rng: random.Random) -> CaseFile | None:
    replacement = _CONFUSABLE.get(case.pattern)
    if replacement is None:
        return None
    out = case.model_copy(deep=True)
    out.pattern = replacement  # type: ignore[assignment]
    return out


def missing_pattern_evidence(case: CaseFile, store: TxStore, rng: random.Random) -> CaseFile | None:
    if case.pattern in ("NONE", "OTHER_SUSPICIOUS"):
        return None
    out = case.model_copy(deep=True)
    out.claims = [c for c in out.claims if c.kind != "pattern"]
    return out


def swapped_citation(case: CaseFile, store: TxStore, rng: random.Random) -> CaseFile | None:
    """Replace a citation with a real but irrelevant transaction the agent never looked at."""
    index = _cited_claim(case)
    if index is None or not hasattr(store, "by_id"):
        return None
    out = case.model_copy(deep=True)
    pool = store.by_id  # type: ignore[attr-defined]
    keys = list(pool)
    out.claims[index].tx_ids[0] = keys[rng.randrange(len(keys))]
    return out


def removed_contrary_evidence(
    case: CaseFile, store: TxStore, rng: random.Random
) -> CaseFile | None:
    """Passes every deterministic check on purpose: only a judge or a human can notice this."""
    if not case.contrary_evidence:
        return None
    out = case.model_copy(deep=True)
    out.contrary_evidence = []
    return out


#: The value says whether the deterministic verifier is expected to catch this mutation.
MUTATIONS: dict[str, tuple[Mutator, bool]] = {
    "fabricated_id": (fabricated_id, True),
    "swapped_citation": (swapped_citation, True),
    "inflated_total": (inflated_total, True),
    "wrong_count": (wrong_count, True),
    "unsupported_dollar_figure": (unsupported_dollar_figure, True),
    "wrong_typology": (wrong_typology, True),
    "missing_pattern_evidence": (missing_pattern_evidence, True),
    "removed_contrary_evidence": (removed_contrary_evidence, False),
}
