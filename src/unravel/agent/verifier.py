"""Deterministic verification of a draft, L1 to L6. These are hard gates.

L7, whether cited transactions truly support a qualitative claim, needs judgement and lives in
``agent_eval``. Everything here is computed from the transactions themselves, so a model's confidence
plays no part.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from unravel.agent.case import CaseFile, Claim
from unravel.agent.store import Tx, TxStore

AMOUNT_TOLERANCE = 0.01  # 1% for USD totals: rounding happens, but real errors don't pass
_MONEY = re.compile(r"(?:USD|US\$|\$)\s?([0-9][0-9,]*(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class Finding:
    level: str
    claim_id: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _check_asserted(claim: Claim, rows: list[Tx], findings: list[Finding]) -> None:
    asserted = claim.asserted
    if asserted is None:
        return
    if asserted.total_usd is not None:
        actual = sum(t.usd for t in rows)
        allowed = max(1.0, abs(actual) * AMOUNT_TOLERANCE)
        if abs(asserted.total_usd - actual) > allowed:
            findings.append(
                Finding(
                    "L3", claim.id, f"claimed total {asserted.total_usd:,.2f}, actual {actual:,.2f}"
                )
            )
    if asserted.count is not None and asserted.count != len(rows):
        findings.append(Finding("L3", claim.id, f"claimed {asserted.count} tx, cited {len(rows)}"))
    if rows and asserted.first_minute is not None:
        first = min(t.minute for t in rows)
        if asserted.first_minute != first:
            findings.append(Finding("L3", claim.id, f"first minute is {first}"))
    if rows and asserted.last_minute is not None:
        last = max(t.minute for t in rows)
        if asserted.last_minute != last:
            findings.append(Finding("L3", claim.id, f"last minute is {last}"))
    if asserted.distinct_counterparties is not None:
        accounts = {t.src for t in rows} | {t.dst for t in rows}
        # A claim's counterparties exclude its most central account, when one exists.
        counts: dict[str, int] = defaultdict(int)
        for t in rows:
            counts[t.src] += 1
            counts[t.dst] += 1
        centre = max(counts, key=lambda a: counts[a], default=None)
        distinct = len(accounts - {centre}) if centre else 0
        if asserted.distinct_counterparties not in (distinct, len(accounts)):
            findings.append(Finding("L3", claim.id, f"distinct counterparties is {distinct}"))


def _shape_problem(pattern: str, rows: list[Tx]) -> str | None:
    """L6: do the cited edges form the claimed structure at all?"""
    clean = [t for t in rows if t.src != t.dst]
    if not clean:
        return "no edges between distinct accounts"
    sources = {t.src for t in clean}
    targets = {t.dst for t in clean}
    if pattern == "FAN-OUT":
        return (
            None
            if any(len({t.dst for t in clean if t.src == s}) >= 2 for s in sources)
            else "no account pays multiple recipients"
        )
    if pattern == "FAN-IN":
        return (
            None
            if any(len({t.src for t in clean if t.dst == d}) >= 2 for d in targets)
            else "no account receives from multiple senders"
        )
    if pattern == "CYCLE":
        graph: dict[str, set[str]] = defaultdict(set)
        for t in clean:
            graph[t.src].add(t.dst)
        for start in graph:
            stack, seen = list(graph[start]), set()
            while stack:
                node = stack.pop()
                if node == start:
                    return None
                if node not in seen:
                    seen.add(node)
                    stack.extend(graph.get(node, ()))
        return "cited edges contain no closed loop"
    if pattern == "SCATTER-GATHER":
        middles = sources & targets
        return None if len(middles) >= 2 else "fewer than two intermediaries"
    if pattern == "GATHER-SCATTER":
        for hub in sources & targets:
            if (
                len({t.src for t in clean if t.dst == hub}) >= 2
                and len({t.dst for t in clean if t.src == hub}) >= 2
            ):
                return None
        return "no hub with multiple inflows and outflows"
    if pattern in ("BIPARTITE", "STACK"):
        if len(sources) >= 2 and len(targets) >= 2:
            return None
        return "needs at least two senders and two receivers"
    if pattern == "RANDOM":
        return None if len(clean) >= 2 else "needs at least two hops"
    return None


def verify(
    case: CaseFile,
    store: TxStore,
    *,
    retrieved_ids: Iterable[str],
    as_of: int,
) -> list[Finding]:
    """Return all problems. An empty list means every deterministic gate passed."""
    findings: list[Finding] = []
    retrieved = set(retrieved_ids)
    claim_ids = {c.id for c in case.all_claims()}
    cited_by_pattern: list[Tx] = []
    checked_amounts: list[float] = []

    for claim in case.all_claims():
        rows: list[Tx] = []
        for tx_id in claim.tx_ids:
            tx = store.get(tx_id)
            if tx is None:
                findings.append(Finding("L1", claim.id, f"{tx_id} does not exist"))
                continue
            rows.append(tx)
            if tx_id not in retrieved:
                findings.append(Finding("L2", claim.id, f"{tx_id} was not retrieved in this run"))
            if tx.minute > as_of:
                findings.append(Finding("L5", claim.id, f"{tx_id} is after the alert time"))
        for support in claim.supports:
            if support not in claim_ids:
                findings.append(Finding("L1", claim.id, f"supports unknown claim {support}"))
        _check_asserted(claim, rows, findings)
        if claim.asserted is not None and claim.asserted.total_usd is not None:
            checked_amounts.append(claim.asserted.total_usd)
        checked_amounts.extend(t.usd for t in rows)
        if claim.kind == "pattern":
            cited_by_pattern.extend(rows)

    if case.pattern not in ("NONE", "OTHER_SUSPICIOUS"):
        if not cited_by_pattern:
            findings.append(
                Finding("L6", "case", f"pattern {case.pattern} has no pattern claim with citations")
            )
        else:
            problem = _shape_problem(case.pattern, cited_by_pattern)
            if problem:
                findings.append(Finding("L6", "case", f"{case.pattern}: {problem}"))

    # L4: a dollar figure in the prose must match a checked value, not just appear plausible.
    for match in _MONEY.finditer(case.narrative.full_text()):
        figure = float(match.group(1).replace(",", ""))
        if not any(abs(figure - a) <= max(1.0, abs(a) * AMOUNT_TOLERANCE) for a in checked_amounts):
            findings.append(Finding("L4", "narrative", f"${figure:,.2f} matches no checked claim"))
    return findings
