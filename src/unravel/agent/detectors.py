"""Deterministic detectors for the eight laundering typologies.

They do two jobs. They give the investigator structural facts it doesn't have to work out itself,
and they serve as the rule baseline an LLM must beat to justify its cost.

Each returns candidates with member transaction IDs, so anything claimed can be checked.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from unravel.agent.store import Tx

PATTERN_NAMES = (
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "BIPARTITE",
    "STACK",
    "RANDOM",
)
MIN_FAN = 4
MAX_CYCLE = 12


@dataclass(frozen=True)
class Candidate:
    pattern: str
    tx_ids: tuple[str, ...]
    accounts: tuple[str, ...]
    strength: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "tx_ids": list(self.tx_ids),
            "accounts": list(self.accounts),
            "strength": round(self.strength, 3),
            "detail": self.detail,
        }


def _non_self(edges: list[Tx]) -> list[Tx]:
    return [t for t in edges if t.src != t.dst]


def fan_out(edges: list[Tx], focus: set[str]) -> list[Candidate]:
    out: list[Candidate] = []
    by_src: dict[str, list[Tx]] = defaultdict(list)
    for t in _non_self(edges):
        if t.src in focus:
            by_src[t.src].append(t)
    for src, rows in by_src.items():
        targets = {t.dst for t in rows}
        if len(targets) >= MIN_FAN:
            senders_in = {t.src for t in edges if t.dst == src and t.src != src}
            # A pure fan-out moves money from one point; heavy inflow suggests another shape.
            purity = len(targets) / (len(targets) + len(senders_in))
            out.append(
                Candidate(
                    "FAN-OUT",
                    tuple(t.tx_id for t in rows),
                    (src, *sorted(targets)),
                    min(1.0, len(targets) / 12) * purity,
                    f"{src} paid {len(targets)} distinct accounts",
                )
            )
    return out


def fan_in(edges: list[Tx], focus: set[str]) -> list[Candidate]:
    out: list[Candidate] = []
    by_dst: dict[str, list[Tx]] = defaultdict(list)
    for t in _non_self(edges):
        if t.dst in focus:
            by_dst[t.dst].append(t)
    for dst, rows in by_dst.items():
        sources = {t.src for t in rows}
        if len(sources) >= MIN_FAN:
            receivers = {t.dst for t in edges if t.src == dst and t.dst != dst}
            purity = len(sources) / (len(sources) + len(receivers))
            out.append(
                Candidate(
                    "FAN-IN",
                    tuple(t.tx_id for t in rows),
                    (dst, *sorted(sources)),
                    min(1.0, len(sources) / 12) * purity,
                    f"{dst} received from {len(sources)} distinct accounts",
                )
            )
    return out


def cycles(edges: list[Tx], seed: Tx) -> list[Candidate]:
    """Find a shortest time-respecting path from ``seed.dst`` back to ``seed.src``.

    Combined with the seed, it is a closed loop. Earlier hops happen before later ones, and funds
    leave the seed's receiver only after the seed arrives, or the loop closes on the seed.
    """
    outgoing: dict[str, list[Tx]] = defaultdict(list)
    for t in _non_self(edges):
        outgoing[t.src].append(t)
    if seed.src == seed.dst:
        return []
    found: list[Candidate] = []
    for start, target, chain in ((seed.dst, seed.src, "forward"),):
        # Breadth-first over (account, earliest usable minute), remembering the path of edges.
        frontier: list[tuple[str, int, tuple[Tx, ...]]] = [(start, -1, ())]
        best_seen: dict[str, int] = {}
        for _ in range(MAX_CYCLE):
            next_frontier: list[tuple[str, int, tuple[Tx, ...]]] = []
            for account, after, path in frontier:
                for t in outgoing.get(account, ()):
                    if t.tx_id == seed.tx_id or t.minute < after:
                        continue
                    if t.dst == target:
                        members = (seed, *path, t)
                        accounts = tuple(dict.fromkeys(x.src for x in members))
                        found.append(
                            Candidate(
                                "CYCLE",
                                tuple(x.tx_id for x in members),
                                accounts,
                                min(1.0, 0.6 + 0.08 * len(members)),
                                f"Funds returned to {target} after {len(members)} hops ({chain})",
                            )
                        )
                        return found
                    previous = best_seen.get(t.dst)
                    if previous is None or t.minute < previous:
                        best_seen[t.dst] = t.minute
                        next_frontier.append((t.dst, t.minute, (*path, t)))
            frontier = next_frontier[:2000]
            if not frontier:
                break
    return found


def scatter_gather(edges: list[Tx], seed: Tx) -> list[Candidate]:
    """One source splits through intermediaries that each pay one destination."""
    clean = _non_self(edges)
    out_of: dict[str, list[Tx]] = defaultdict(list)
    into: dict[str, list[Tx]] = defaultdict(list)
    for t in clean:
        out_of[t.src].append(t)
        into[t.dst].append(t)
    pairs = {(seed.src, m.dst) for m in out_of.get(seed.src, [])} | {
        (s.src, seed.dst) for s in into.get(seed.dst, [])
    }
    pairs |= {(a.src, b.dst) for a in into.get(seed.src, []) for b in out_of.get(seed.src, [])}
    pairs |= {(a.src, b.dst) for a in into.get(seed.dst, []) for b in out_of.get(seed.dst, [])}
    candidates = []
    for source, sink in pairs:
        if source == sink:
            continue
        first = {t.dst: t for t in out_of.get(source, [])}
        second = {t.src: t for t in into.get(sink, [])}
        middle = [
            m
            for m in first
            if m in second and first[m].minute <= second[m].minute and m not in (source, sink)
        ]
        if len(middle) >= 2:
            members = [first[m] for m in middle] + [second[m] for m in middle]
            candidates.append(
                Candidate(
                    "SCATTER-GATHER",
                    tuple(t.tx_id for t in members),
                    (source, *sorted(middle), sink),
                    min(1.0, 0.45 + 0.1 * len(middle)),
                    f"{source} split funds through {len(middle)} intermediaries into {sink}",
                )
            )
    return sorted(candidates, key=lambda c: -c.strength)[:3]


def gather_scatter(edges: list[Tx], focus: set[str]) -> list[Candidate]:
    """Funds collect in one hub, then are paid out to several accounts."""
    candidates = []
    clean = _non_self(edges)
    for hub in focus:
        inflow = [t for t in clean if t.dst == hub]
        outflow = [t for t in clean if t.src == hub]
        if len({t.src for t in inflow}) < 2 or len({t.dst for t in outflow}) < 2 or not inflow:
            continue
        first_in = min(t.minute for t in inflow)
        last_in = max(t.minute for t in inflow)
        later = [t for t in outflow if t.minute >= first_in]
        if len({t.dst for t in later}) < 2:
            continue
        # The more of the outflow that follows the last inflow, the cleaner the two phases are.
        ordered = sum(t.minute >= last_in for t in later) / len(later)
        size = min(1.0, (len(inflow) + len(later)) / 10)
        candidates.append(
            Candidate(
                "GATHER-SCATTER",
                tuple(t.tx_id for t in (*inflow, *later)),
                (hub, *sorted({t.src for t in inflow}), *sorted({t.dst for t in later})),
                (0.35 + 0.35 * ordered) * (0.5 + 0.5 * size),
                f"{hub} gathered from {len({t.src for t in inflow})} accounts, "
                f"then paid {len({t.dst for t in later})}",
            )
        )
    return candidates


def _biclique_from(
    clean: list[Tx], seed_src: str, seed_dst: str
) -> tuple[set[str], set[str], list[Tx]] | None:
    """Grow a dense two-sided block around the edge ``seed_src -> seed_dst``."""
    targets_of: dict[str, set[str]] = defaultdict(set)
    sources_of: dict[str, set[str]] = defaultdict(set)
    for t in clean:
        targets_of[t.src].add(t.dst)
        sources_of[t.dst].add(t.src)
    right = set(targets_of.get(seed_src, set()))
    left = {
        s
        for s in sources_of.get(seed_dst, set())
        if len(targets_of.get(s, set()) & right) >= 2 or s == seed_src
    }
    if len(left) < 2:
        return None
    right = {d for d in right if len(sources_of.get(d, set()) & left) >= 2}
    if len(right) < 2 or left & right:
        return None
    members = [t for t in clean if t.src in left and t.dst in right]
    return left, right, members


def bipartite(edges: list[Tx], seed: Tx) -> list[Candidate]:
    clean = _non_self(edges)
    block = _biclique_from(clean, seed.src, seed.dst)
    if block is None:
        return []
    left, right, members = block
    density = len({(t.src, t.dst) for t in members}) / (len(left) * len(right))
    return [
        Candidate(
            "BIPARTITE",
            tuple(t.tx_id for t in members),
            (*sorted(left), *sorted(right)),
            0.4 + 0.5 * density,
            f"{len(left)} accounts paying {len(right)} accounts at density {density:.2f}",
        )
    ]


def stack(edges: list[Tx], seed: Tx) -> list[Candidate]:
    """Two bipartite layers joined through a shared middle group."""
    clean = _non_self(edges)
    block = _biclique_from(clean, seed.src, seed.dst)
    if block is None:
        return []
    left, right, members = block
    downstream = [t for t in clean if t.src in right and t.dst not in left | right]
    upstream = [t for t in clean if t.dst in left and t.src not in left | right]
    layer: list[Tx] = []
    for rows, key in ((downstream, "dst"), (upstream, "src")):
        ends: dict[str, set[str]] = defaultdict(set)
        for t in rows:
            ends[getattr(t, key)].add(t.src if key == "dst" else t.dst)
        dense = {end for end, peers in ends.items() if len(peers) >= 2}
        kept = [t for t in rows if getattr(t, key) in dense]
        if len(dense) >= 2 and len(kept) > len(layer):
            layer = kept
    if not layer:
        return []
    all_members = [*members, *layer]
    accounts = sorted({t.src for t in all_members} | {t.dst for t in all_members})
    return [
        Candidate(
            "STACK",
            tuple(t.tx_id for t in all_members),
            tuple(accounts),
            min(1.0, 0.92 + 0.005 * len(all_members)),
            f"Layered block of {len(accounts)} accounts across {len(all_members)} payments",
        )
    ]


def random_chain(edges: list[Tx], seed: Tx) -> list[Candidate]:
    """A chain of hops that passes funds along without a recognisable shape."""
    clean = _non_self(edges)
    out_of: dict[str, list[Tx]] = defaultdict(list)
    into: dict[str, list[Tx]] = defaultdict(list)
    for t in clean:
        out_of[t.src].append(t)
        into[t.dst].append(t)
    path = [seed]
    seen = {seed.src, seed.dst}
    current = seed
    while len(path) < 12:
        options = [
            t
            for t in out_of.get(current.dst, [])
            if t.minute >= current.minute and t.dst not in seen and t.usd >= 0.5 * current.usd
        ]
        if not options:
            break
        current = min(options, key=lambda t: t.minute)
        path.append(current)
        seen.add(current.dst)
    current = seed
    while len(path) < 12:
        options = [
            t
            for t in into.get(current.src, [])
            if t.minute <= current.minute and t.src not in seen and t.usd >= 0.5 * current.usd
        ]
        if not options:
            break
        current = max(options, key=lambda t: t.minute)
        path.insert(0, current)
        seen.add(current.src)
    if len(path) < 3:
        return []
    accounts = tuple(dict.fromkeys([path[0].src, *(t.dst for t in path)]))
    return [
        Candidate(
            "RANDOM",
            tuple(t.tx_id for t in path),
            accounts,
            min(0.55, 0.2 + 0.05 * len(path)),
            f"Funds passed along {len(path)} hops in sequence",
        )
    ]


def _unusual_edges(edges: list[Tx], seed: Tx, max_repeats: int = 2) -> list[Tx]:
    """Drop routine activity, so that a busy account's normal payments don't swamp the shape.

    Recurring relationships and other payment formats are background around most accounts, and
    they create false cycles and fans. This uses only observable behaviour, not labels.
    """
    repeats: dict[tuple[str, str], int] = defaultdict(int)
    for t in edges:
        repeats[(t.src, t.dst)] += 1
    return [
        t
        for t in edges
        if t.tx_id == seed.tx_id
        or (t.payment_format == seed.payment_format and repeats[(t.src, t.dst)] <= max_repeats)
    ]


def detect(edges: list[Tx], seed: Tx) -> list[Candidate]:
    """All candidates around ``seed``, strongest first. An empty list means no recognised shape."""
    if seed.tx_id not in {t.tx_id for t in edges}:
        edges = [*edges, seed]
    edges = _unusual_edges(edges, seed)
    focus = {seed.src, seed.dst}
    found = [
        *cycles(edges, seed),
        *stack(edges, seed),
        *scatter_gather(edges, seed),
        *bipartite(edges, seed),
        *gather_scatter(edges, focus),
        *fan_out(edges, focus),
        *fan_in(edges, focus),
        *random_chain(edges, seed),
    ]
    # Only keep candidates the seed is actually part of, or whose accounts include it.
    relevant = [
        c
        for c in found
        if seed.tx_id in c.tx_ids or seed.src in c.accounts or seed.dst in c.accounts
    ]
    return sorted(relevant, key=lambda c: (-c.strength, c.pattern))


def rule_baseline_label(edges: list[Tx], seed: Tx) -> str:
    """The detector-only answer: the strongest pattern, or ``NONE``."""
    candidates = detect(edges, seed)
    return candidates[0].pattern if candidates else "NONE"
