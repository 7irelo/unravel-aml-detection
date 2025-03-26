"""Seeded synthetic data in the raw AMLworld format, with one or more of each named pattern.

It serves CI, unit tests and the no-credentials demo. It is not a substitute for AMLworld: it exists
so the pipeline can run end to end, and results on it say nothing about real detection quality.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

from unravel.config import DatasetPaths

START = datetime(2022, 9, 1, tzinfo=UTC)
CURRENCIES = ["US Dollar", "Euro", "Yuan", "Rupee"]
RATES = {"US Dollar": 1.0, "Euro": 1.1, "Yuan": 0.15, "Rupee": 0.012}
BACKGROUND_FORMATS = ["Cheque", "Credit Card", "Cash", "Wire", "ACH", "Reinvestment"]
HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)


class _Writer:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.rows: list[tuple[datetime, str]] = []
        self.pattern_blocks: list[str] = []

    def row(
        self,
        when: datetime,
        src: tuple[str, str],
        dst: tuple[str, str],
        amount: float,
        fmt: str,
        label: int,
        currency: str = "US Dollar",
    ) -> str:
        received_currency = currency
        received = amount
        if label == 0 and self.rng.random() < 0.03:
            received_currency = self.rng.choice(CURRENCIES)
            received = amount * RATES[currency] / RATES[received_currency]
        line = (
            f"{when:%Y/%m/%d %H:%M},{src[0]},{src[1]},{dst[0]},{dst[1]},"
            f"{received:.2f},{received_currency},{amount:.2f},{currency},{fmt},{label}"
        )
        self.rows.append((when, line))
        return line


def _account(rng: random.Random, prefix: str) -> tuple[str, str]:
    return f"{rng.randint(1, 30):03d}", f"{prefix}{rng.randrange(16**7):07X}"


def _shape(rng: random.Random, pattern: str) -> list[tuple[int, int]]:
    """Return directed edges between integer node IDs for one pattern instance."""
    if pattern == "FAN-OUT":
        return [(0, i) for i in range(1, rng.randint(5, 9))]
    if pattern == "FAN-IN":
        return [(i, 0) for i in range(1, rng.randint(5, 9))]
    if pattern == "CYCLE":
        n = rng.randint(3, 7)
        return [(i, (i + 1) % n) for i in range(n)]
    if pattern == "SCATTER-GATHER":
        mids = rng.randint(3, 6)
        return [(0, i + 2) for i in range(mids)] + [(i + 2, 1) for i in range(mids)]
    if pattern == "GATHER-SCATTER":
        sides = rng.randint(3, 5)
        return [(i + 1, 0) for i in range(sides)] + [(0, i + 1 + sides) for i in range(sides)]
    if pattern == "BIPARTITE":
        left, right = rng.randint(2, 4), rng.randint(2, 4)
        return [(i, left + j) for i in range(left) for j in range(right)]
    if pattern == "STACK":
        a, b = 2, 2
        first = [(i, a + j) for i in range(a) for j in range(b)]
        second = [(a + j, a + b + k) for j in range(b) for k in range(2)]
        return first + second
    # RANDOM: a loose chain with an occasional branch.
    n = rng.randint(4, 7)
    return [(i, i + 1) for i in range(n - 1)] + [(0, n)]


FIXTURE_PATTERNS = (
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "BIPARTITE",
    "STACK",
    "RANDOM",
)


def generate_fixture(
    paths: DatasetPaths,
    *,
    seed: int = 7,
    days: int = 10,
    accounts: int = 400,
    background_per_day: int = 800,
    instances_per_pattern: int = 6,
    natural_laundering: int = 40,
) -> dict[str, int]:
    """Write ``<name>_Trans.csv`` and ``<name>_Patterns.txt``."""
    rng = random.Random(seed)
    writer = _Writer(rng)
    population = [_account(rng, "A") for _ in range(accounts)]

    for day in range(days):
        for _ in range(background_per_day):
            when = START + timedelta(days=day, minutes=rng.randrange(1440))
            src, dst = rng.sample(population, 2)
            amount = round(rng.lognormvariate(6.0, 1.5), 2)
            currency = rng.choices(CURRENCIES, weights=[6, 2, 1, 1])[0]
            writer.row(when, src, dst, amount, rng.choice(BACKGROUND_FORMATS), 0, currency)

    for pattern in FIXTURE_PATTERNS:
        for _ in range(instances_per_pattern):
            start = START + timedelta(days=rng.randrange(days - 1), minutes=rng.randrange(1440))
            nodes: dict[int, tuple[str, str]] = {}
            edges = _shape(rng, pattern)
            lines: list[str] = []
            base = rng.lognormvariate(8.0, 1.0)
            offset = 0
            for a, b in edges:
                src = nodes.setdefault(a, _account(rng, "L"))
                dst = nodes.setdefault(b, _account(rng, "L"))
                offset += rng.randint(5, 180)
                when = min(
                    start + timedelta(minutes=offset), START + timedelta(days=days, minutes=-1)
                )
                amount = base * rng.uniform(0.85, 1.05)
                lines.append(writer.row(when, src, dst, amount, "ACH", 1))
            lines.sort()
            detail = f"{pattern}:  fixture" if pattern != "CYCLE" else f"{pattern}:  Max 10 hops"
            writer.pattern_blocks.append(
                f"BEGIN LAUNDERING ATTEMPT - {detail}\n"
                + "\n".join(lines)
                + f"\nEND LAUNDERING ATTEMPT - {pattern}\n"
            )

    # "Natural" laundering: labelled positive but belonging to no named pattern.
    for _ in range(natural_laundering):
        when = START + timedelta(days=rng.randrange(days), minutes=rng.randrange(1440))
        src, dst = rng.sample(population, 2)
        writer.row(when, src, dst, round(rng.lognormvariate(7.0, 1.0), 2), "Wire", 1)

    # The real file is roughly but not exactly time-ordered; shuffle so ingest must sort.
    rng.shuffle(writer.rows)
    paths.raw_trans.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_trans.write_text(
        HEADER + "\n" + "\n".join(line for _, line in writer.rows) + "\n", encoding="utf-8"
    )
    Path(paths.raw_patterns).write_text("\n".join(writer.pattern_blocks) + "\n", encoding="utf-8")
    return {"rows": len(writer.rows), "patterns": len(writer.pattern_blocks)}
