"""Online capacity: a live system can't sort a day that hasn't finished.

An alert is raised when a score clears a threshold derived from recent days, and the day's queue
stops growing at analyst capacity. Offline "top-k per day" is an oracle; ``online_regret`` in
``eval.metrics`` measures how much this policy gives up against it.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

MINUTES_PER_DAY = 1440


def risk_band(probability: float) -> str:
    if probability >= 0.5:
        return "critical"
    if probability >= 0.2:
        return "high"
    if probability >= 0.05:
        return "elevated"
    return "watch"


@dataclass(order=True)
class QueuedAlert:
    priority: float
    tx_id: str = field(compare=False)
    day: int = field(compare=False)
    raw_score: float = field(compare=False)
    probability: float = field(compare=False)
    band: str = field(compare=False)


class CapacityQueue:
    """Admit alerts within a daily capacity, using a threshold learned from completed days."""

    def __init__(
        self,
        capacity: int = 100,
        history_days: int = 3,
        initial_threshold: float | None = None,
    ) -> None:
        self.capacity = capacity
        self.history: deque[float] = deque(maxlen=history_days)
        self.initial_threshold = initial_threshold
        self.day: int | None = None
        self.admitted_today = 0
        self.day_top: list[float] = []  # min-heap of today's top scores
        self.queue: list[QueuedAlert] = []  # max-heap by raw score, via negated priority

    @property
    def threshold(self) -> float:
        if self.history:
            return sum(self.history) / len(self.history)
        return self.initial_threshold if self.initial_threshold is not None else float("inf")

    def _roll_day(self, day: int) -> None:
        if self.day is not None and self.day_top:
            self.history.append(self.day_top[0])
        self.day = day
        self.admitted_today = 0
        self.day_top = []

    def offer(
        self, *, tx_id: str, minute: int, raw_score: float, probability: float
    ) -> QueuedAlert | None:
        """Consider one scored transaction. Returns the alert if it was admitted."""
        day = minute // MINUTES_PER_DAY
        if day != self.day:
            self._roll_day(day)
        # Whether or not it's admitted, the score informs tomorrow's threshold.
        if len(self.day_top) < self.capacity:
            heapq.heappush(self.day_top, raw_score)
        elif raw_score > self.day_top[0]:
            heapq.heapreplace(self.day_top, raw_score)

        if raw_score < self.threshold or self.admitted_today >= self.capacity:
            return None
        self.admitted_today += 1
        alert = QueuedAlert(-raw_score, tx_id, day, raw_score, probability, risk_band(probability))
        heapq.heappush(self.queue, alert)
        return alert

    def next_for_analyst(self) -> QueuedAlert | None:
        """Highest-priority alert first. An analyst pulls from here rather than in arrival order."""
        return heapq.heappop(self.queue) if self.queue else None
