"""Feature consumer: reuses ``FeatureEngine`` unchanged, and recovers exactly after a crash.

Delivery is at-least-once with effectively-once results. State and input offset are saved together,
atomically, so a restart resumes from a consistent point. Anything emitted after that point is
emitted again with identical content, and downstream readers upsert on ``tx_id``.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from unravel.features.engine import FeatureEngine
from unravel.streaming.messages import decode_event, encode_vector


class Sink(Protocol):
    def features(self, tx_id: str, payload: bytes) -> None: ...

    def dead_letter(self, offset: int, payload: bytes, reason: str) -> None: ...


@dataclass
class ConsumerState:
    engine: FeatureEngine
    next_offset: int = 0
    seen: set[str] | None = None

    def __post_init__(self) -> None:
        if self.seen is None:
            self.seen = set()


class SnapshotStore:
    """Atomic snapshots: write to a temporary file, then rename over the previous one."""

    def __init__(self, path: Path, before_save: Callable[[], object] | None = None) -> None:
        self.path = path
        self.before_save = before_save

    def load(self) -> ConsumerState | None:
        if not self.path.exists():
            return None
        with self.path.open("rb") as handle:
            blob = pickle.load(handle)
        return ConsumerState(FeatureEngine.restore(blob["engine"]), blob["offset"], blob["seen"])

    def save(self, state: ConsumerState) -> None:
        if self.before_save is not None:
            self.before_save()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(
                {
                    "engine": state.engine.snapshot(),
                    "offset": state.next_offset,
                    "seen": state.seen,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)


class FeatureProcessor:
    """Turn raw messages into feature messages. It knows nothing about Kafka."""

    def __init__(
        self,
        state: ConsumerState,
        sink: Sink,
        snapshots: SnapshotStore | None = None,
        snapshot_every: int = 50_000,
        on_processed: Callable[[int], None] | None = None,
    ) -> None:
        self.state = state
        self.sink = sink
        self.snapshots = snapshots
        self.snapshot_every = snapshot_every
        self.on_processed = on_processed
        self.since_snapshot = 0

    def handle(self, offset: int, payload: bytes) -> None:
        state = self.state
        if offset < state.next_offset:
            return  # Redelivery of something already reflected in restored state.
        seen = state.seen if state.seen is not None else set()
        try:
            event = decode_event(payload)
            if event.tx_id not in seen:
                vector = state.engine.process(event)
                seen.add(event.tx_id)
                self.sink.features(event.tx_id, encode_vector(event.tx_id, event.minute, vector))
        except ValidationError as error:
            self.sink.dead_letter(offset, payload, f"schema: {error.error_count()} errors")
        except ValueError as error:
            self.sink.dead_letter(offset, payload, str(error)[:300])
        state.next_offset = offset + 1
        self.since_snapshot += 1
        if self.on_processed is not None:
            self.on_processed(offset)
        if self.snapshots is not None and self.since_snapshot >= self.snapshot_every:
            self.checkpoint()

    def checkpoint(self) -> None:
        if self.snapshots is not None:
            self.snapshots.save(self.state)
        self.since_snapshot = 0

    def run(self, messages: Iterable[tuple[int, bytes]]) -> None:
        for offset, payload in messages:
            self.handle(offset, payload)


def load_or_create_state(
    fx_rates: dict[str, float], snapshots: SnapshotStore | None
) -> ConsumerState:
    if snapshots is not None:
        restored = snapshots.load()
        if restored is not None:
            return restored
    return ConsumerState(FeatureEngine(fx_rates))
