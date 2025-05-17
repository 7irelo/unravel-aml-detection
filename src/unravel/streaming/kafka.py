"""Thin Kafka adapters. All behaviour worth testing lives in ``consumer.py`` and ``scorer.py``.

``tx.raw`` uses a single partition. A transaction touches two accounts that would hash to different
partitions, and the parity test needs a strict total order, so ordering wins over parallelism here.
See ``docs/adr/0005-single-partition.md`` for the partitioned design this would become.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition
from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

from unravel.data.schema import TX_COLUMNS
from unravel.features.engine import TxEvent
from unravel.streaming.consumer import FeatureProcessor, SnapshotStore, load_or_create_state
from unravel.streaming.messages import (
    ALL_TOPICS,
    TOPIC_DLQ,
    TOPIC_FEATURES,
    TOPIC_RAW,
    encode_event,
)


def ensure_topics(bootstrap: str, *, recreate: bool = False) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    if recreate:
        for future in admin.delete_topics(list(ALL_TOPICS), operation_timeout=20).values():
            with contextlib.suppress(Exception):
                future.result()
        time.sleep(1.0)
    topics = [NewTopic(name, num_partitions=1, replication_factor=1) for name in ALL_TOPICS]
    for future in admin.create_topics(topics, operation_timeout=20).values():
        try:
            future.result()
        except Exception as error:
            if "TOPIC_ALREADY_EXISTS" not in str(error):
                raise


def make_producer(bootstrap: str) -> Producer:
    return Producer(
        {
            "bootstrap.servers": bootstrap,
            "enable.idempotence": True,
            "acks": "all",
            "linger.ms": 20,
            "compression.type": "lz4",
            "queue.buffering.max.messages": 500_000,
        }
    )


def replay(
    transactions: Path,
    bootstrap: str,
    *,
    limit: int | None = None,
    minutes_per_second: float = 0.0,
    log_every: int = 200_000,
) -> int:
    """Publish in canonical order. Pace by event time, or as fast as possible when speed is 0."""
    frame = pl.read_parquet(transactions).select(TX_COLUMNS)
    if limit is not None:
        frame = frame.head(limit)
    producer = make_producer(bootstrap)
    started = time.perf_counter()
    first_minute: int | None = None
    sent = 0
    for row in frame.iter_rows():
        event = TxEvent(*row)
        if minutes_per_second > 0:
            first_minute = event.minute if first_minute is None else first_minute
            due = (event.minute - first_minute) / minutes_per_second
            delay = due - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
        while True:
            try:
                producer.produce(TOPIC_RAW, key=event.tx_id, value=encode_event(event))
                break
            except BufferError:
                producer.poll(0.05)
        producer.poll(0)
        sent += 1
        if sent % log_every == 0:
            print(f"replayed {sent:,}", flush=True)
    producer.flush()
    return sent


class KafkaSink:
    def __init__(self, producer: Producer) -> None:
        self.producer = producer

    def _send(self, topic: str, key: str, payload: bytes) -> None:
        while True:
            try:
                self.producer.produce(topic, key=key, value=payload)
                self.producer.poll(0)
                return
            except BufferError:
                self.producer.poll(0.05)

    def features(self, tx_id: str, payload: bytes) -> None:
        self._send(TOPIC_FEATURES, tx_id, payload)

    def dead_letter(self, offset: int, payload: bytes, reason: str) -> None:
        body = json.dumps(
            {"offset": offset, "reason": reason, "payload": payload.decode("replace")}
        )
        self._send(TOPIC_DLQ, str(offset), body.encode())


def read_topic(
    bootstrap: str,
    topic: str,
    *,
    start_offset: int = 0,
    idle_timeout: float | None = None,
    group: str = "unravel-reader",
    idle_ticks: bool = False,
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(offset, value)``. Offsets are tracked by our snapshots, not by consumer groups.

    With ``idle_ticks``, an empty poll yields ``(-1, b"")`` so callers can flush partial batches.
    """
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "fetch.min.bytes": 65536,
        }
    )
    consumer.assign([TopicPartition(topic, 0, start_offset)])
    idle_since = time.perf_counter()
    try:
        while True:
            message = consumer.poll(0.5)
            if message is None:
                if idle_timeout is not None and time.perf_counter() - idle_since > idle_timeout:
                    return
                if idle_ticks:
                    yield -1, b""
                continue
            error = message.error()
            if error is not None:
                if error.code() == KafkaError._PARTITION_EOF:
                    continue
                raise RuntimeError(str(error))
            idle_since = time.perf_counter()
            yield message.offset() or 0, message.value() or b""
    finally:
        consumer.close()


def run_feature_consumer(
    bootstrap: str,
    fx_rates: dict[str, float],
    snapshot_path: Path,
    *,
    snapshot_every: int = 50_000,
    idle_timeout: float | None = None,
) -> int:
    """Run until idle (or forever). Restarts resume from the last snapshot."""
    producer = make_producer(bootstrap)
    # Features must be durable on the broker before a snapshot claims they are done.
    snapshots = SnapshotStore(snapshot_path, before_save=producer.flush)
    state = load_or_create_state(fx_rates, snapshots)
    processor = FeatureProcessor(state, KafkaSink(producer), snapshots, snapshot_every)

    print(f"starting from offset {state.next_offset:,}", flush=True)
    processor.run(
        read_topic(bootstrap, TOPIC_RAW, start_offset=state.next_offset, idle_timeout=idle_timeout)
    )
    processor.checkpoint()
    return state.next_offset
