"""The scorer service, and loading reference data into Postgres."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psycopg
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from psycopg.types.json import Jsonb

from unravel.monitoring.drift import DriftMonitor
from unravel.streaming.kafka import make_producer, read_topic
from unravel.streaming.messages import TOPIC_ALERTS, TOPIC_FEATURES, TOPIC_SCORES
from unravel.streaming.scorer import ScoredTransaction, Scorer

SCORED = Counter("unravel_scored_total", "Transactions scored")
ALERTS = Counter("unravel_alerts_total", "Alerts admitted", ["band"])
BATCH_SECONDS = Histogram("unravel_score_batch_seconds", "Time to score one batch")
THRESHOLD = Gauge("unravel_alert_threshold", "Current admission threshold")
EVENT_MINUTE = Gauge("unravel_event_minute", "Latest event time scored")
FEATURE_PSI = Gauge("unravel_feature_psi", "PSI against the training reference", ["feature"])
FEATURES_DRIFTING = Gauge("unravel_features_drifting", "Features above their alert level")
DRIFT_WINDOW = 20_000


def load_transactions(database_url: str, transactions: Path, fx: dict[str, float]) -> int:
    """Bulk-load transactions for the investigator. Labels are not part of this."""
    frame = pl.read_parquet(transactions).select(
        "tx_id",
        "minute",
        "src",
        "dst",
        "amount_paid",
        "payment_currency",
        "amount_received",
        "receiving_currency",
        "payment_format",
    )
    statement = (
        "COPY transactions (tx_id, minute, src, dst, amount_paid, payment_currency, "
        "amount_received, receiving_currency, payment_format, usd) FROM STDIN"
    )
    with (
        psycopg.connect(database_url) as connection,
        connection.cursor() as cursor,
        cursor.copy(statement) as copy,
    ):
        for row in frame.iter_rows():
            copy.write_row((*row, row[4] * fx.get(row[5], 0.0)))
    return frame.height


def _persist(
    connection: psycopg.Connection[Any], results: list[ScoredTransaction], version: str
) -> None:
    """Upsert on ``tx_id``, so redelivered features can't create duplicates."""
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO scores (tx_id, minute, raw_score, probability, model_version)
            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (tx_id) DO NOTHING
            """,
            [(r.tx_id, r.minute, r.raw_score, r.probability, version) for r in results],
        )
        for result in results:
            if result.alert is None:
                continue
            cursor.execute(
                """
                INSERT INTO alerts (alert_id, tx_id, minute, raw_score, probability, band,
                                    model_version, reasons)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (alert_id) DO NOTHING
                """,
                (
                    f"A-{result.tx_id}",
                    result.tx_id,
                    result.minute,
                    result.raw_score,
                    result.probability,
                    result.alert.band,
                    version,
                    Jsonb(result.reasons or []),
                ),
            )


def run_scorer(
    bootstrap: str,
    database_url: str,
    model_directory: Path,
    *,
    capacity: int = 100,
    batch_size: int = 512,
    idle_timeout: float | None = None,
    metrics_port: int | None = None,
) -> int:
    scorer = Scorer.from_directory(model_directory, capacity)
    version = f"{scorer.model.config.arm}-{scorer.model.model_hash()}"
    producer = make_producer(bootstrap)
    connection = psycopg.connect(database_url, autocommit=False)
    if metrics_port:
        start_http_server(metrics_port)

    pending: list[bytes] = []
    total = 0
    reference = model_directory / "drift_reference.json"
    monitor = DriftMonitor.load(reference) if reference.exists() else None
    window: list[np.ndarray] = []
    window_rows = 0

    def update_drift(vectors: np.ndarray) -> None:
        nonlocal window, window_rows
        if monitor is None:
            return
        window.append(vectors)
        window_rows += len(vectors)
        if window_rows < DRIFT_WINDOW:
            return
        matrix = np.concatenate(window)
        window, window_rows = [], 0
        for name, value in monitor.check(matrix).items():
            FEATURE_PSI.labels(name).set(value)
        FEATURES_DRIFTING.set(len(monitor.alerts(matrix)))

    def flush() -> None:
        nonlocal total, pending
        if not pending:
            return
        started = time.perf_counter()
        results = scorer.score_batch(pending)
        BATCH_SECONDS.observe(time.perf_counter() - started)
        update_drift(scorer.last_vectors)
        _persist(connection, results, version)
        connection.commit()
        for result in results:
            producer.produce(
                TOPIC_SCORES,
                key=result.tx_id,
                value=json.dumps(
                    {"tx_id": result.tx_id, "p": result.probability, "raw": result.raw_score}
                ).encode(),
            )
            if result.alert is not None:
                ALERTS.labels(result.alert.band).inc()
                producer.produce(
                    TOPIC_ALERTS,
                    key=result.tx_id,
                    value=json.dumps(
                        {
                            "tx_id": result.tx_id,
                            "band": result.alert.band,
                            "reasons": result.reasons,
                        }
                    ).encode(),
                )
            producer.poll(0)
        SCORED.inc(len(results))
        EVENT_MINUTE.set(results[-1].minute)
        if scorer.queue.threshold != float("inf"):
            THRESHOLD.set(scorer.queue.threshold)
        total += len(results)
        pending = []

    for offset, payload in read_topic(
        bootstrap,
        TOPIC_FEATURES,
        idle_timeout=idle_timeout,
        group="unravel-scorer",
        idle_ticks=True,
    ):
        if offset < 0:  # A quiet stream mustn't leave a partial batch unscored.
            flush()
            continue
        pending.append(payload)
        if len(pending) >= batch_size:
            flush()
    flush()
    producer.flush()
    connection.close()
    return total
