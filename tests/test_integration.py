"""Against real services. Run with `docker compose -f deploy/compose.yaml --profile core up -d
redpanda postgres`, then `pytest -m integration`."""

from __future__ import annotations

import os

import psycopg
import pytest

from unravel.config import DatasetPaths
from unravel.features.backfill import backfill, load_features
from unravel.features.engine import FEATURE_NAMES
from unravel.features.fx import load_fx
from unravel.streaming.kafka import ensure_topics, read_topic, replay, run_feature_consumer
from unravel.streaming.messages import TOPIC_FEATURES
from unravel.streaming.parity import compare

BOOTSTRAP = os.environ.get("UNRAVEL_KAFKA_BOOTSTRAP", "localhost:19092")
OWNER = "postgresql://unravel:unravel_local@localhost:55432/unravel"
AGENT = "postgresql://unravel_agent:agent_read_only_local@localhost:55432/unravel"

pytestmark = pytest.mark.integration


def test_redpanda_round_trip_matches_offline(fixture_paths: DatasetPaths, tmp_path) -> None:  # type: ignore[no-untyped-def]
    backfill(fixture_paths, chunk_size=2000)
    stored = load_features(fixture_paths.features)
    ensure_topics(BOOTSTRAP, recreate=True)
    sent = replay(fixture_paths.transactions, BOOTSTRAP)
    assert sent == stored.height
    end = run_feature_consumer(
        BOOTSTRAP, load_fx(fixture_paths.fx), tmp_path / "state.pkl", idle_timeout=5.0
    )
    assert end == sent
    messages = (p for _, p in read_topic(BOOTSTRAP, TOPIC_FEATURES, idle_timeout=5.0, group="t"))
    report = compare(stored["tx_id"].to_list(), stored.select(FEATURE_NAMES).to_numpy(), messages)
    assert report.passed, report


def test_raw_topic_carries_no_labels(fixture_paths: DatasetPaths) -> None:
    from unravel.streaming.messages import TOPIC_RAW

    first = next(read_topic(BOOTSTRAP, TOPIC_RAW, idle_timeout=5.0, group="labels"))[1]
    assert b"laundering" not in first
    assert b"pattern" not in first


def test_agent_role_cannot_read_labels_or_write() -> None:
    """The label boundary is enforced by Postgres grants, not by convention."""
    with psycopg.connect(AGENT, autocommit=True) as agent:
        agent.execute("SELECT count(*) FROM transactions")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            agent.execute("SELECT * FROM ground_truth.labels")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            agent.execute("DELETE FROM transactions")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            agent.execute("SELECT * FROM cases")


def test_a_case_cannot_be_marked_as_filed() -> None:
    with psycopg.connect(OWNER, autocommit=True) as owner:
        owner.execute(
            "INSERT INTO alerts (alert_id, tx_id, minute, raw_score, probability, band, "
            "model_version) VALUES ('A-x', 'x', 0, 0, 0, 'watch', 'm') ON CONFLICT DO NOTHING"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            owner.execute(
                "INSERT INTO cases (alert_id, status, model) VALUES ('A-x', 'filed', 'm')"
            )
        owner.execute("DELETE FROM alerts WHERE alert_id = 'A-x'")


def test_idle_ticks_let_a_reader_flush_partial_batches(fixture_paths: DatasetPaths) -> None:
    """An empty poll must surface, or a scorer holding a partial batch would wait forever."""
    from itertools import islice

    from unravel.streaming.messages import TOPIC_DLQ

    ensure_topics(BOOTSTRAP)
    ticks = list(
        islice(
            read_topic(BOOTSTRAP, TOPIC_DLQ, idle_ticks=True, group="ticks", idle_timeout=5.0), 2
        )
    )
    assert ticks == [(-1, b""), (-1, b"")]
