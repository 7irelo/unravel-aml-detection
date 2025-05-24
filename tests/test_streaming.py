"""The parity suite, using an in-memory log. The same runtime runs against Redpanda in integration.

These are the project's most important tests, so the suite includes a deliberately skewed engine
that must make it fail.
"""

from __future__ import annotations

import json
import random

import numpy as np
import polars as pl
import pytest

from unravel.alerts.queue import CapacityQueue, risk_band
from unravel.config import DatasetPaths
from unravel.data.schema import TX_COLUMNS
from unravel.features.backfill import backfill, iter_events, load_features
from unravel.features.engine import FEATURE_NAMES, FeatureEngine, TxEvent
from unravel.features.fx import load_fx
from unravel.monitoring.drift import DriftMonitor, build_reference, calibrate_alert_level, psi
from unravel.streaming.consumer import (
    ConsumerState,
    FeatureProcessor,
    SnapshotStore,
    load_or_create_state,
)
from unravel.streaming.messages import decode_event, decode_vector, encode_event, encode_vector
from unravel.streaming.parity import compare


class MemorySink:
    def __init__(self) -> None:
        self.messages: list[bytes] = []
        self.dead: list[tuple[int, str]] = []

    def features(self, tx_id: str, payload: bytes) -> None:
        self.messages.append(payload)

    def dead_letter(self, offset: int, payload: bytes, reason: str) -> None:
        self.dead.append((offset, reason))


@pytest.fixture(scope="module")
def setup(fixture_paths: DatasetPaths):  # type: ignore[no-untyped-def]
    backfill(fixture_paths, chunk_size=2000)
    stored = load_features(fixture_paths.features)
    transactions = pl.read_parquet(fixture_paths.transactions).select(TX_COLUMNS)
    log = [(i, encode_event(e)) for i, e in enumerate(iter_events(transactions))]
    return {
        "ids": stored["tx_id"].to_list(),
        "matrix": stored.select(FEATURE_NAMES).to_numpy(),
        "log": log,
        "fx": load_fx(fixture_paths.fx),
    }


def _run_clean(setup) -> MemorySink:  # type: ignore[no-untyped-def]
    sink = MemorySink()
    FeatureProcessor(ConsumerState(FeatureEngine(setup["fx"])), sink).run(setup["log"])
    return sink


def test_wire_round_trip_is_exact() -> None:
    event = TxEvent("T1", 3, 60, "a", "b", "001", "002", 0.1 + 0.2, "Euro", 1e-7, "Yen", "ACH")
    assert decode_event(encode_event(event)) == event
    vector = [0.1 + 0.2, float("nan"), 1.7976931348623157e308, 5e-324, -0.0]
    tx_id, minute, out = decode_vector(encode_vector("T1", 60, vector))
    assert (tx_id, minute) == ("T1", 60)
    assert out[0] == vector[0]
    assert np.isnan(out[1])
    assert out[2:] == vector[2:]


def test_raw_schema_rejects_a_label_field() -> None:
    body = json.loads(encode_event(TxEvent("T", 0, 0, "a", "b", "1", "2", 1.0, "E", 1.0, "E", "A")))
    body["is_laundering"] = 1
    with pytest.raises(ValueError, match="is_laundering"):
        decode_event(json.dumps(body).encode())


def test_end_to_end_replay_matches_offline(setup) -> None:  # type: ignore[no-untyped-def]
    report = compare(setup["ids"], setup["matrix"], _run_clean(setup).messages)
    assert report.passed, report
    assert report.vectors_exact == report.offline_rows


def test_parity_check_can_fail_on_real_skew(setup) -> None:  # type: ignore[no-untyped-def]
    """Proof the test has teeth: a plausible bug, using wall-clock-like time, must be caught."""

    class SkewedEngine(FeatureEngine):
        def process(self, event: TxEvent) -> list[float]:
            return super().process(event._replace(minute=event.minute + (event.order_key % 2)))

    sink = MemorySink()
    FeatureProcessor(ConsumerState(SkewedEngine(setup["fx"])), sink).run(setup["log"])
    report = compare(setup["ids"], setup["matrix"], sink.messages)
    assert not report.passed
    assert report.vectors_within_tolerance < report.vectors_compared


@pytest.mark.parametrize(
    "mutation",
    ["drop_message", "change_value", "add_nan", "extra_id", "conflicting_duplicate"],
)
def test_parity_check_catches_each_kind_of_damage(setup, mutation: str) -> None:  # type: ignore[no-untyped-def]
    messages = list(_run_clean(setup).messages)
    tx_id, minute, vector = decode_vector(messages[100])
    if mutation == "drop_message":
        del messages[100]
    elif mutation == "change_value":
        vector[0] += 0.001
        messages[100] = encode_vector(tx_id, minute, vector)
    elif mutation == "add_nan":
        vector[0] = float("nan")
        messages[100] = encode_vector(tx_id, minute, vector)
    elif mutation == "extra_id":
        messages.append(encode_vector("Tnew", minute, vector))
    else:
        vector[0] += 1.0
        messages.append(encode_vector(tx_id, minute, vector))
    assert not compare(setup["ids"], setup["matrix"], messages).passed


@pytest.mark.parametrize("seed", range(6))
def test_crash_at_random_points_gives_identical_output(setup, tmp_path, seed: int) -> None:  # type: ignore[no-untyped-def]
    """Kill the consumer repeatedly. Offsets rewind to the snapshot, and output must still match."""
    rng = random.Random(seed)
    store = SnapshotStore(tmp_path / f"state-{seed}.pkl")
    sink = MemorySink()

    class CrashError(Exception):
        pass

    remaining = list(setup["log"])
    crashes = 0
    while True:
        state = load_or_create_state(setup["fx"], store)
        crash_at = state.next_offset + rng.randint(50, len(setup["log"]) // 2)

        def crash_check(offset: int, limit: int = crash_at) -> None:
            if offset == limit:
                raise CrashError

        processor = FeatureProcessor(
            state, sink, store, snapshot_every=500, on_processed=crash_check
        )
        try:
            processor.run(remaining[state.next_offset :])
            processor.checkpoint()
            break
        except CrashError:
            crashes += 1
    assert crashes >= 1
    report = compare(setup["ids"], setup["matrix"], sink.messages)
    assert report.passed, report
    assert report.online_messages >= report.offline_rows  # replays happened, and all agreed


def test_duplicates_in_the_log_change_nothing(setup) -> None:  # type: ignore[no-untyped-def]
    log: list[tuple[int, bytes]] = []
    for offset, payload in setup["log"]:
        log.append((len(log), payload))
        if offset % 50 == 0:
            log.append((len(log), payload))
    sink = MemorySink()
    FeatureProcessor(ConsumerState(FeatureEngine(setup["fx"])), sink).run(log)
    report = compare(setup["ids"], setup["matrix"], sink.messages)
    assert report.passed, report
    assert report.online_messages == report.offline_rows


def test_malformed_messages_go_to_dlq_without_stopping(setup) -> None:  # type: ignore[no-untyped-def]
    log = list(setup["log"][:50])
    log.insert(10, (10, b"{not json"))
    log.insert(11, (11, b'{"tx_id":"x"}'))
    log = [(i, payload) for i, (_, payload) in enumerate(log)]
    sink = MemorySink()
    FeatureProcessor(ConsumerState(FeatureEngine(setup["fx"])), sink).run(log)
    assert [offset for offset, _ in sink.dead] == [10, 11]
    assert len(sink.messages) == 50


def test_out_of_order_message_goes_to_dlq(setup) -> None:  # type: ignore[no-untyped-def]
    late = decode_event(setup["log"][0][1])._replace(tx_id="Tlate", minute=0)
    last = decode_event(setup["log"][30][1])
    assert last.minute > 0
    log = [*setup["log"][:31], (31, encode_event(late))]
    sink = MemorySink()
    FeatureProcessor(ConsumerState(FeatureEngine(setup["fx"])), sink).run(log)
    assert sink.dead
    assert "out-of-order" in sink.dead[0][1]


# -- capacity queue -------------------------------------------------------------------------------


def test_queue_holds_capacity_and_rolls_threshold_forward() -> None:
    queue = CapacityQueue(capacity=2, initial_threshold=0.5)
    admitted = [
        queue.offer(tx_id=str(i), minute=i, raw_score=s, probability=s)
        for i, s in enumerate([0.9, 0.1, 0.8, 0.95])
    ]
    assert [a is not None for a in admitted] == [True, False, True, False]  # capacity reached
    assert queue.next_for_analyst().tx_id == "0"  # type: ignore[union-attr]
    # The next day's threshold is the previous day's kth-highest score (0.9).
    assert queue.offer(tx_id="n", minute=1440, raw_score=0.85, probability=0.85) is None
    assert queue.threshold == pytest.approx(0.9)
    assert queue.offer(tx_id="m", minute=1441, raw_score=0.91, probability=0.91) is not None


def test_queue_without_history_or_initial_threshold_stays_silent_on_day_one() -> None:
    queue = CapacityQueue(capacity=1)
    assert queue.offer(tx_id="a", minute=0, raw_score=0.99, probability=0.99) is None
    assert queue.offer(tx_id="b", minute=1440, raw_score=0.99, probability=0.99) is not None


def test_risk_bands() -> None:
    assert [risk_band(p) for p in (0.6, 0.3, 0.1, 0.001)] == [
        "critical",
        "high",
        "elevated",
        "watch",
    ]


# -- drift ----------------------------------------------------------------------------------------


def test_psi_is_low_for_same_distribution_and_high_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    reference = build_reference("x", rng.normal(size=100_000))
    assert psi(reference, rng.normal(size=50_000)) < 0.01
    assert psi(reference, rng.normal(loc=1.0, size=50_000)) > 0.25


def test_psi_detects_a_feature_going_null() -> None:
    rng = np.random.default_rng(1)
    reference = build_reference("x", rng.normal(size=10_000))
    assert psi(reference, np.full(1000, np.nan)) > 1.0


def test_constant_feature_has_a_valid_reference() -> None:
    reference = build_reference("x", np.zeros(100))
    assert psi(reference, np.zeros(10)) < 1e-9
    assert psi(reference, np.ones(10)) >= 0


def test_alert_level_rises_with_noisy_baseline() -> None:
    rng = np.random.default_rng(2)
    reference = build_reference("x", rng.normal(size=100_000))
    quiet = [rng.normal(size=50_000) for _ in range(3)]
    noisy = [rng.normal(loc=0.8, size=50_000) for _ in range(3)]
    assert calibrate_alert_level(reference, quiet) == 0.1
    assert calibrate_alert_level(reference, noisy) > 0.1


def test_monitor_names_only_the_broken_feature() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(50_000, 3))
    monitor = DriftMonitor.from_matrix(["a", "b", "c"], train)
    live = rng.normal(size=(20_000, 3))
    live[:, 1] = np.nan
    assert list(monitor.alerts(live)) == ["b"]


def test_part_matrix_reads_the_same_rows_as_memory(fixture_paths, setup) -> None:  # type: ignore[no-untyped-def]
    from unravel.streaming.parity import PartMatrix

    backfill(fixture_paths, chunk_size=1000)
    parts = PartMatrix(fixture_paths.features, FEATURE_NAMES)
    for row in (0, 999, 1000, 2500, len(setup["ids"]) - 1, 2):
        assert np.array_equal(parts[row], setup["matrix"][row], equal_nan=True)
    report = compare(setup["ids"], parts, _run_clean(setup).messages)
    assert report.passed, report


def test_drift_reference_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(4)
    train = rng.normal(size=(5000, 2))
    train[:100, 1] = np.nan
    monitor = DriftMonitor.from_matrix(["a", "b"], train)
    monitor.levels = {"a": 0.2, "b": 0.3}
    monitor.save(tmp_path / "ref.json")
    loaded = DriftMonitor.load(tmp_path / "ref.json")
    live = rng.normal(loc=0.5, size=(1000, 2))
    assert loaded.levels == monitor.levels
    assert loaded.check(live) == pytest.approx(monitor.check(live))


def test_one_unusual_day_does_not_inflate_the_alert_level() -> None:
    rng = np.random.default_rng(5)
    reference = build_reference("x", rng.normal(size=100_000))
    days = [rng.normal(size=50_000) for _ in range(4)] + [rng.normal(loc=2.0, size=50_000)]
    assert calibrate_alert_level(reference, days) == 0.1
