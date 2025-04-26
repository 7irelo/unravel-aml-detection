"""Hand-computed feature checks, plus the causality and determinism properties."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import FX, make_event
from unravel.features.engine import FEATURE_NAMES, FeatureEngine, TxEvent, vectors_equal
from unravel.features.registry import all_described, family


def value(vector: list[float], name: str) -> float:
    return vector[FEATURE_NAMES.index(name)]


def engine() -> FeatureEngine:
    return FeatureEngine(FX)


def test_every_feature_is_described_and_named_once() -> None:
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert set(all_described()) == set(FEATURE_NAMES)
    assert {family(n) for n in FEATURE_NAMES} >= {"transaction", "graph", "velocity_diversity"}


def test_no_identifier_or_label_features() -> None:
    banned = ("label", "laundering", "tx_id", "order_key", "pattern", "bank_id", "account_id")
    assert not [n for n in FEATURE_NAMES if any(b in n for b in banned)]


def test_a_transaction_does_not_see_itself() -> None:
    v = engine().process(make_event(10, "a", "b"))
    assert value(v, "src_out_cnt_1h") == 0
    assert value(v, "dst_in_cnt_1h") == 0
    assert value(v, "pair_is_new") == 1
    assert value(v, "pair_count_prior") == 0


def test_velocity_counts_sums_and_distinct() -> None:
    e = engine()
    e.process(make_event(0, "a", "b", 100))
    e.process(make_event(10, "a", "c", 300))
    v = e.process(make_event(20, "a", "d", 50))
    assert value(v, "src_out_cnt_1h") == 2
    assert value(v, "src_out_distinct_1h") == 2
    assert math.isclose(value(v, "src_out_sum_1h"), math.log1p(400))


def test_window_expiry_is_exact_at_the_boundary() -> None:
    e = engine()
    e.process(make_event(0, "a", "b", 100))
    inside = e.process(make_event(59, "a", "x", 1))
    assert value(inside, "src_out_cnt_1h") == 1
    e2 = engine()
    e2.process(make_event(0, "a", "b", 100))
    outside = e2.process(make_event(60, "a", "x", 1))
    assert value(outside, "src_out_cnt_1h") == 0
    assert value(outside, "src_out_cnt_6h") == 1
    assert value(outside, "src_out_sum_1h") == 0


def test_repeat_counterparty_counts_once_until_all_copies_expire() -> None:
    e = engine()
    e.process(make_event(0, "a", "b"))
    e.process(make_event(30, "a", "b"))
    v = e.process(make_event(40, "a", "z"))
    assert value(v, "src_out_cnt_1h") == 2
    assert value(v, "src_out_distinct_1h") == 1
    later = e.process(make_event(70, "a", "z"))
    # The minute-0 and minute-30 "b" events: only minute-30 survives; plus "z" at 40.
    assert value(later, "src_out_distinct_1h") == 2


def test_entropy_matches_definition() -> None:
    e = engine()
    for i, peer in enumerate(["b", "b", "c", "d"]):
        e.process(make_event(i, "a", peer))
    v = e.process(make_event(10, "a", "z"))
    probabilities = [0.5, 0.25, 0.25]
    expected = -sum(p * math.log(p) for p in probabilities)
    assert math.isclose(value(v, "src_out_entropy_24h"), expected, abs_tol=1e-12)


def test_round_amount_ratio() -> None:
    e = engine()
    e.process(make_event(0, "a", "b", 1000.0))
    e.process(make_event(1, "a", "b", 123.45))
    v = e.process(make_event(2, "a", "b", 7.0))
    assert value(v, "src_out_round_ratio_24h") == 0.5
    assert value(v, "amount_round_100") == 0
    round_v = engine().process(make_event(0, "a", "b", 5000.0))
    assert value(round_v, "amount_round_100") == 1
    assert value(round_v, "amount_round_1000") == 1


def test_pass_through_ratio() -> None:
    e = engine()
    e.process(make_event(0, "x", "mule", 1000))
    e.process(make_event(5, "mule", "y", 400))
    v = e.process(make_event(6, "mule", "z", 100))
    assert math.isclose(value(v, "src_forward_ratio_6h"), 0.4)
    assert math.isclose(value(v, "amount_over_last_in"), 0.1)


def test_fx_normalisation() -> None:
    euro = engine().process(make_event(0, "a", "b", 100, currency="Euro"))
    assert math.isclose(value(euro, "log_usd"), math.log1p(110))
    unknown = engine().process(make_event(0, "a", "b", 100, currency="Doubloon"))
    assert value(unknown, "log_usd") == 0.0


def test_scatter_gather_two_paths() -> None:
    e = engine()
    for mid in ("m1", "m2", "m3"):
        e.process(make_event(0, "s", mid))
    for mid in ("m1", "m2", "m3"):
        e.process(make_event(1, mid, "t"))
    v = e.process(make_event(2, "s", "t"))
    assert value(v, "two_paths_6h") == 3


def test_cycle_closure_detected_at_correct_length() -> None:
    e = engine()
    e.process(make_event(0, "a", "b"))
    e.process(make_event(1, "b", "c"))
    e.process(make_event(2, "c", "d"))
    closing = e.process(make_event(3, "d", "a"))
    assert value(closing, "cycle_min_hops_96h") == 4
    assert math.isnan(value(e.process(make_event(4, "z", "y")), "cycle_min_hops_96h"))


def test_reciprocal_and_three_cycle() -> None:
    e = engine()
    e.process(make_event(0, "a", "b"))
    v = e.process(make_event(1, "b", "a"))
    assert value(v, "reciprocal_96h") == 1
    assert value(v, "cycle_min_hops_96h") == 2
    e = engine()
    e.process(make_event(0, "b", "c"))
    e.process(make_event(1, "c", "a"))
    three = e.process(make_event(2, "a", "b"))
    assert value(three, "cycle3_paths_96h") == 1
    assert value(three, "cycle_min_hops_96h") == 3


def test_out_of_order_event_is_rejected() -> None:
    e = engine()
    e.process(make_event(10, "a", "b"))
    with pytest.raises(ValueError, match="out-of-order"):
        e.process(make_event(9, "a", "b"))


# -- properties -----------------------------------------------------------------------------------

_events = st.lists(
    st.tuples(
        st.integers(0, 9000),
        st.sampled_from(list("abcdefgh")),
        st.sampled_from(list("abcdefgh")),
        st.floats(0.01, 1e7, allow_nan=False),
        st.sampled_from(["ACH", "Wire", "Cash", "Cheque"]),
        st.sampled_from(["US Dollar", "Euro", "Yuan"]),
    ),
    min_size=2,
    max_size=70,
)


def _stream(rows: list[tuple[int, str, str, float, str, str]]) -> list[TxEvent]:
    return [
        make_event(minute, s, d, amount, tx_id=f"T{i}", fmt=fmt, currency=currency)
        for i, (minute, s, d, amount, fmt, currency) in enumerate(sorted(rows, key=lambda r: r[0]))
    ]


@settings(max_examples=60, deadline=None)
@given(_events, st.data())
def test_future_events_cannot_change_earlier_vectors(rows, data) -> None:  # type: ignore[no-untyped-def]
    """Causality: changing anything after position ``i`` leaves positions up to ``i`` unchanged."""
    events = _stream(rows)
    cut = data.draw(st.integers(0, len(events) - 1))
    original = engine().process_all(events)
    mutated = list(events[: cut + 1])
    for event in events[cut + 1 :]:
        mutated.append(event._replace(amount_paid=event.amount_paid * 3 + 1, dst="zz"))
    changed = engine().process_all(mutated)
    for a, b in zip(original[: cut + 1], changed[: cut + 1], strict=True):
        assert vectors_equal(a, b, tol=0.0)


@settings(max_examples=40, deadline=None)
@given(_events)
def test_determinism(rows) -> None:  # type: ignore[no-untyped-def]
    events = _stream(rows)
    a = engine().process_all(events)
    b = engine().process_all(events)
    assert all(vectors_equal(x, y, tol=0.0) for x, y in zip(a, b, strict=True))


@settings(max_examples=40, deadline=None)
@given(_events, st.data())
def test_snapshot_restore_is_exact(rows, data) -> None:  # type: ignore[no-untyped-def]
    """Recovery: restoring from a snapshot at any point gives exactly the same later vectors."""
    events = _stream(rows)
    cut = data.draw(st.integers(0, len(events)))
    straight = engine().process_all(events)
    first = engine()
    head = first.process_all(events[:cut])
    resumed = FeatureEngine.restore(first.snapshot())
    tail = resumed.process_all(events[cut:])
    for a, b in zip(straight, head + tail, strict=True):
        assert vectors_equal(a, b, tol=0.0)


@settings(max_examples=40, deadline=None)
@given(_events)
def test_rolling_aggregates_match_brute_force(rows) -> None:  # type: ignore[no-untyped-def]
    """Incremental windows must agree with recomputing the definition from scratch."""
    events = _stream(rows)
    e = engine()
    history: list[TxEvent] = []
    for event in events:
        v = e.process(event)
        for label, width in (("1h", 60), ("24h", 1440), ("96h", 5760)):
            mine = [h for h in history if h.src == event.src and event.minute - h.minute < width]
            assert value(v, f"src_out_cnt_{label}") == len(mine)
            assert value(v, f"src_out_distinct_{label}") == len({h.dst for h in mine})
        history.append(event)
