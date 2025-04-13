"""Metrics are checked against hand-computed cases and against scikit-learn where the two overlap."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from sklearn.metrics import log_loss as sk_log_loss

from unravel.eval import metrics as m
from unravel.eval.bootstrap import cluster_bootstrap_indices, cluster_ci, paired_difference_ci
from unravel.eval.ledger import read_ledger, record_test_evaluation
from unravel.eval.splits import TEST, TRAIN, VAL_A, VAL_B, make_split, search_boundaries


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 1), st.integers(0, 20)), min_size=2, max_size=200).filter(
        lambda rows: any(y for y, _ in rows)
    )
)
def test_pr_auc_matches_sklearn_with_ties(rows) -> None:  # type: ignore[no-untyped-def]
    y = np.array([r[0] for r in rows])
    s = np.array([r[1] for r in rows], dtype=float)
    assert m.pr_auc(y, s) == pytest.approx(average_precision_score(y, s))


def test_f1_matches_sklearn_and_threshold_is_optimal() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    s = rng.random(2000) + y * 0.4
    threshold = m.best_f1_threshold(y, s)
    best = m.f1_at(y, (s >= threshold).astype(int))["f1"]
    assert best == pytest.approx(f1_score(y, (s >= threshold).astype(int)))
    for other in np.quantile(s, np.linspace(0.5, 0.99, 25)):
        assert m.f1_at(y, (s >= other).astype(int))["f1"] <= best + 1e-12


def test_calibration_scores_match_sklearn() -> None:
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(int)
    p = rng.random(500)
    assert m.log_loss(y, p) == pytest.approx(sk_log_loss(y, p))
    assert m.brier(y, p) == pytest.approx(brier_score_loss(y, p))
    assert m.brier_skill(y, np.full(500, y.mean())) == pytest.approx(0.0, abs=1e-12)


def test_pr_auc_without_positives_is_nan() -> None:
    assert np.isnan(m.pr_auc(np.zeros(4), np.arange(4.0)))


def _frame(y, scores, day=None, account=None, instance=None, usd=None):  # type: ignore[no-untyped-def]
    n = len(y)
    return m.AlertFrame(
        y=np.array(y),
        scores=np.array(scores, dtype=float),
        day=np.zeros(n, dtype=int) if day is None else np.array(day),
        account=np.arange(n) if account is None else np.array(account),
        usd=np.ones(n) if usd is None else np.array(usd, dtype=float),
        pattern_instance=np.full(n, -1) if instance is None else np.array(instance),
    )


def test_capacity_metrics_hand_computed() -> None:
    frame = _frame(
        y=[1, 0, 1, 0, 1, 0],
        scores=[9, 8, 7, 3, 2, 1],
        day=[0, 0, 0, 1, 1, 1],
        usd=[10, 1, 20, 1, 40, 1],
        instance=[5, -1, 6, -1, 6, -1],
    )
    out = m.capacity_metrics(frame, k=2)
    # Day 0 tops 9 and 8 (one hit). Day 1 tops 3 and 2 (one hit).
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["value_captured"] == pytest.approx(50 / 70)
    assert out["pattern_hit_rate"] == pytest.approx(1.0)
    # Ceiling: day 0 has two positives and day 1 has one, all reachable with k=2.
    assert out["recall_ceiling"] == pytest.approx(1.0)


def test_recall_ceiling_when_positives_exceed_capacity() -> None:
    out = m.capacity_metrics(_frame(y=[1, 1, 1, 1], scores=[4, 3, 2, 1]), k=1)
    assert out["recall_ceiling"] == pytest.approx(0.25)
    assert out["precision"] == pytest.approx(1.0)


def test_k_above_day_volume_and_day_without_positives() -> None:
    out = m.capacity_metrics(_frame(y=[0, 0], scores=[1, 2]), k=10)
    assert out["precision"] == 0.0
    assert out["recall"] == 0.0
    assert np.isnan(out["per_day"][0]["recall"])  # type: ignore[index]


def test_case_unit_collapses_an_account_day() -> None:
    frame = _frame(y=[1, 1, 1, 0], scores=[9, 8, 7, 6], account=[1, 1, 1, 2])
    out = m.capacity_metrics(frame, k=2)
    assert out["precision"] == pytest.approx(1.0)  # the transaction queue is all one account
    assert out["case_precision"] == pytest.approx(0.5)  # but only one of two cases is real
    assert out["case_recall"] == pytest.approx(1.0)


def test_rate_metrics() -> None:
    y = np.array([1] + [0] * 9999)
    s = np.r_[1.0, np.zeros(9999)]
    out = m.rate_metrics(y, s, rate=0.0002)
    assert out["k"] == 2
    assert out["precision"] == 0.5
    assert out["recall"] == 1.0


def test_per_pattern_recall_groups_natural_positives() -> None:
    y = np.array([1, 1, 1, 0])
    flagged = np.array([1, 0, 1, 1])
    out = m.per_pattern_recall(y, flagged, ["CYCLE", "CYCLE", None, None])
    assert out["CYCLE"] == {"n": 2.0, "recall": 0.5}
    assert out["natural"] == {"n": 1.0, "recall": 1.0}


def test_online_regret_never_beats_oracle_on_a_full_queue() -> None:
    rng = np.random.default_rng(2)
    n = 6000
    y = (rng.random(n) < 0.02).astype(int)
    frame = _frame(y=y, scores=rng.random(n) + y, day=np.repeat(np.arange(3), 2000))
    out = m.online_regret(frame, k=50)
    assert out["days_evaluated"] == 2
    assert out["live_alerts_per_day"] <= 50
    assert out["oracle_precision"] >= out["live_precision"] - 1e-9


def test_reliability_flags_a_miscalibrated_model() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(200_000) ** 8
    y = (rng.random(200_000) < p).astype(int)
    assert m.adaptive_ece(y, p) < m.adaptive_ece(y, np.clip(p * 3, 0, 1))
    assert sum(r["n"] for r in m.reliability_table(y, p)) == 200_000


# -- bootstrap ------------------------------------------------------------------------------------


def test_cluster_resample_keeps_clusters_whole() -> None:
    clusters = np.array([0, 0, 0, 1, 1, 2])
    rng = np.random.default_rng(0)
    for _ in range(50):
        counts = np.bincount(clusters[cluster_bootstrap_indices(clusters, rng)], minlength=3)
        assert counts[0] % 3 == 0
        assert counts[1] % 2 == 0


def test_clustered_data_gets_a_wider_interval_than_iid() -> None:
    """The reason for clustering: correlated positives make the true uncertainty larger."""
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(300), 20)
    y = np.repeat((rng.random(300) < 0.3).astype(float), 20)
    metric = lambda idx: float(y[idx].mean())  # noqa: E731
    clustered = cluster_ci(metric, groups, n_boot=200)
    iid = cluster_ci(metric, np.arange(len(y)), n_boot=200)
    assert clustered["se"] > 2.5 * iid["se"]


def test_paired_difference_detects_a_real_gap() -> None:
    rng = np.random.default_rng(5)
    a = rng.random(2000) + 0.1
    b = a - 0.1
    out = paired_difference_ci(
        lambda i: float(a[i].mean()), lambda i: float(b[i].mean()), np.arange(2000), n_boot=50
    )
    assert out["mean_difference"] == pytest.approx(0.1)
    assert out["probability_positive"] == 1.0


# -- splits ---------------------------------------------------------------------------------------


def _ibm_search(d_ts: np.ndarray) -> tuple[int, int]:
    """IBM's loop, transcribed from Multi-GNN ``data_loading.py``, to pin our behaviour to theirs."""
    import itertools

    split_per = [0.6, 0.2, 0.2]
    scores = {}
    for i, j in itertools.combinations(range(len(d_ts)), 2):
        totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
        props = [v / np.sum(totals) for v in totals]
        scores[(i, j)] = max(abs(v - t) / t for v, t in zip(props, split_per, strict=True))
    return min(scores, key=scores.get)  # type: ignore[arg-type]


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(1, 10_000), min_size=3, max_size=20))
def test_boundary_search_matches_ibm(counts) -> None:  # type: ignore[no-untyped-def]
    d = np.array(counts)
    assert search_boundaries(d) == _ibm_search(d)


def test_split_partitions_are_ordered_and_complete() -> None:
    minutes = np.sort(np.random.default_rng(0).integers(0, 1440 * 10, 50_000))
    split = make_split(minutes)
    codes = split.assign(minutes)
    assert set(np.unique(codes)) == {TRAIN, VAL_A, VAL_B, TEST}
    assert np.all(np.diff(codes) >= 0)
    assert minutes[codes == TEST].min() >= split.test_start
    assert minutes[codes == TRAIN].max() < split.val_start


# -- ledger ---------------------------------------------------------------------------------------


def test_ledger_appends_and_requires_a_reason(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "ledger.jsonl"
    record_test_evaluation(path, dataset="d", arm="T3", protocol="P2", model_hash="x", reason="r")
    record_test_evaluation(path, dataset="d", arm="T3", protocol="P2", model_hash="x", reason="r")
    assert len(read_ledger(path)) == 2
    with pytest.raises(ValueError, match="reason"):
        record_test_evaluation(path, dataset="d", arm="a", protocol="P", model_hash="", reason=" ")
