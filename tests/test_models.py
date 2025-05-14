"""Calibration, prior correction, reason codes and an end-to-end tree fit on the fixture."""

from __future__ import annotations

import numpy as np
import pytest

from unravel.config import DatasetPaths
from unravel.eval import metrics as m
from unravel.eval.report import evaluate_scores
from unravel.eval.splits import TEST, TRAIN, VAL_A, VAL_B
from unravel.explain.reasons import global_importance, reason_codes
from unravel.features.backfill import backfill
from unravel.features.engine import FEATURE_NAMES, FEATURE_SETS
from unravel.features.registry import describe
from unravel.models.baselines import RULE_COLUMNS, amount_scores, random_scores, rule_scores
from unravel.models.calibration import Calibrator, correct_for_downsampling
from unravel.models.dataset import ModelData, load_model_data
from unravel.models.tree import TreeConfig, TreeModel, feature_matrix, train_tree


def _synthetic(n: int = 200_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true = 1 / (1 + np.exp(-(rng.normal(size=n) * 1.5 - 6)))
    y = (rng.random(n) < true).astype(np.int64)
    return true, y


@pytest.mark.parametrize("method", ["platt", "beta", "isotonic"])
def test_calibrators_repair_a_distorted_score(method: str) -> None:
    true, y = _synthetic()
    distorted = np.clip(true**0.5, 1e-6, 1 - 1e-6)
    calibrator = Calibrator.fit(method, distorted[:100_000], y[:100_000])  # type: ignore[arg-type]
    fixed = calibrator.transform(distorted[100_000:])
    assert m.log_loss(y[100_000:], fixed) < m.log_loss(y[100_000:], distorted[100_000:])
    # Monotonic: the map can't reorder distinct scores.
    grid = np.linspace(0.001, 0.999, 500)
    assert np.all(np.diff(calibrator.transform(grid)) >= -1e-12)


def test_monotonic_calibration_preserves_ranking_metrics() -> None:
    true, y = _synthetic()
    calibrated = Calibrator.fit("platt", true, y).transform(true)
    assert m.pr_auc(y, calibrated) == pytest.approx(m.pr_auc(y, true), abs=1e-9)


def test_isotonic_ties_would_change_top_k_so_we_rank_on_raw() -> None:
    """Why the raw score stays the ranking key."""
    true, y = _synthetic()
    iso = Calibrator.fit("isotonic", true, y).transform(true)
    assert len(np.unique(iso)) < len(np.unique(true)) / 10


def test_calibrator_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    true, y = _synthetic(20_000)
    calibrator = Calibrator.fit("isotonic", true, y)
    calibrator.save(tmp_path / "c.json")
    again = Calibrator.load(tmp_path / "c.json")
    assert np.array_equal(calibrator.transform(true), again.transform(true))


def test_downsampling_correction_recovers_the_prior() -> None:
    rng = np.random.default_rng(1)
    n = 2_000_000
    y = (rng.random(n) < 0.001).astype(int)
    keep = 0.05
    kept = (y == 1) | (rng.random(n) < keep)
    sampled_rate = y[kept].mean()
    assert correct_for_downsampling(np.array([sampled_rate]), keep)[0] == pytest.approx(
        y.mean(), rel=0.1
    )


def test_calibrator_refuses_a_window_without_positives() -> None:
    with pytest.raises(ValueError, match="no positives"):
        Calibrator.fit("platt", np.array([0.1, 0.2]), np.array([0, 0]))


def test_reason_codes_rank_only_positive_contributions() -> None:
    columns = ["log_usd", "two_paths_6h", "cross_bank"]
    reasons = reason_codes(columns, np.array([5.0, 3.0, np.nan]), np.array([0.2, 1.5, -0.3]), top=3)
    assert [r.feature for r in reasons] == ["two_paths_6h", "log_usd"]
    assert reasons[0].description == describe("two_paths_6h")
    assert global_importance(columns, np.array([[1.0, -3.0, 0.0]]))[0] == ("two_paths_6h", 3.0)


@pytest.fixture(scope="module")
def data(fixture_paths: DatasetPaths) -> ModelData:
    backfill(fixture_paths, chunk_size=2000)
    return load_model_data(fixture_paths)


def test_partitions_are_temporal(data: ModelData) -> None:
    for earlier, later in ((TRAIN, VAL_A), (VAL_A, VAL_B), (VAL_B, TEST)):
        assert data.minute[data.partition == earlier].max() < (
            data.minute[data.partition == later].min()
        )


def test_tree_fit_save_load_and_explain(data: ModelData, tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = TreeConfig(
        "T3",
        FEATURE_SETS["T3"],
        keep_rate=0.5,
        rounds=40,
        early_stopping=20,
        params={"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 5},
    )
    model = train_tree(data, config)
    matrix = feature_matrix(data, config.columns, data.mask(TEST))
    p = model.predict_proba(matrix)
    assert np.all((p >= 0) & (p <= 1))

    model.save(tmp_path / "m")
    loaded = TreeModel.load(tmp_path / "m")
    assert np.allclose(loaded.predict_margin(matrix), model.predict_margin(matrix))
    assert loaded.model_hash() == model.model_hash()

    # TreeSHAP's additivity: contributions plus the base value equal the margin.
    contributions = model.contributions(matrix)
    assert contributions.shape == (matrix.shape[0], matrix.shape[1] + 1)
    assert np.allclose(contributions.sum(axis=1), model.predict_margin(matrix), atol=1e-4)


def test_xgboost_arm_trains(data: ModelData) -> None:
    config = TreeConfig(
        "T3x",
        FEATURE_SETS["T2"],
        rounds=15,
        early_stopping=10,
        framework="xgboost",
        keep_rate=0.5,
    )
    model = train_tree(data, config)
    assert model.predict_proba(feature_matrix(data, config.columns, data.mask(VAL_B))).max() <= 1


def test_baselines_and_full_report(data: ModelData) -> None:
    assert set(RULE_COLUMNS) <= set(FEATURE_NAMES)
    n = data.features.height
    frame = data.features.frame(RULE_COLUMNS)
    assert len(random_scores(n)) == n
    assert len(amount_scores(frame)) == n
    scores = rule_scores(frame)
    mask = data.mask(TEST)
    report = evaluate_scores(data, scores, mask, threshold=1.0, n_boot=10)
    assert report["positives"] > 0
    assert 0 <= report["pr_auc"] <= 1
    assert report["capacity"]["100"]["days"] >= 1
    assert "natural" in report["per_pattern"] or report["per_pattern"]


def test_tail_days_are_reported_separately(data: ModelData) -> None:
    """Days far below median volume must not hide inside a headline metric."""
    from dataclasses import replace

    from unravel.eval.report import full_volume_mask

    assert full_volume_mask(data).all()  # the fixture has no tail
    tail_day = int(data.day.max()) + 1
    day = data.day.copy()
    last = np.flatnonzero(data.partition == TEST)[-3:]
    day[last] = tail_day
    with_tail = replace(data, day=day)
    mask = full_volume_mask(with_tail)
    assert not mask[last].any()
    assert mask.sum() == len(day) - 3
    report = evaluate_scores(
        with_tail, np.arange(len(day), dtype=float), data.mask(TEST), 1.0, n_boot=5
    )
    assert report["full_volume_days"]["transactions"] == report["transactions"] - 3
