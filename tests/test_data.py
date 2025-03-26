"""Ingest, label isolation and fixture behaviour."""

from __future__ import annotations

import polars as pl

from tests.conftest import make_event
from unravel.config import DatasetPaths
from unravel.data.audit import run_audit
from unravel.data.fixtures import FIXTURE_PATTERNS, generate_fixture
from unravel.data.ingest import ingest, parse_patterns
from unravel.data.schema import LABEL_COLUMNS, TX_COLUMNS, make_tx_id
from unravel.features.backfill import backfill, iter_events, load_features
from unravel.features.engine import FEATURE_NAMES, FeatureEngine, vectors_equal
from unravel.features.fx import fit_fx, load_fx


def test_transaction_id_is_stable_and_row_sensitive() -> None:
    assert make_tx_id(0, "x") == make_tx_id(0, "x")
    assert make_tx_id(0, "x")[0] != make_tx_id(1, "x")[0]
    assert make_tx_id(0, "x")[0].startswith("T")


def test_labels_are_separate_from_transactions(fixture_paths: DatasetPaths) -> None:
    tx = pl.read_parquet(fixture_paths.transactions)
    assert tx.columns == TX_COLUMNS
    assert not {"is_laundering", "pattern", "pattern_instance"} & set(tx.columns)
    assert pl.read_parquet(fixture_paths.labels).columns == LABEL_COLUMNS
    assert "is_laundering" not in make_event(0, "a", "b")._fields


def test_ingest_sorts_canonically(fixture_paths: DatasetPaths) -> None:
    tx = pl.read_parquet(fixture_paths.transactions)
    assert tx.select("minute", "order_key").is_duplicated().sum() == 0
    assert tx["minute"].is_sorted()
    keys = tx.select(pl.struct("minute", "order_key").alias("k"))["k"].to_list()
    assert keys == sorted(keys, key=lambda k: (k["minute"], k["order_key"]))


def test_bank_codes_keep_leading_zeros(fixture_paths: DatasetPaths) -> None:
    tx = pl.read_parquet(fixture_paths.transactions)
    assert tx["from_bank"].str.starts_with("0").any()


def test_patterns_parse_and_join(fixture_paths: DatasetPaths) -> None:
    rows = parse_patterns(fixture_paths.raw_patterns)
    assert {r.pattern for r in rows} == set(FIXTURE_PATTERNS)
    labels = pl.read_parquet(fixture_paths.labels)
    assert (
        labels.filter(pl.col("is_laundering") == 0)["pattern"].null_count()
        == labels.filter(pl.col("is_laundering") == 0).height
    )
    # Fixture "natural" laundering has no pattern, like most positives in the real data.
    positives = labels.filter(pl.col("is_laundering") == 1)
    assert positives["pattern"].null_count() > 0
    assert positives["pattern"].is_not_null().sum() > 0


def test_parser_reads_the_real_header_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "p.txt"
    path.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out\n"
        "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops\n"
        "2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )
    rows = parse_patterns(path)
    assert [(r.pattern, r.instance) for r in rows] == [("FAN-OUT", 0), ("CYCLE", 1)]
    assert rows[0].key.endswith("Euro,ACH")


def test_fixture_is_seeded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    a = DatasetPaths("a", root=tmp_path)
    b = DatasetPaths("b", root=tmp_path)
    generate_fixture(a, days=3, background_per_day=20, accounts=30, instances_per_pattern=1)
    generate_fixture(b, days=3, background_per_day=20, accounts=30, instances_per_pattern=1)
    assert a.raw_trans.read_bytes() == b.raw_trans.read_bytes()
    ingest(a)
    assert pl.read_parquet(a.labels)["is_laundering"].sum() > 0


def test_fx_recovers_rates_from_cross_currency_rows() -> None:
    frame = pl.DataFrame(
        {
            "payment_currency": ["US Dollar", "Euro", "US Dollar"],
            "receiving_currency": ["Euro", "Yuan", "US Dollar"],
            "amount_paid": [110.0, 1.0, 5.0],
            "amount_received": [100.0, 8.0, 5.0],
        }
    )
    rates = fit_fx(frame)
    assert abs(rates["Euro"] - 1.1) < 1e-12
    assert abs(rates["Yuan"] - 1.1 / 8) < 1e-12


def test_backfill_equals_direct_engine_run(fixture_paths: DatasetPaths) -> None:
    backfill(fixture_paths, chunk_size=1000)
    stored = load_features(fixture_paths.features)
    assert stored.columns == ["tx_id", *FEATURE_NAMES]
    transactions = pl.read_parquet(fixture_paths.transactions)
    assert stored["tx_id"].to_list() == transactions["tx_id"].to_list()
    engine = FeatureEngine(load_fx(fixture_paths.fx))
    matrix = stored.select(FEATURE_NAMES).to_numpy()
    for i, event in enumerate(iter_events(transactions)):
        assert vectors_equal(engine.process(event), matrix[i].tolist(), tol=0.0)


def test_audit_reports_real_measurements(fixture_paths: DatasetPaths) -> None:
    report = run_audit(fixture_paths)
    for heading in ("## Overview", "## Split", "## Named patterns", "## Single-feature signal"):
        assert heading in report
