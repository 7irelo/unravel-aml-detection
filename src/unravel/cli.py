"""Command-line entry point. Heavy dependencies are imported inside commands to keep it fast."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

from unravel.config import ROOT, DatasetPaths

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, default=float))


# -- data -----------------------------------------------------------------------------------------


@app.command()
def fixture(name: str = "fixture", seed: int = 7) -> None:
    """Generate the synthetic fixture in raw AMLworld format."""
    from unravel.data.fixtures import generate_fixture

    _print(generate_fixture(DatasetPaths(name), seed=seed))


@app.command()
def ingest(name: str = "HI-Small") -> None:
    """Parse raw files to canonical Parquet, then fix the split and the training-window FX table."""
    from unravel.data.ingest import ingest as run_ingest
    from unravel.data.ingest import write_manifest
    from unravel.eval.splits import make_split
    from unravel.features.fx import fit_fx, save_fx

    paths = DatasetPaths(name)
    _print(run_ingest(paths))
    if name != "fixture":
        write_manifest(paths, ROOT / "data" / "manifest.json")
    tx = pl.read_parquet(paths.transactions)
    split = make_split(tx["minute"].to_numpy())
    split.save(paths.split)
    save_fx(fit_fx(tx.filter(pl.col("minute") < split.val_start)), paths.fx)
    print(split)


@app.command()
def audit(name: str = "HI-Small") -> None:
    """Write the data audit to ``reports/<name>/data_audit.md``."""
    from unravel.data.audit import run_audit

    paths = DatasetPaths(name)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "data_audit.md").write_text(run_audit(paths), encoding="utf-8")
    print(f"wrote {paths.reports / 'data_audit.md'}")


@app.command()
def features(name: str = "HI-Small") -> None:
    """Backfill features through the shared engine."""
    from unravel.features.backfill import backfill

    _print(backfill(DatasetPaths(name)))


# -- experiments ----------------------------------------------------------------------------------


@app.command()
def baselines(name: str = "HI-Small") -> None:
    """Evaluate random, amount and handwritten rule baselines on validation A."""
    from unravel.experiments import run_baselines

    results = run_baselines(DatasetPaths(name))
    _print({k: {"pr_auc": v["pr_auc"], "f1": v["minority_f1"]["f1"]} for k, v in results.items()})


@app.command("train-tree")
def train_tree(arm: str, name: str = "HI-Small", seeds: int = 5) -> None:
    """Train one tree arm across seeds, evaluated on validation A."""
    from unravel.experiments import SEEDS, run_tree_arm
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    summary = run_tree_arm(paths, load_model_data(paths), arm, SEEDS[:seeds])
    _print({k: summary[k] for k in ("arm", "pr_auc_mean", "pr_auc_std", "f1_mean", "f1_std")})


@app.command()
def study(which: str, name: str = "HI-Small", trials: int = 30) -> None:
    """Run a validation study: imbalance, tune, calibration, shap, drift, order or agent-rules."""
    from unravel import experiments as ex
    from unravel.eval.splits import VAL_A
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    if which == "order":
        _print(ex.order_sensitivity(paths))
        return
    data = load_model_data(paths)
    if which == "imbalance":
        _print(ex.imbalance_study(paths, data))
    elif which == "tune":
        result = ex.tune(paths, data, trials)
        _print({"best_value": result["best_value"], "best_params": result["best_params"]})
    elif which == "calibration":
        result = ex.calibration_study(paths, data)
        _print(
            {k: v["log_loss"] for k, v in result["methods"].items()} | {"chosen": result["chosen"]}
        )
    elif which == "shap":
        _print(ex.explain_global(paths, data))
    elif which == "drift":
        _print(ex.drift_study(paths, data))
    elif which == "agent-rules":
        _print(ex.rule_agent_eval(paths, data, VAL_A))
    else:
        raise typer.BadParameter(f"unknown study {which!r}")


@app.command("final-eval")
def final_eval(
    arm: str,
    reason: str = typer.Option(..., help="Why test is being evaluated. Recorded in the ledger."),
    name: str = "HI-Small",
) -> None:
    """Evaluate on TEST. Each call leaves a permanent ledger entry."""
    from unravel.experiments import final_evaluation
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    summary = final_evaluation(paths, load_model_data(paths), arm, reason)
    _print({k: v for k, v in summary.items() if k != "runs"})


@app.command("final-eval-gnn")
def final_eval_gnn(
    arm: str = "G3",
    seed: int = 0,
    reason: str = typer.Option(..., help="Why test is being evaluated. Recorded in the ledger."),
    name: str = "HI-Small",
) -> None:
    """Evaluate a saved GNN on TEST. Leaves a ledger entry."""
    from unravel.experiments import gnn_final_evaluation
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    result = gnn_final_evaluation(paths, load_model_data(paths), arm, seed, reason)
    _print({"pr_auc": result["pr_auc"], "f1": result["minority_f1"]})


@app.command("agent-rules-test")
def agent_rules_test(name: str = "HI-Small") -> None:
    """Run the no-LLM agent evaluation on real test-window alerts."""
    from unravel.eval.splits import TEST
    from unravel.experiments import rule_agent_eval
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    result = rule_agent_eval(paths, load_model_data(paths), TEST)
    _print({k: result[k] for k in ("strata_available", "rule_baseline")})


@app.command("agent-eval")
def agent_eval(
    budget: float = typer.Option(..., help="Hard spending limit in USD."),
    test: bool = False,
    name: str = "HI-Small",
) -> None:
    """Evaluate the real LLM investigator and judge. Needs ANTHROPIC_API_KEY and spends money."""
    from unravel.eval.splits import TEST, VAL_A
    from unravel.experiments import agent_llm_eval
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    code = TEST if test else VAL_A
    result = agent_llm_eval(paths, load_model_data(paths), code, budget_usd=budget)
    _print({k: result[k] for k in ("strata_available", "total_cost_usd", "citation_support_rate")})


@app.command("gold-sheet")
def gold_sheet_command(name: str = "HI-Small") -> None:
    """Export a blind rating sheet for calibrating the judge against human ratings."""
    from unravel.eval.splits import VAL_A
    from unravel.experiments import gold_sheet
    from unravel.models.dataset import load_model_data

    paths = DatasetPaths(name)
    print(gold_sheet(paths, load_model_data(paths), VAL_A))


@app.command()
def report(name: str = "HI-Small") -> None:
    """Build ``docs/results.md`` from the JSON reports."""
    from unravel.reporting import write_results

    print(write_results(name))


@app.command("train-gnn")
def train_gnn_command(
    arm: str = "G3",
    name: str = "HI-Small",
    protocol: str = "P2",
    seed: int = 0,
    epochs: int = 20,
    hidden: int = 64,
    negatives_per_positive: int = 10,
    fanout: str = "20,10",
    batch_size: int = 512,
    plain: bool = typer.Option(
        False, help="Plain PNA: no ego IDs, ports, reverse MP or edge updates."
    ),
) -> None:
    """Train a GNN arm. It scores validation only; see ``final-eval-gnn`` for test."""
    import numpy as np

    from unravel.eval import metrics as m
    from unravel.eval.splits import VAL_A
    from unravel.features.engine import FEATURE_SETS
    from unravel.models.gnn import GnnConfig, load_edges, save_model, score_partitions, train_gnn

    paths = DatasetPaths(name)
    extra = tuple(FEATURE_SETS["T2"]) if arm == "G4" else ()
    config = GnnConfig(
        arm=arm,
        protocol=protocol,  # type: ignore[arg-type]
        seed=seed,
        epochs=epochs,
        hidden=hidden,
        negatives_per_positive=negatives_per_positive,
        fanout=tuple(int(x) for x in fanout.split(",")),
        batch_size=batch_size,
        extra_columns=extra,
        ego=not plain,
        ports=not plain,
        reverse=not plain,
        edge_updates=not plain,
    )
    table = load_edges(paths, extra)
    result = train_gnn(table, config)
    directory = paths.artifacts / "models" / arm / f"seed{seed}"
    save_model(result, config, directory)
    scores = score_partitions(table, result, config, (VAL_A,))
    mask = table.partition == VAL_A
    threshold = m.best_f1_threshold(table.y[mask], scores[VAL_A])
    report = {
        "arm": arm,
        "protocol": protocol,
        "seed": seed,
        "best_epoch": result.best_epoch,
        "training_seconds": result.seconds,
        "device": "cpu",
        "config": {
            "hidden": hidden,
            "fanout": fanout,
            "epochs": epochs,
            "negatives_per_positive": negatives_per_positive,
        },
        "val_a_pr_auc": m.pr_auc(table.y[mask], scores[VAL_A]),
        "val_a_f1": m.f1_at(table.y[mask], (scores[VAL_A] >= threshold).astype(np.int64)),
        "threshold": threshold,
    }
    output = paths.reports / "validation" / f"{arm}_{protocol}_seed{seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=float))
    np.savez_compressed(directory / "validation_scores.npz", val_a=scores[VAL_A])
    _print(report)


# -- streaming ------------------------------------------------------------------------------------


@app.command("stream-replay")
def stream_replay(
    name: str = "HI-Small",
    limit: int | None = None,
    minutes_per_second: float = 0.0,
    recreate: bool = False,
    bootstrap: str | None = None,
) -> None:
    """Publish transactions to ``tx.raw`` in canonical order, with no labels."""
    from unravel.api.settings import Settings
    from unravel.streaming.kafka import ensure_topics, replay

    server = bootstrap or Settings().kafka_bootstrap
    ensure_topics(server, recreate=recreate)
    sent = replay(
        DatasetPaths(name).transactions, server, limit=limit, minutes_per_second=minutes_per_second
    )
    print(f"replayed {sent:,}")


@app.command("stream-features")
def stream_features(
    name: str = "HI-Small",
    idle_timeout: float | None = None,
    snapshot_every: int = 50_000,
    bootstrap: str | None = None,
    state: Path | None = None,
) -> None:
    """Run the feature consumer. Restarts resume from the latest snapshot."""
    from unravel.api.settings import Settings
    from unravel.features.fx import load_fx
    from unravel.streaming.kafka import ensure_topics, run_feature_consumer

    paths = DatasetPaths(name)
    server = bootstrap or Settings().kafka_bootstrap
    ensure_topics(server)
    offset = run_feature_consumer(
        server,
        load_fx(paths.fx),
        state or paths.artifacts / "consumer_state.pkl",
        snapshot_every=snapshot_every,
        idle_timeout=idle_timeout,
    )
    print(f"stopped at offset {offset:,}")


@app.command("stream-score")
def stream_score(
    name: str = "HI-Small",
    arm: str = "T3",
    seed: int = 0,
    idle_timeout: float | None = None,
    metrics_port: int | None = None,
) -> None:
    """Score feature messages, write scores and alerts."""
    from unravel.api.settings import Settings
    from unravel.streaming.pipeline import run_scorer

    settings = Settings()
    paths = DatasetPaths(name)
    total = run_scorer(
        settings.kafka_bootstrap,
        settings.database_url,
        paths.artifacts / "models" / arm / f"seed{seed}",
        capacity=settings.capacity_per_day,
        idle_timeout=idle_timeout,
        metrics_port=metrics_port,
    )
    print(f"scored {total:,}")


@app.command("stream-parity")
def stream_parity(
    name: str = "HI-Small",
    limit: int | None = None,
    bootstrap: str | None = None,
    idle_timeout: float = 20.0,
) -> None:
    """Read ``tx.features`` back from the broker and compare it with offline Parquet."""
    from unravel.api.settings import Settings
    from unravel.features.engine import FEATURE_NAMES
    from unravel.models.dataset import FeatureReader
    from unravel.streaming.kafka import read_topic
    from unravel.streaming.messages import TOPIC_FEATURES
    from unravel.streaming.parity import PartMatrix, compare

    paths = DatasetPaths(name)
    ids = FeatureReader(paths.features).tx_ids()
    if limit is not None:
        ids = ids.head(limit)
    server = bootstrap or Settings().kafka_bootstrap
    messages = (
        payload
        for _, payload in read_topic(
            server, TOPIC_FEATURES, idle_timeout=idle_timeout, group="unravel-parity"
        )
    )
    report = compare(ids.to_list(), PartMatrix(paths.features, FEATURE_NAMES), messages)
    result = report.as_dict()
    _print(result)
    output = paths.reports / "parity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=float))
    if not report.passed:
        raise typer.Exit(1)


# -- agent and demo -------------------------------------------------------------------------------


@app.command("agent-worker")
def agent_worker(once: bool = False) -> None:
    """Turn queued alerts into case files for review."""
    from unravel.api.cases import run_worker
    from unravel.api.settings import Settings

    print(f"drafted {run_worker(Settings(), once=once)} cases")


@app.command("demo-prepare")
def demo_prepare(name: str = "fixture") -> None:
    """Build everything the no-credentials demo needs, on synthetic data."""
    import psycopg

    from unravel.api.settings import Settings
    from unravel.data.fixtures import generate_fixture
    from unravel.data.ingest import ingest as run_ingest
    from unravel.eval.splits import TRAIN, VAL_B, make_split
    from unravel.features.backfill import backfill
    from unravel.features.engine import FEATURE_SETS
    from unravel.features.fx import fit_fx, load_fx, save_fx
    from unravel.models.calibration import Calibrator
    from unravel.models.dataset import load_model_data
    from unravel.models.tree import TreeConfig, feature_matrix
    from unravel.models.tree import train_tree as fit
    from unravel.monitoring.drift import DriftMonitor
    from unravel.streaming.kafka import ensure_topics
    from unravel.streaming.pipeline import load_transactions

    settings = Settings()
    paths = DatasetPaths(name)
    # Safe to re-run: a stack that's already loaded is left alone. A fresh one has empty topics, so
    # a consumer snapshot left by an earlier run would point at offsets that no longer exist.
    with psycopg.connect(settings.database_url) as connection:
        row = connection.execute("SELECT count(*) FROM transactions").fetchone()
    if row is not None and row[0] > 0:
        print(f"already loaded: {row[0]:,} transactions")
        return
    (paths.artifacts / "consumer_state.pkl").unlink(missing_ok=True)
    generate_fixture(paths)
    run_ingest(paths)
    tx = pl.read_parquet(paths.transactions)
    split = make_split(tx["minute"].to_numpy())
    split.save(paths.split)
    save_fx(fit_fx(tx.filter(pl.col("minute") < split.val_start)), paths.fx)
    backfill(paths, chunk_size=5000)
    data = load_model_data(paths)
    model = fit(
        data,
        TreeConfig(
            "T3",
            FEATURE_SETS["T3"],
            keep_rate=0.5,
            rounds=120,
            early_stopping=40,
            params={"learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 5},
        ),
    )
    directory = paths.artifacts / "models" / "T3" / "seed0"
    model.save(directory)
    mask = data.mask(VAL_B)
    raw = model.predict_proba(feature_matrix(data, model.config.columns, mask))
    Calibrator.fit("platt", raw, data.y[mask].astype("int64")).save(directory / "calibrator.json")
    train_rows = data.features.matrix(FEATURE_SETS["T3"], data.mask(TRAIN))
    DriftMonitor.from_matrix(FEATURE_SETS["T3"], train_rows).save(
        directory / "drift_reference.json"
    )
    # Fixture volumes are small, so the demo opens its queue from a low first-day threshold.
    (directory / "policy.json").write_text(json.dumps({"initial_threshold": 0.01}))
    ensure_topics(settings.kafka_bootstrap)
    print(
        f"loaded {load_transactions(settings.database_url, paths.transactions, load_fx(paths.fx)):,}"
    )


if __name__ == "__main__":
    app()
