"""Repeatable experiments. Everything here uses validation, except ``final_evaluation``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import polars as pl

from unravel.agent.store import MemoryStore
from unravel.agent_eval.dataset import build_cases
from unravel.agent_eval.score import rule_baseline, run_agent, summarise, verifier_mutation_test
from unravel.config import DatasetPaths
from unravel.eval import metrics as m
from unravel.eval.ledger import read_ledger, record_test_evaluation
from unravel.eval.report import calibration_summary, evaluate_scores, full_volume_mask
from unravel.eval.splits import TEST, TRAIN, VAL_A, VAL_B
from unravel.explain.reasons import global_importance
from unravel.features.engine import FEATURE_NAMES, FEATURE_SETS, GRAPH_FEATURES
from unravel.features.fx import load_fx
from unravel.features.registry import describe, family
from unravel.models import baselines
from unravel.models.calibration import Calibrator
from unravel.models.dataset import ModelData, load_model_data
from unravel.models.tree import DEFAULT_PARAMS, TreeConfig, TreeModel, feature_matrix, train_tree
from unravel.monitoring.drift import DriftMonitor

SEEDS = (0, 1, 2, 3, 4)
LEDGER_NAME = "test_ledger.jsonl"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float) + "\n")


def _model_dir(paths: DatasetPaths, arm: str, seed: int) -> Path:
    return paths.artifacts / "models" / arm / f"seed{seed}"


def _params(paths: DatasetPaths) -> dict[str, Any]:
    tuned = paths.artifacts / "tuned_params.json"
    if tuned.exists():
        params: dict[str, Any] = json.loads(tuned.read_text())
        return params
    return dict(DEFAULT_PARAMS)


def arm_config(paths: DatasetPaths, arm: str, seed: int, **overrides: Any) -> TreeConfig:
    """The arms from the protocol. ``T3x`` is XGBoost; ``T3-nofmt`` tests a generator artefact."""
    columns = {
        "T1": FEATURE_SETS["T1"],
        "T2": FEATURE_SETS["T2"],
        "T3": FEATURE_SETS["T3"],
        "T3x": FEATURE_SETS["T3"],
        "T3-nofmt": [c for c in FEATURE_SETS["T3"] if c != "payment_format"],
        "T3-nograph-cycles": [c for c in FEATURE_SETS["T3"] if "cycle" not in c],
    }[arm]
    framework: Literal["lightgbm", "xgboost"] = "xgboost" if arm == "T3x" else "lightgbm"
    params = _params(paths) if framework == "lightgbm" else dict(DEFAULT_PARAMS)
    return TreeConfig(
        arm=arm, columns=columns, seed=seed, params=params, framework=framework, **overrides
    )


def _raw_scores(data: ModelData, model: TreeModel, mask: np.ndarray) -> np.ndarray:
    scores = np.full(len(data.y), np.nan)
    scores[mask] = model.predict_proba(feature_matrix(data, model.config.columns, mask))
    return scores


def run_baselines(paths: DatasetPaths) -> dict[str, Any]:
    data = load_model_data(paths)
    mask = data.mask(VAL_A)
    frame = data.features.frame(baselines.RULE_COLUMNS)
    arms = {
        "B0-random": baselines.random_scores(len(data.y)),
        "B0-amount": baselines.amount_scores(frame),
        "B1-rules": baselines.rule_scores(frame),
    }
    out = {}
    for name, scores in arms.items():
        threshold = m.best_f1_threshold(data.y[mask], scores[mask])
        out[name] = evaluate_scores(data, scores, mask, threshold)
    _write_json(paths.reports / "validation" / "baselines.json", out)
    return out


def run_tree_arm(
    paths: DatasetPaths, data: ModelData, arm: str, seeds: tuple[int, ...] = SEEDS
) -> dict[str, Any]:
    """Train an arm across seeds and evaluate each seed on validation A."""
    runs = []
    for seed in seeds:
        model = train_tree(data, arm_config(paths, arm, seed))
        model.save(_model_dir(paths, arm, seed))
        mask = data.mask(VAL_A)
        scores = _raw_scores(data, model, mask)
        threshold = m.best_f1_threshold(data.y[mask], scores[mask])
        report = evaluate_scores(data, scores, mask, threshold, n_boot=50, seed=seed)
        report.update(
            seed=seed,
            best_iteration=model.best_iteration,
            training_seconds=model.seconds,
            model_hash=model.model_hash(),
        )
        print(
            f"{arm} seed={seed} pr_auc={report['pr_auc']:.4f} "
            f"f1={report['minority_f1']['f1']:.4f} iters={model.best_iteration} "
            f"t={model.seconds:.0f}s",
            flush=True,
        )
        runs.append(report)
    summary = {
        "arm": arm,
        "seeds": list(seeds),
        "pr_auc_mean": float(np.mean([r["pr_auc"] for r in runs])),
        "pr_auc_std": float(np.std([r["pr_auc"] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
        "f1_mean": float(np.mean([r["minority_f1"]["f1"] for r in runs])),
        "f1_std": float(np.std([r["minority_f1"]["f1"] for r in runs], ddof=1))
        if len(runs) > 1
        else 0.0,
        "runs": runs,
    }
    _write_json(paths.reports / "validation" / f"{arm}.json", summary)
    return summary


def imbalance_study(paths: DatasetPaths, data: ModelData, arm: str = "T3") -> dict[str, Any]:
    """Compare ways of handling imbalance, on ranking and on raw probability quality."""
    variants: dict[str, dict[str, Any]] = {
        "none": {"imbalance": "none"},
        "scale_pos_weight": {"imbalance": "scale_pos_weight"},
        "downsample_5pct": {"imbalance": "downsample", "keep_rate": 0.05},
        "downsample_1pct": {"imbalance": "downsample", "keep_rate": 0.01},
    }
    out = {}
    mask = data.mask(VAL_A)
    y = data.y[mask].astype(np.int64)
    for name, overrides in variants.items():
        model = train_tree(data, arm_config(paths, arm, 0, **overrides))
        p = model.predict_proba(feature_matrix(data, model.config.columns, mask))
        out[name] = {
            "pr_auc": m.pr_auc(y, p),
            "log_loss": m.log_loss(y, p),
            "brier_skill": m.brier_skill(y, p),
            "mean_predicted": float(p.mean()),
            "observed_rate": float(y.mean()),
            "seconds": model.seconds,
            "best_iteration": model.best_iteration,
        }
        print(name, json.dumps(out[name], default=float), flush=True)
    _write_json(paths.reports / "validation" / "imbalance_study.json", out)
    return out


def tune(paths: DatasetPaths, data: ModelData, trials: int = 30) -> dict[str, Any]:
    """Optuna on the T3 feature set, optimising PR-AUC on validation A with a time-based split."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 50.0, log=True),
        }
        model = train_tree(
            data, TreeConfig("T3", FEATURE_SETS["T3"], params=params, rounds=400, early_stopping=30)
        )
        trial.set_user_attr("best_iteration", model.best_iteration)
        print(
            f"trial {trial.number}: pr_auc={model.val_a_pr_auc:.4f} t={model.seconds:.0f}s",
            flush=True,
        )
        return model.val_a_pr_auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials)
    best = {**study.best_params, "bagging_freq": 1}
    _write_json(paths.artifacts / "tuned_params.json", best)
    summary = {
        "best_value": study.best_value,
        "best_params": best,
        "trials": [{"number": t.number, "value": t.value, **t.params} for t in study.trials],
    }
    _write_json(paths.reports / "validation" / "tuning.json", summary)
    return summary


def calibration_study(
    paths: DatasetPaths, data: ModelData, arm: str = "T3", seed: int = 0
) -> dict[str, Any]:
    """Compare calibrators by cross-fitting inside validation B, then fit the winner on all of it.

    Test data plays no part in choosing the method. Validation B is split by time: earlier half to
    fit, later half to compare.
    """
    model = TreeModel.load(_model_dir(paths, arm, seed))
    mask = data.mask(VAL_B)
    raw = model.predict_proba(feature_matrix(data, model.config.columns, mask))
    y = data.y[mask].astype(np.int64)
    middle = len(y) // 2
    out: dict[str, Any] = {
        "validation_b": {"transactions": len(y), "positives": int(y.sum())},
        "raw_on_second_half": calibration_summary(y[middle:], raw[middle:]),
        "methods": {},
    }
    for method in ("platt", "beta", "isotonic"):
        fitted = Calibrator.fit(method, raw[:middle], y[:middle])  # type: ignore[arg-type]
        calibrated = fitted.transform(raw[middle:])
        out["methods"][method] = calibration_summary(y[middle:], calibrated)
        out["methods"][method]["top_k_unchanged"] = bool(
            m.pr_auc(y[middle:], calibrated) == m.pr_auc(y[middle:], raw[middle:])
        )
        out["methods"][method]["distinct_values"] = len(np.unique(calibrated))

    # Prefer the simplest method that isn't measurably worse on log loss.
    order = ["platt", "beta", "isotonic"]
    best_loss = min(out["methods"][k]["log_loss"] for k in order)
    chosen = next(k for k in order if out["methods"][k]["log_loss"] <= best_loss * 1.005)
    out["chosen"] = chosen
    final = Calibrator.fit(chosen, raw, y)  # type: ignore[arg-type]
    directory = _model_dir(paths, arm, seed)
    final.save(directory / "calibrator.json")
    _write_json(paths.reports / "validation" / f"calibration_{arm}.json", out)
    return out


def explain_global(paths: DatasetPaths, data: ModelData, arm: str = "T3", seed: int = 0) -> Any:
    """TreeSHAP importance on top-scoring validation transactions, plus stability across seeds."""
    tables: dict[int, list[tuple[str, float]]] = {}
    mask = data.mask(VAL_A)
    for current in SEEDS:
        directory = _model_dir(paths, arm, current)
        if not directory.exists():
            continue
        model = TreeModel.load(directory)
        matrix = feature_matrix(data, model.config.columns, mask)
        margin = model.predict_margin(matrix)
        top = np.argsort(-margin)[:5000]
        tables[current] = global_importance(
            model.config.columns, model.contributions(matrix[top])[:, :-1]
        )
    reference = dict(tables[seed])
    names = list(reference)
    rank_agreement = []
    for current, table in tables.items():
        if current == seed:
            continue
        other = dict(table)
        a = np.array([reference[n] for n in names])
        b = np.array([other[n] for n in names])
        rank_agreement.append(
            float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        )
    top_rows = [
        {
            "feature": name,
            "mean_abs_shap": value,
            "family": family(name),
            "description": describe(name),
        }
        for name, value in tables[seed][:25]
    ]
    out = {"arm": arm, "top_features": top_rows, "seed_rank_correlation": rank_agreement}
    _write_json(paths.reports / "validation" / f"shap_{arm}.json", out)
    return out


def drift_study(paths: DatasetPaths, data: ModelData) -> dict[str, Any]:
    """Run the injection scenarios against the training reference."""
    rng = np.random.default_rng(0)
    names = list(FEATURE_SETS["T3"])
    train_mask = data.partition == TRAIN
    sample = rng.choice(np.flatnonzero(train_mask), min(300_000, int(train_mask.sum())), False)

    def rows(indices: np.ndarray) -> np.ndarray:
        selected = np.zeros(len(data.y), dtype=bool)
        selected[indices] = True
        return data.features.matrix(names, selected)

    reference = rows(sample)
    monitor = DriftMonitor.from_matrix(names, reference)

    days = list(np.unique(data.day[train_mask]))[1:]
    daily = []
    for day in days:
        in_day = np.flatnonzero((data.day == day) & train_mask)
        chosen = rng.choice(in_day, min(100_000, len(in_day)), False)
        daily.append(rows(chosen))
    monitor.calibrate(daily)
    # All engine features, in engine order: this is what the live scorer compares against.
    monitor.save(_model_dir(paths, "T3", 0) / "drift_reference.json")

    validation = np.flatnonzero(data.partition == VAL_A)
    live = rows(rng.choice(validation, min(200_000, len(validation)), False))
    index = {n: i for i, n in enumerate(names)}

    def scenario(frame: np.ndarray) -> dict[str, Any]:
        alerts = monitor.alerts(frame)
        ranked = sorted(alerts.items(), key=lambda kv: -kv[1])
        return {"features_alerting": len(alerts), "top": [[k, round(v, 3)] for k, v in ranked[:6]]}

    scaled = live.copy()
    scaled[:, index["log_usd"]] += np.log(1.5)
    for name in names:
        if "_sum_" in name:
            scaled[:, index[name]] += np.log(1.5)
    broken = live.copy()
    broken[:, index["two_paths_24h"]] = np.nan
    constant = live.copy()
    constant[:, index["src_out_cnt_24h"]] = 0.0
    no_ach = live[live[:, index["payment_format"]] != 0.0]
    baseline_alerts = [len(monitor.alerts(day)) for day in daily]
    out = {
        "alert_levels_median": float(np.median(list(monitor.levels.values()))),
        "training_days_replayed_alerts": baseline_alerts,
        "scenarios": {
            "validation_window_no_injection": scenario(live),
            "amounts_times_1_5": scenario(scaled),
            "feature_goes_null": scenario(broken),
            "feature_goes_constant": scenario(constant),
            "payment_format_dropped": scenario(no_ach),
        },
    }
    _write_json(paths.reports / "validation" / "drift_scenarios.json", out)
    return out


def rule_agent_eval(
    paths: DatasetPaths, data: ModelData, code: int, arm: str = "T3", seed: int = 0
) -> dict[str, Any]:
    """Run what needs no LLM on real alerts: the rule baseline and the verifier mutation tests."""
    model = TreeModel.load(_model_dir(paths, arm, seed))
    calibrator = Calibrator.load(_model_dir(paths, arm, seed) / "calibrator.json")
    mask = data.mask(code)
    raw = _raw_scores(data, model, mask)
    probability = np.full(len(raw), 0.0)
    probability[mask] = calibrator.transform(raw[mask])
    scores = np.nan_to_num(raw, nan=-1.0)
    cases, counts = build_cases(
        data, scores, probability, mask, model_version=f"{arm}-{model.model_hash()}"
    )
    store = MemoryStore.from_parquet(paths.transactions, load_fx(paths.fx))

    from unravel.agent.scripted import ScriptedClient

    results = run_agent(cases, store, ScriptedClient())
    out = {
        "strata_available": counts,
        "rule_baseline": rule_baseline(cases, store),
        "scripted_template": summarise(results, ScriptedClient.name),
        "verifier_mutations": verifier_mutation_test(results, store),
    }
    if code == TEST:
        _write_json(paths.reports / "test" / "agent_rules.json", out)
    else:
        _write_json(paths.reports / "validation" / "agent_rules_validation.json", out)
    return out


def final_evaluation(paths: DatasetPaths, data: ModelData, arm: str, reason: str) -> dict[str, Any]:
    """Evaluate an arm's seeds on test, using thresholds and calibration fixed on validation.

    Every call leaves a permanent ledger entry.
    """
    ledger = paths.reports / LEDGER_NAME
    test_mask = data.mask(TEST)
    val_a = data.mask(VAL_A)
    val_b = data.mask(VAL_B)
    runs = []
    for seed in SEEDS:
        directory = _model_dir(paths, arm, seed)
        if not directory.exists():
            continue
        model = TreeModel.load(directory)
        record_test_evaluation(
            ledger,
            dataset=paths.name,
            arm=arm,
            protocol="causal-tree",
            model_hash=model.model_hash(),
            reason=reason,
        )
        scores = _raw_scores(data, model, val_a | test_mask)
        threshold = m.best_f1_threshold(data.y[val_a], scores[val_a])
        report = evaluate_scores(data, scores, test_mask, threshold, n_boot=50, seed=seed)
        report["seed"] = seed
        # Validation days warm up the live threshold, then only test days are counted.
        history = (val_a | val_b | test_mask) & full_volume_mask(data)
        all_scores = _raw_scores(data, model, history)
        report["online_regret_full_volume"] = m.online_regret(
            data.alert_frame(all_scores, history),
            100,
            evaluate_from_day=int(data.day[test_mask].min()),
        )

        raw_b = model.predict_proba(feature_matrix(data, model.config.columns, val_b))
        calibration_method = "platt"
        chosen_path = paths.reports / "validation" / f"calibration_{arm}.json"
        if chosen_path.exists():
            calibration_method = json.loads(chosen_path.read_text())["chosen"]
        calibrator = Calibrator.fit(calibration_method, raw_b, data.y[val_b].astype(np.int64))  # type: ignore[arg-type]
        y_test = data.y[test_mask].astype(np.int64)
        report["calibration_raw"] = calibration_summary(y_test, scores[test_mask])
        report["calibration_calibrated"] = calibration_summary(
            y_test, calibrator.transform(scores[test_mask])
        )
        report["calibration_method"] = calibration_method
        runs.append(report)
        print(
            f"{arm} seed={seed} test pr_auc={report['pr_auc']:.4f} "
            f"f1={report['minority_f1']['f1']:.4f}",
            flush=True,
        )
    f1 = [r["minority_f1"]["f1"] for r in runs]
    pr = [r["pr_auc"] for r in runs]
    summary = {
        "arm": arm,
        "reason": reason,
        "test_evaluations_so_far": len(read_ledger(ledger)),
        "f1_mean": float(np.mean(f1)),
        "f1_std": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
        "pr_auc_mean": float(np.mean(pr)),
        "pr_auc_std": float(np.std(pr, ddof=1)) if len(pr) > 1 else 0.0,
        "runs": runs,
    }
    _write_json(paths.reports / "test" / f"{arm}.json", summary)
    return summary


def agent_llm_eval(
    paths: DatasetPaths, data: ModelData, code: int, *, budget_usd: float, arm: str = "T3"
) -> dict[str, Any]:
    """Evaluate the real LLM investigator and judge. This spends money, up to ``budget_usd``."""
    from unravel.agent.llm import AnthropicClient, SpendGuard
    from unravel.agent_eval.judge import citation_support_rate, export_gold_template, run_judge

    model = TreeModel.load(_model_dir(paths, arm, 0))
    calibrator = Calibrator.load(_model_dir(paths, arm, 0) / "calibrator.json")
    mask = data.mask(code)
    raw = _raw_scores(data, model, mask)
    probability = np.zeros(len(raw))
    probability[mask] = calibrator.transform(raw[mask])
    cases, counts = build_cases(
        data, np.nan_to_num(raw, nan=-1.0), probability, mask, model_version=arm
    )
    store = MemoryStore.from_parquet(paths.transactions, load_fx(paths.fx))
    guard = SpendGuard(budget_usd)
    client = AnthropicClient(guard=guard)
    results = run_agent(cases, store, client)
    scores = [run_judge(r.draft, store, client) for r in results if r.draft is not None]
    # These real drafts are what a human rates, to calibrate the judge.
    export_gold_template(
        [(f"case-{i:03d}", r.draft) for i, r in enumerate(results) if r.draft is not None],
        store,
        paths.reports / "gold_sheet.csv",
    )
    out = {
        "strata_available": counts,
        "agent": summarise(results, client.name),
        "rule_baseline": rule_baseline(cases, store),
        "citation_support_rate": citation_support_rate(scores),
        "judge_scores": [s.model_dump() for s in scores],
        "total_cost_usd": guard.usage.cost(client.name),
    }
    name = "agent_llm.json" if code == TEST else "agent_llm_validation.json"
    folder = "test" if code == TEST else "validation"
    _write_json(paths.reports / folder / name, out)
    return out


def gold_sheet(paths: DatasetPaths, data: ModelData, code: int, arm: str = "T3") -> Path:
    """Export template drafts as a blind sheet, to try the rubric before spending on an LLM.

    For real judge calibration, use the sheet that ``agent_llm_eval`` writes from LLM drafts.
    """
    from unravel.agent.scripted import ScriptedClient
    from unravel.agent_eval.judge import export_gold_template

    model = TreeModel.load(_model_dir(paths, arm, 0))
    mask = data.mask(code)
    raw = _raw_scores(data, model, mask)
    cases, _ = build_cases(data, np.nan_to_num(raw, nan=-1.0), np.nan_to_num(raw), mask)
    store = MemoryStore.from_parquet(paths.transactions, load_fx(paths.fx))
    results = run_agent(cases[:60], store, ScriptedClient())
    drafts = [(f"case-{i:03d}", r.draft) for i, r in enumerate(results) if r.draft is not None]
    path = paths.reports / "gold_sheet.csv"
    export_gold_template(drafts, store, path)
    return path


def gnn_final_evaluation(
    paths: DatasetPaths, data: ModelData, arm: str, seed: int, reason: str
) -> dict[str, Any]:
    """Score a saved GNN on test using its validation threshold. Each call leaves a ledger entry."""
    import torch

    from unravel.models.gnn import (
        GnnConfig,
        MultiPna,
        NeighbourSampler,
        degree_histogram,
        load_edges,
    )
    from unravel.models.gnn import predict as gnn_predict

    directory = _model_dir(paths, arm, seed)
    meta = json.loads((directory / "meta.json").read_text())
    settings = dict(meta["config"])
    settings["fanout"] = tuple(settings["fanout"])
    settings["extra_columns"] = tuple(settings["extra_columns"])
    config = GnnConfig(**settings)
    table = load_edges(paths, config.extra_columns)
    train = table.partition == TRAIN
    sampler = NeighbourSampler(table, config, len(table.src), seed + 2)
    targets = np.flatnonzero(table.partition == TEST)
    probe = sampler.batch(targets[:4])
    model = MultiPna(
        probe.edge_attr.shape[1],
        probe.target_attr.shape[1],
        degree_histogram(table, train),
        config,
    )
    state = torch.load(directory / "model.pt", weights_only=True)
    model.load_state_dict(state)
    record_test_evaluation(
        paths.reports / LEDGER_NAME,
        dataset=paths.name,
        arm=arm,
        protocol=config.protocol,
        model_hash=f"seed{seed}",
        reason=reason,
    )
    validation = np.load(directory / "validation_scores.npz")
    threshold = m.best_f1_threshold(data.y[data.partition == VAL_A], validation["val_a"])
    scores = np.full(len(data.y), np.nan)
    scores[data.mask(TEST)] = gnn_predict(model, sampler, targets)
    report = evaluate_scores(data, scores, data.mask(TEST), threshold, n_boot=30, seed=seed)
    report.update(arm=arm, protocol=config.protocol, seed=seed, device="cpu")
    _write_json(paths.reports / "test" / f"{arm}_{config.protocol}_seed{seed}.json", report)
    return report


def order_sensitivity(paths: DatasetPaths) -> dict[str, Any]:
    """How much can same-minute order matter? It compares vectors under two tie-break orders.

    Comparing canonical order against file order would need labels to interpret, so this reports
    how many feature values change: an upper bound on what order could possibly leak.
    """
    from unravel.data.schema import TX_COLUMNS
    from unravel.features.engine import FeatureEngine, TxEvent

    tx = pl.read_parquet(paths.transactions).select(TX_COLUMNS).head(300_000)
    fx = load_fx(paths.fx)
    forward = FeatureEngine(fx)
    a = {}
    for row in tx.iter_rows():
        event = TxEvent(*row)
        a[event.tx_id] = forward.process(event)
    backward = FeatureEngine(fx)
    # Same minutes, reversed inside each minute.
    reordered = tx.with_row_index("i").sort(["minute", "i"], descending=[False, True]).drop("i")
    changed = total = 0
    columns_changed = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    for row in reordered.iter_rows():
        event = TxEvent(*row)
        v = np.array(backward.process(event))
        o = np.array(a[event.tx_id])
        differs = ~((v == o) | (np.isnan(v) & np.isnan(o)))
        columns_changed += differs
        changed += bool(differs.any())
        total += 1
    out = {
        "events": total,
        "vectors_changed_share": changed / total,
        "most_affected_features": [
            [FEATURE_NAMES[i], float(columns_changed[i] / total)]
            for i in np.argsort(-columns_changed)[:8]
        ],
        "graph_features": GRAPH_FEATURES,
    }
    _write_json(paths.reports / "validation" / "order_sensitivity.json", out)
    return out


def score_table(
    paths: DatasetPaths, arm: str, seed: int, partitions: tuple[int, ...]
) -> pl.DataFrame:
    """Raw scores for later analysis, without labels."""
    data = load_model_data(paths)
    model = TreeModel.load(_model_dir(paths, arm, seed))
    mask = data.mask(*partitions)
    scores = model.predict_proba(feature_matrix(data, model.config.columns, mask))
    return pl.DataFrame({"tx_id": data.tx_id[mask], "score": scores})


__all__ = ["TRAIN", "VAL_A", "VAL_B"]
