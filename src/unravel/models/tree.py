"""LightGBM and XGBoost arms. Fit on train, early-stop on validation A, never touch test here."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from unravel.eval.metrics import pr_auc
from unravel.eval.splits import TRAIN, VAL_A
from unravel.features.engine import CATEGORICAL_FEATURES
from unravel.models.calibration import correct_for_downsampling
from unravel.models.dataset import ModelData

Imbalance = Literal["none", "scale_pos_weight", "downsample"]

DEFAULT_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
}


@dataclass(frozen=True)
class TreeConfig:
    arm: str
    columns: list[str]
    imbalance: Imbalance = "downsample"
    keep_rate: float = 0.05
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    seed: int = 0
    rounds: int = 2000
    early_stopping: int = 100
    framework: Literal["lightgbm", "xgboost"] = "lightgbm"


@dataclass
class TreeModel:
    config: TreeConfig
    booster: Any
    best_iteration: int
    seconds: float
    val_a_pr_auc: float

    def predict_margin(self, matrix: np.ndarray) -> np.ndarray:
        if self.config.framework == "lightgbm":
            return self.booster.predict(
                matrix, num_iteration=self.best_iteration, raw_score=True, num_threads=8
            )
        dmatrix = xgb.DMatrix(matrix)
        return self.booster.predict(
            dmatrix, output_margin=True, iteration_range=(0, self.best_iteration)
        )

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        """Raw model probability, corrected to the true prior if negatives were downsampled."""
        p = 1.0 / (1.0 + np.exp(-self.predict_margin(matrix)))
        if self.config.imbalance == "downsample":
            return correct_for_downsampling(p, self.config.keep_rate)
        return p

    def contributions(self, matrix: np.ndarray) -> np.ndarray:
        """TreeSHAP values on the margin, one row per input, with the base value last."""
        if self.config.framework == "lightgbm":
            return self.booster.predict(
                matrix, num_iteration=self.best_iteration, pred_contrib=True, num_threads=8
            )
        return self.booster.predict(
            xgb.DMatrix(matrix), pred_contribs=True, iteration_range=(0, self.best_iteration)
        )

    def model_hash(self) -> str:
        text = (
            self.booster.model_to_string()
            if self.config.framework == "lightgbm"
            else bytes(self.booster.save_raw("json")).decode()
        )
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.config.framework == "lightgbm":
            self.booster.save_model(str(directory / "model.txt"))
        else:
            self.booster.save_model(str(directory / "model.json"))
        meta = {
            "arm": self.config.arm,
            "columns": self.config.columns,
            "imbalance": self.config.imbalance,
            "keep_rate": self.config.keep_rate,
            "params": self.config.params,
            "seed": self.config.seed,
            "framework": self.config.framework,
            "best_iteration": self.best_iteration,
            "seconds": self.seconds,
            "val_a_pr_auc": self.val_a_pr_auc,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> TreeModel:
        meta = json.loads((directory / "meta.json").read_text())
        config = TreeConfig(
            arm=meta["arm"],
            columns=meta["columns"],
            imbalance=meta["imbalance"],
            keep_rate=meta["keep_rate"],
            params=meta["params"],
            seed=meta["seed"],
            framework=meta["framework"],
        )
        if config.framework == "lightgbm":
            booster: Any = lgb.Booster(model_file=str(directory / "model.txt"))
        else:
            booster = xgb.Booster()
            booster.load_model(str(directory / "model.json"))
        return cls(config, booster, meta["best_iteration"], meta["seconds"], meta["val_a_pr_auc"])


def feature_matrix(data: ModelData, columns: list[str], mask: np.ndarray) -> np.ndarray:
    return data.features.matrix(columns, mask)


def _training_rows(y: np.ndarray, config: TreeConfig) -> tuple[np.ndarray, float]:
    """Return row indices and ``scale_pos_weight`` for the chosen imbalance strategy."""
    rng = np.random.default_rng(config.seed)
    if config.imbalance == "downsample":
        keep = (y == 1) | (rng.random(len(y)) < config.keep_rate)
        return np.flatnonzero(keep), 1.0
    if config.imbalance == "scale_pos_weight":
        return np.arange(len(y)), float((y == 0).sum() / max((y == 1).sum(), 1))
    return np.arange(len(y)), 1.0


def train_tree(data: ModelData, config: TreeConfig) -> TreeModel:
    started = time.perf_counter()
    train_mask = data.partition == TRAIN
    val_mask = data.partition == VAL_A
    x_val = feature_matrix(data, config.columns, val_mask)
    y_val = data.y[val_mask]
    # Choose rows from labels first, so unused negatives are never read from disk.
    train_indices = np.flatnonzero(train_mask)
    y_all = data.y[train_mask]
    rows, spw = _training_rows(y_all, config)
    selected = np.zeros(len(data.y), dtype=bool)
    selected[train_indices[rows]] = True
    x_train = feature_matrix(data, config.columns, selected)
    y_train = data.y[selected]

    if config.framework == "lightgbm":
        categorical = [c for c in CATEGORICAL_FEATURES if c in config.columns]
        train_set = lgb.Dataset(
            x_train,
            label=y_train,
            feature_name=config.columns,
            categorical_feature=categorical,
            free_raw_data=True,
        )
        val_set = lgb.Dataset(x_val, label=y_val, reference=train_set, free_raw_data=True)
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "scale_pos_weight": spw,
            "seed": config.seed,
            "verbosity": -1,
            "num_threads": 8,
            "feature_pre_filter": False,
            **config.params,
        }
        booster: Any = lgb.train(
            params,
            train_set,
            num_boost_round=config.rounds,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(config.early_stopping, verbose=False)],
        )
        best = int(booster.best_iteration or config.rounds)
    else:
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_val = xgb.DMatrix(x_val, label=y_val)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "eta": config.params.get("learning_rate", 0.05),
            "max_depth": config.params.get("max_depth", 8),
            "subsample": config.params.get("bagging_fraction", 0.8),
            "colsample_bytree": config.params.get("feature_fraction", 0.7),
            "lambda": config.params.get("lambda_l2", 1.0),
            "scale_pos_weight": spw,
            "tree_method": "hist",
            "seed": config.seed,
            "nthread": 8,
        }
        booster = xgb.train(
            params,
            d_train,
            num_boost_round=config.rounds,
            evals=[(d_val, "val")],
            early_stopping_rounds=config.early_stopping,
            verbose_eval=False,
        )
        best = int(booster.best_iteration) + 1

    model = TreeModel(config, booster, best, 0.0, 0.0)
    model.val_a_pr_auc = pr_auc(y_val, model.predict_margin(x_val))
    model.seconds = time.perf_counter() - started
    return model
