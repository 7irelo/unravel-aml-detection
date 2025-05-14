"""Probability calibration, fitted on validation B only.

Isotonic regression produces plateaus, so many transactions end up with identical probabilities.
Sorting on those ties would make precision@k depend on arbitrary order. ``Calibrator`` therefore
keeps the raw score for ranking and uses calibrated values only as probabilities.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

Method = Literal["platt", "beta", "isotonic"]
_EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def correct_for_downsampling(p: np.ndarray, keep_rate: float) -> np.ndarray:
    """Undo negative downsampling: ``p = βp_s / (1 - p_s + βp_s)``, where β is the keep rate."""
    return keep_rate * p / (1.0 - p + keep_rate * p)


def _fit_logistic(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit an L2-free logistic regression by L-BFGS on the log loss."""

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = design @ w
        # log(1 + exp(z)) written so large z doesn't overflow.
        loss = np.logaddexp(0.0, z) - y * z
        gradient = design.T @ (sigmoid(z) - y)
        return float(loss.sum()), gradient

    start = np.zeros(design.shape[1])
    start[-1] = math.log(max(y.mean(), 1e-9) / max(1 - y.mean(), 1e-9))
    return minimize(objective, start, jac=True, method="L-BFGS-B").x


@dataclass
class Calibrator:
    """Monotonic mapping from raw model probability to calibrated probability."""

    method: Method
    params: list[float]
    iso_x: list[float] | None = None
    iso_y: list[float] | None = None

    @classmethod
    def fit(cls, method: Method, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if y.sum() == 0:
            raise ValueError("cannot calibrate on a window with no positives")
        if method == "platt":
            design = np.c_[logit(raw), np.ones(len(raw))]
            return cls(method, _fit_logistic(design, y).tolist())
        if method == "beta":
            clipped = np.clip(raw, _EPS, 1 - _EPS)
            design = np.c_[np.log(clipped), -np.log(1 - clipped), np.ones(len(raw))]
            # Bound the shape parameters to keep the map monotonic (Kull et al., 2017).
            weights = _fit_logistic(design, y)
            weights[0] = max(weights[0], 0.0)
            weights[1] = max(weights[1], 0.0)
            return cls(method, weights.tolist())
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(raw, y)
        return cls(
            method,
            [],
            iso_x=model.X_thresholds_.tolist(),
            iso_y=model.y_thresholds_.tolist(),
        )

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self.method == "platt":
            a, b = self.params
            return sigmoid(a * logit(raw) + b)
        if self.method == "beta":
            a, b, c = self.params
            clipped = np.clip(raw, _EPS, 1 - _EPS)
            return sigmoid(a * np.log(clipped) - b * np.log(1 - clipped) + c)
        if self.iso_x is None or self.iso_y is None:
            raise ValueError("isotonic calibrator is missing its thresholds")
        return np.interp(raw, self.iso_x, self.iso_y)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__))

    @classmethod
    def load(cls, path: Path) -> Calibrator:
        data: dict[str, Any] = json.loads(path.read_text())
        return cls(**data)
