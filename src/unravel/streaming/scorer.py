"""Score feature messages, calibrate, and feed the capacity queue. Kafka stays outside this class."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unravel.alerts.queue import CapacityQueue, QueuedAlert
from unravel.explain.reasons import reason_codes
from unravel.features.engine import FEATURE_NAMES
from unravel.models.calibration import Calibrator
from unravel.models.tree import TreeModel
from unravel.streaming.messages import decode_vector


@dataclass(frozen=True)
class ScoredTransaction:
    tx_id: str
    minute: int
    raw_score: float
    probability: float
    alert: QueuedAlert | None
    reasons: list[dict[str, object]] | None


class Scorer:
    """Load a versioned model plus calibrator, and keep a fixed mapping from engine to model columns."""

    def __init__(
        self,
        model: TreeModel,
        calibrator: Calibrator,
        queue: CapacityQueue,
        top_reasons: int = 5,
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.queue = queue
        self.top_reasons = top_reasons
        self.columns = model.config.columns
        lookup = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self.indices = np.array([lookup[c] for c in self.columns], dtype=np.int64)
        self.last_vectors = np.zeros((0, len(FEATURE_NAMES)))

    @classmethod
    def from_directory(cls, directory: Path, capacity: int = 100) -> Scorer:
        model = TreeModel.load(directory)
        calibrator = Calibrator.load(directory / "calibrator.json")
        policy_path = directory / "policy.json"
        initial = None
        if policy_path.exists():
            initial = json.loads(policy_path.read_text()).get("initial_threshold")
        return cls(model, calibrator, CapacityQueue(capacity, initial_threshold=initial))

    def score_batch(self, payloads: list[bytes]) -> list[ScoredTransaction]:
        if not payloads:
            return []
        decoded = [decode_vector(p) for p in payloads]
        full = np.array([d[2] for d in decoded], dtype=np.float64)
        self.last_vectors = full
        matrix = full[:, self.indices].astype(np.float32)
        raw = self.model.predict_proba(matrix)
        calibrated = self.calibrator.transform(raw)
        results = []
        admitted_rows: list[int] = []
        for i, (tx_id, minute, _) in enumerate(decoded):
            alert = self.queue.offer(
                tx_id=tx_id,
                minute=minute,
                raw_score=float(raw[i]),
                probability=float(calibrated[i]),
            )
            if alert is not None:
                admitted_rows.append(i)
            results.append(
                ScoredTransaction(tx_id, minute, float(raw[i]), float(calibrated[i]), alert, None)
            )
        if admitted_rows:
            contributions = self.model.contributions(matrix[admitted_rows])[:, :-1]
            for row, values, shap in zip(
                admitted_rows, matrix[admitted_rows], contributions, strict=True
            ):
                reasons = [
                    r.as_dict()
                    for r in reason_codes(self.columns, values, shap, top=self.top_reasons)
                ]
                current = results[row]
                results[row] = ScoredTransaction(
                    current.tx_id,
                    current.minute,
                    current.raw_score,
                    current.probability,
                    current.alert,
                    reasons,
                )
        return results
