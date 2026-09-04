"""Calibration-partition Platt scaling with prevalence-tier fallback."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_platt(logits: np.ndarray, targets: np.ndarray, l2: float = 1e-3) -> Tuple[float, float]:
    x = np.asarray(logits, dtype=float)
    y = np.asarray(targets, dtype=float)
    a, b = 1.0, 0.0
    for _ in range(100):
        p = _sigmoid(a * x + b)
        residual = p - y
        weight = p * (1.0 - p)
        gradient = np.array([(residual * x).sum() + l2 * a, residual.sum()])
        hessian = np.array(
            [
                [(weight * x * x).sum() + l2, (weight * x).sum()],
                [(weight * x).sum(), weight.sum() + l2],
            ]
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        a -= float(step[0])
        b -= float(step[1])
        if np.max(np.abs(step)) < 1e-7:
            break
    return float(np.clip(a, 0.05, 20.0)), float(np.clip(b, -20.0, 20.0))


def _best_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    best = (0.0, 0.5)
    for threshold in np.linspace(0.02, 0.98, 97):
        predicted = probabilities >= threshold
        tp = float(((targets == 1) & predicted).sum())
        fp = float(((targets == 0) & predicted).sum())
        fn = float(((targets == 1) & ~predicted).sum())
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else 2 * tp / denominator
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def _tier(prevalence: float) -> str:
    if prevalence < 0.05:
        return "rare"
    if prevalence < 0.20:
        return "medium"
    return "common"


@dataclass
class CalibrationBundle:
    label_names: Tuple[str, ...]
    slopes: Tuple[float, ...]
    intercepts: Tuple[float, ...]
    thresholds: Tuple[float, ...]
    methods: Tuple[str, ...]
    calibration_version: str = "platt-tier-v1"

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        targets: np.ndarray,
        label_names: Sequence[str],
        mask: Optional[np.ndarray] = None,
        minimum_support: int = 50,
    ) -> "CalibrationBundle":
        logits = np.asarray(logits, dtype=float)
        targets = np.asarray(targets, dtype=float)
        valid = np.isfinite(targets) if mask is None else np.asarray(mask, dtype=bool)
        if logits.shape != targets.shape or logits.shape[1] != len(label_names):
            raise ValueError("Calibration arrays do not match label names")

        tier_values: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
            "rare": [],
            "medium": [],
            "common": [],
        }
        prevalence = []
        for label in range(targets.shape[1]):
            selected = valid[:, label]
            y = targets[selected, label]
            rate = float(y.mean()) if len(y) else 0.0
            prevalence.append(rate)
            if len(y) and y.sum() > 0 and (1 - y).sum() > 0:
                tier_values[_tier(rate)].append((logits[selected, label], y))
        tier_parameters = {}
        for name, values in tier_values.items():
            if values:
                tier_parameters[name] = _fit_platt(
                    np.concatenate([value[0] for value in values]),
                    np.concatenate([value[1] for value in values]),
                )
            else:
                tier_parameters[name] = (1.0, 0.0)

        slopes: List[float] = []
        intercepts: List[float] = []
        thresholds: List[float] = []
        methods: List[str] = []
        for label in range(targets.shape[1]):
            selected = valid[:, label]
            y = targets[selected, label]
            x = logits[selected, label]
            positives = int(y.sum())
            negatives = int((1 - y).sum())
            if positives >= minimum_support and negatives >= minimum_support:
                a, b = _fit_platt(x, y)
                method = "per_label_platt"
            else:
                a, b = tier_parameters[_tier(prevalence[label])]
                method = f"{_tier(prevalence[label])}_tier_platt"
            probability = _sigmoid(a * x + b)
            threshold = _best_f1_threshold(y, probability) if positives and negatives else 0.5
            slopes.append(a)
            intercepts.append(b)
            thresholds.append(threshold)
            methods.append(method)
        return cls(
            tuple(str(name) for name in label_names),
            tuple(slopes),
            tuple(intercepts),
            tuple(thresholds),
            tuple(methods),
        )

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=float)
        if values.shape[-1] != len(self.label_names):
            raise ValueError("Logit dimension does not match calibration labels")
        return _sigmoid(values * np.asarray(self.slopes) + np.asarray(self.intercepts))

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CalibrationBundle":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            tuple(payload["label_names"]),
            tuple(payload["slopes"]),
            tuple(payload["intercepts"]),
            tuple(payload["thresholds"]),
            tuple(payload["methods"]),
            payload["calibration_version"],
        )
