"""Dependency-light metrics for masked multi-label and intensity targets."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=float)
    scores = np.asarray(scores, dtype=float)
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = targets[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(len(targets), 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            selected = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if not selected.any():
            continue
        result += selected.sum() / total * abs(targets[selected].mean() - probabilities[selected].mean())
    return float(result)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_rank = _rank(np.asarray(left, dtype=float))
    right_rank = _rank(np.asarray(right, dtype=float))
    if left_rank.std() == 0 or right_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    support_threshold: int = 10,
    top_k: int = 5,
) -> Dict[str, object]:
    targets = np.asarray(targets, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.shape != probabilities.shape or targets.ndim != 2:
        raise ValueError("Targets and probabilities must have the same two-dimensional shape")
    valid = np.isfinite(targets) if mask is None else np.asarray(mask, dtype=bool)
    per_label_ap = []
    supports = []
    briers = []
    eces = []
    for label in range(targets.shape[1]):
        selected = valid[:, label]
        y = targets[selected, label]
        p = probabilities[selected, label]
        supports.append(int(y.sum()))
        per_label_ap.append(average_precision(y, p) if len(y) else float("nan"))
        briers.append(float(np.mean((p - y) ** 2)) if len(y) else float("nan"))
        eces.append(expected_calibration_error(y, p) if len(y) else float("nan"))
    flattened = valid.ravel()
    micro_ap = average_precision(targets.ravel()[flattened], probabilities.ravel()[flattened])
    supported = np.asarray(supports) >= support_threshold
    macro_ap = float(np.nanmean(np.asarray(per_label_ap)[supported])) if supported.any() else float("nan")
    all_supported = np.asarray(supports) > 0
    macro_ap_all = float(np.nanmean(np.asarray(per_label_ap)[all_supported])) if all_supported.any() else float("nan")

    recalls = []
    precisions = []
    for row in range(targets.shape[0]):
        assessed = valid[row]
        positives = (targets[row] == 1) & assessed
        available = np.where(assessed)[0]
        if not len(available):
            continue
        selected = available[np.argsort(-probabilities[row, available])[: min(top_k, len(available))]]
        hits = int(positives[selected].sum())
        if positives.sum():
            recalls.append(hits / positives.sum())
        precisions.append(hits / len(selected))
    return {
        "micro_average_precision": micro_ap,
        "macro_average_precision_supported": macro_ap,
        "macro_average_precision_all_labels": macro_ap_all,
        "support_threshold": support_threshold,
        "per_label_average_precision": per_label_ap,
        "positive_support": supports,
        "mean_brier": float(np.nanmean(briers)),
        "mean_label_ece": float(np.nanmean(eces)),
        f"recall_at_{top_k}": float(np.mean(recalls)) if recalls else float("nan"),
        f"precision_at_{top_k}": float(np.mean(precisions)) if precisions else float("nan"),
    }


def intensity_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    valid = np.isfinite(targets) if mask is None else np.asarray(mask, dtype=bool)
    y = targets[valid]
    p = predictions[valid]
    if not len(y):
        return {"masked_mae": float("nan"), "spearman": float("nan"), "count": 0}
    return {
        "masked_mae": float(np.mean(np.abs(y - p))),
        "spearman": spearman_correlation(y, p),
        "count": int(len(y)),
    }
