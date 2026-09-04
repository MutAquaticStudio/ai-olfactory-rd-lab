"""Calibration-aware target matching for candidate design."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence, Tuple

import numpy as np


SUPPORTED_POSITIVES = 50
LIMITED_POSITIVES = 10
DEFAULT_TARGET_FLOOR = 0.30
DEFAULT_FIT_FLOOR = 0.40
RELAXATION_STEP = 0.05


class DescriptorMaturity(str, Enum):
    SUPPORTED = "SUPPORTED"
    LIMITED_EVIDENCE = "LIMITED_EVIDENCE"
    INSUFFICIENT = "INSUFFICIENT"


class TargetMatchTier(str, Enum):
    STRICT = "STRICT"
    RELAXED = "RELAXED"


@dataclass(frozen=True)
class DescriptorEvidence:
    name: str
    positive_support: int
    assessed_negative_support: int
    maturity: DescriptorMaturity
    decision_threshold: float
    calibration_method: str


@dataclass(frozen=True)
class TargetEvidence:
    name: str
    probability: float
    uncertainty: float
    conservative_probability: float
    maturity: DescriptorMaturity
    requested_floor: float
    applied_floor: float
    passed_requested_floor: bool
    passed_applied_floor: bool


@dataclass(frozen=True)
class TargetMatch:
    target_fit: float
    robust_target_fit: float
    requested_fit_floor: float
    applied_fit_floor: float
    relaxation_factor: float
    tier: TargetMatchTier
    met_requested_gate: bool
    calibrated: bool
    uses_absolute_probability_gate: bool
    targets: Tuple[TargetEvidence, ...]


def maturity_from_support(
    positive_support: int,
    assessed_negative_support: int,
) -> DescriptorMaturity:
    """Classify a descriptor without treating unassessed rows as negatives."""
    if (
        positive_support >= SUPPORTED_POSITIVES
        and assessed_negative_support >= SUPPORTED_POSITIVES
    ):
        return DescriptorMaturity.SUPPORTED
    if positive_support >= LIMITED_POSITIVES:
        return DescriptorMaturity.LIMITED_EVIDENCE
    return DescriptorMaturity.INSUFFICIENT


def descriptor_evidence(
    names: Sequence[str],
    positive_support: Sequence[int],
    assessed_negative_support: Sequence[int] | None = None,
    decision_thresholds: Sequence[float] | None = None,
    calibration_methods: Sequence[str] | None = None,
) -> Tuple[DescriptorEvidence, ...]:
    count = len(names)
    negatives = assessed_negative_support or [0] * count
    thresholds = decision_thresholds or [DEFAULT_TARGET_FLOOR] * count
    methods = calibration_methods or ["uncalibrated"] * count
    if not all(len(values) == count for values in (positive_support, negatives, thresholds, methods)):
        raise ValueError("Descriptor evidence arrays must align with label names")
    records = []
    for name, positives, negative_count, threshold, method in zip(
        names, positive_support, negatives, thresholds, methods
    ):
        maturity = maturity_from_support(int(positives), int(negative_count))
        floor = (
            DEFAULT_TARGET_FLOOR
            if maturity is DescriptorMaturity.SUPPORTED
            else float(np.clip(threshold, 0.02, 0.98))
        )
        records.append(
            DescriptorEvidence(
                str(name),
                int(positives),
                int(negative_count),
                maturity,
                floor,
                str(method),
            )
        )
    return tuple(records)


def conservative_probability(probability: float, uncertainty: float) -> float:
    uncertainty_value = 0.0 if not np.isfinite(uncertainty) else float(uncertainty)
    return float(np.clip(float(probability) - 1.64 * uncertainty_value, 0.0, 1.0))


def _geometric_mean(values: Iterable[float]) -> float:
    array = np.clip(np.asarray(tuple(values), dtype=float), 1e-12, 1.0)
    if not array.size:
        raise ValueError("At least one target probability is required")
    return float(np.exp(np.log(array).mean()))


def evaluate_target_match(
    probabilities: Sequence[float],
    uncertainties: Sequence[float],
    evidence: Sequence[DescriptorEvidence],
    *,
    relaxation_factor: float = 1.0,
    calibrated: bool,
) -> TargetMatch:
    if not (len(probabilities) == len(uncertainties) == len(evidence)):
        raise ValueError("Target score arrays must have the same length")
    if not evidence:
        raise ValueError("At least one target descriptor is required")
    if any(item.maturity is DescriptorMaturity.INSUFFICIENT for item in evidence):
        raise ValueError("Insufficient-evidence descriptors cannot be target conditions")
    factor = float(np.clip(relaxation_factor, 0.0, 1.0))
    conservative = [
        conservative_probability(probability, uncertainty)
        for probability, uncertainty in zip(probabilities, uncertainties)
    ]
    mean_fit = _geometric_mean(probabilities)
    robust_fit = _geometric_mean(conservative)
    absolute_gate = all(
        item.maturity is DescriptorMaturity.SUPPORTED for item in evidence
    )
    requested_fit_floor = (
        DEFAULT_FIT_FLOOR
        if absolute_gate
        else _geometric_mean(item.decision_threshold for item in evidence)
    )
    targets = tuple(
        TargetEvidence(
            name=item.name,
            probability=float(probability),
            uncertainty=0.0 if not np.isfinite(uncertainty) else float(uncertainty),
            conservative_probability=value,
            maturity=item.maturity,
            requested_floor=item.decision_threshold,
            applied_floor=item.decision_threshold * factor,
            passed_requested_floor=value >= item.decision_threshold,
            passed_applied_floor=value >= item.decision_threshold * factor,
        )
        for probability, uncertainty, value, item in zip(
            probabilities, uncertainties, conservative, evidence
        )
    )
    # A raw sigmoid score is not an absolute probability claim.  The requested
    # gate can only be declared met after a held-out calibration artifact has
    # been loaded; legacy/uncalibrated models always remain transparent RELAXED
    # matches even when their numeric score happens to cross the same floor.
    requested = bool(calibrated) and all(item.passed_requested_floor for item in targets) and (
        robust_fit >= requested_fit_floor
    )
    return TargetMatch(
        target_fit=mean_fit,
        robust_target_fit=robust_fit,
        requested_fit_floor=requested_fit_floor,
        applied_fit_floor=requested_fit_floor * factor,
        relaxation_factor=factor,
        tier=TargetMatchTier.STRICT if requested else TargetMatchTier.RELAXED,
        met_requested_gate=requested,
        calibrated=bool(calibrated),
        uses_absolute_probability_gate=absolute_gate,
        targets=targets,
    )


def select_target_matches(
    probability_rows: np.ndarray,
    uncertainty_rows: np.ndarray,
    evidence: Sequence[DescriptorEvidence],
    *,
    count: int = 3,
    calibrated: bool,
) -> Tuple[Tuple[int, TargetMatch], ...]:
    """Select up to ``count`` rows, relaxing declared gates transparently."""
    probabilities = np.asarray(probability_rows, dtype=float)
    uncertainties = np.asarray(uncertainty_rows, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != uncertainties.shape:
        raise ValueError("Probability and uncertainty matrices must align")
    if probabilities.shape[1] != len(evidence):
        raise ValueError("Target evidence does not match matrix width")
    if count < 1 or probabilities.shape[0] == 0:
        return ()

    required = min(int(count), probabilities.shape[0])
    selected_factor = 0.0
    selected_indices: list[int] = []
    factors = [round(value, 2) for value in np.arange(1.0, -0.001, -RELAXATION_STEP)]
    for factor in factors:
        eligible = []
        for row in range(probabilities.shape[0]):
            match = evaluate_target_match(
                probabilities[row],
                uncertainties[row],
                evidence,
                relaxation_factor=factor,
                calibrated=calibrated,
            )
            if (
                all(item.passed_applied_floor for item in match.targets)
                and match.robust_target_fit >= match.applied_fit_floor
            ):
                eligible.append((row, match.robust_target_fit))
        eligible.sort(key=lambda item: (-item[1], item[0]))
        if len(eligible) >= required:
            selected_factor = factor
            selected_indices = [row for row, _ in eligible[:required]]
            break

    if not selected_indices:
        ranked = []
        for row in range(probabilities.shape[0]):
            match = evaluate_target_match(
                probabilities[row], uncertainties[row], evidence,
                relaxation_factor=0.0, calibrated=calibrated,
            )
            ranked.append((row, match.robust_target_fit))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        selected_indices = [row for row, _ in ranked[:required]]

    result = []
    for row in selected_indices:
        strict = evaluate_target_match(
            probabilities[row], uncertainties[row], evidence,
            relaxation_factor=1.0, calibrated=calibrated,
        )
        match = strict if strict.met_requested_gate else evaluate_target_match(
            probabilities[row], uncertainties[row], evidence,
            relaxation_factor=selected_factor, calibrated=calibrated,
        )
        result.append((row, match))
    result.sort(key=lambda item: (-item[1].robust_target_fit, item[0]))
    return tuple(result)


def target_match_payload(match: TargetMatch) -> dict[str, object]:
    return {
        "target_fit": match.target_fit,
        "robust_target_fit": match.robust_target_fit,
        "requested_fit_floor": match.requested_fit_floor,
        "applied_fit_floor": match.applied_fit_floor,
        "relaxation_factor": match.relaxation_factor,
        "tier": match.tier.value,
        "met_requested_gate": match.met_requested_gate,
        "calibrated": match.calibrated,
        "uses_absolute_probability_gate": match.uses_absolute_probability_gate,
        "targets": [
            {
                "name": item.name,
                "probability": item.probability,
                "uncertainty": item.uncertainty,
                "conservative_probability": item.conservative_probability,
                "maturity": item.maturity.value,
                "requested_floor": item.requested_floor,
                "applied_floor": item.applied_floor,
                "passed_requested_floor": item.passed_requested_floor,
                "passed_applied_floor": item.passed_applied_floor,
            }
            for item in match.targets
        ],
    }


def descriptor_evidence_payload(
    records: Sequence[DescriptorEvidence],
) -> list[dict[str, object]]:
    return [
        {
            "name": item.name,
            "positive_support": item.positive_support,
            "assessed_negative_support": item.assessed_negative_support,
            "maturity": item.maturity.value,
            "decision_threshold": item.decision_threshold,
            "calibration_method": item.calibration_method,
        }
        for item in records
    ]


def descriptor_evidence_from_payload(
    payload: Sequence[dict[str, object]],
) -> Tuple[DescriptorEvidence, ...]:
    records = tuple(
        DescriptorEvidence(
            name=str(item["name"]),
            positive_support=int(item["positive_support"]),
            assessed_negative_support=int(item["assessed_negative_support"]),
            maturity=DescriptorMaturity(str(item["maturity"])),
            decision_threshold=float(item["decision_threshold"]),
            calibration_method=str(item["calibration_method"]),
        )
        for item in payload
    )
    if len({item.name for item in records}) != len(records):
        raise ValueError("Descriptor evidence contains duplicate labels")
    return records
