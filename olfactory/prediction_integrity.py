"""Version, calibration and applicability-domain metadata for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch


@dataclass(frozen=True)
class PredictionIdentity:
    model_version: str
    dataset_version: str
    calibration_version: str
    model_status: str


def nearest_training_similarity(
    fingerprint: torch.Tensor,
    training_fingerprints: Optional[torch.Tensor],
) -> Optional[float]:
    """Return maximum binary Tanimoto similarity to the training reference set."""
    if training_fingerprints is None or training_fingerprints.numel() == 0:
        return None
    query = fingerprint.detach().to(dtype=torch.bool, device="cpu").reshape(1, -1)
    reference = training_fingerprints.detach().to(dtype=torch.bool, device="cpu")
    if reference.ndim != 2 or reference.shape[1] != query.shape[1]:
        raise ValueError("Training fingerprints do not match the inference feature size")
    intersection = (reference & query).sum(dim=1, dtype=torch.float32)
    union = (reference | query).sum(dim=1, dtype=torch.float32).clamp_min(1.0)
    return float((intersection / union).max().item())


def reliability_state(similarity: Optional[float]) -> str:
    """Conservative evidence label; thresholds are versioned with this contract."""
    if similarity is None or similarity < 0.35:
        return "OUT_OF_DOMAIN"
    if similarity < 0.60:
        return "LIMITED_EVIDENCE"
    return "IN_DOMAIN"


def legacy_prediction_payload(
    probabilities: Sequence[float],
    labels: Sequence[str],
    identity: PredictionIdentity,
    similarity: Optional[float],
) -> Dict[str, object]:
    """Expose legacy output honestly while preserving the v1 response for one release."""
    return {
        "model_version": identity.model_version,
        "dataset_version": identity.dataset_version,
        "calibration_version": identity.calibration_version,
        "model_status": identity.model_status,
        "calibrated": identity.calibration_version not in {"", "uncalibrated"},
        "nearest_training_similarity": similarity,
        "reliability_state": reliability_state(similarity),
        "presence_predictions": [
            {
                "name": str(label),
                "probability": float(probability),
                "expected_intensity": None,
                "uncertainty": None,
                "decision_threshold": None,
            }
            for label, probability in zip(labels, probabilities)
        ],
        "limitations": [
            "Legacy catalog labels treat unmentioned descriptors as negatives.",
            "Probabilities are not calibrated and do not include ensemble uncertainty.",
            "Reliability is based on nearest chiral Morgan fingerprint similarity only.",
        ],
    }
