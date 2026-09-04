"""Stable predictor boundary shared by the API and training adapters.

The application deliberately depends on this small protocol rather than on a
particular neural-network implementation.  This keeps the production Morgan
baseline and future graph models interchangeable without leaking model code
into the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import torch
from rdkit import Chem, rdBase

from .features import create_morgan_tensor, model_device
from .models import ODOR_LABEL_COUNT
from .prediction_integrity import PredictionIdentity, nearest_training_similarity, reliability_state


@dataclass(frozen=True)
class PredictionBatch:
    """Model output with provenance and applicability-domain information.

    Arrays are row-major (one row per input SMILES).  Missing intensity and
    uncertainty are represented by NaN, not zero, so callers cannot confuse
    an unassessed value with a measured zero.
    """

    model_version: str
    dataset_version: str
    calibration_version: str
    presence_probability: np.ndarray
    expected_intensity: np.ndarray
    ensemble_uncertainty: np.ndarray
    training_similarity: np.ndarray
    reliability_state: Tuple[str, ...]
    label_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        probabilities = _float_array(self.presence_probability)
        intensity = _float_array(self.expected_intensity)
        uncertainty = _float_array(self.ensemble_uncertainty)
        similarity = _float_array(self.training_similarity)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.label_names):
            raise ValueError("presence_probability must have shape (rows, label_count)")
        if len(self.label_names) != ODOR_LABEL_COUNT:
            raise ValueError(f"Predictors must expose exactly {ODOR_LABEL_COUNT} labels")
        expected_shape = probabilities.shape
        for name, value in (("expected_intensity", intensity), ("ensemble_uncertainty", uncertainty)):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must match presence_probability shape")
        if similarity.ndim != 1 or similarity.shape[0] != probabilities.shape[0]:
            raise ValueError("training_similarity must contain one value per row")
        if len(self.reliability_state) != probabilities.shape[0]:
            raise ValueError("reliability_state must contain one value per row")
        object.__setattr__(self, "presence_probability", np.clip(probabilities, 0.0, 1.0))
        object.__setattr__(self, "expected_intensity", intensity)
        object.__setattr__(self, "ensemble_uncertainty", uncertainty)
        object.__setattr__(self, "training_similarity", similarity)

    @property
    def row_count(self) -> int:
        return int(self.presence_probability.shape[0])

    def to_payload(self) -> dict:
        """Serialize the contract without exposing numpy scalar objects."""
        rows = []
        for row in range(self.row_count):
            rows.append(
                {
                    "presence_predictions": [
                        {
                            "name": label,
                            "probability": float(self.presence_probability[row, index]),
                            "expected_intensity": _finite_or_none(self.expected_intensity[row, index]),
                            "uncertainty": _finite_or_none(self.ensemble_uncertainty[row, index]),
                        }
                        for index, label in enumerate(self.label_names)
                    ],
                    "nearest_training_similarity": _finite_or_none(self.training_similarity[row]),
                    "reliability_state": self.reliability_state[row],
                }
            )
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "calibration_version": self.calibration_version,
            "rows": rows,
        }


def _finite_or_none(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def _float_array(value: object) -> np.ndarray:
    """Convert optional JSON-style numbers to float arrays (None → NaN)."""
    raw = np.asarray(value)
    if raw.dtype.kind in {"O", "U", "S"}:
        raw = np.where(raw == None, np.nan, raw)  # noqa: E711 - intentional None test
    return np.asarray(raw, dtype=np.float32)


@runtime_checkable
class MoleculePredictor(Protocol):
    """Minimal inference seam implemented by v1 and v2 model adapters."""

    label_names: Tuple[str, ...]

    def predict(self, isomeric_smiles: Sequence[str]) -> PredictionBatch:
        ...


class LegacyMorganPredictor:
    """Adapter for the unchanged production Morgan MLP (Judge v1)."""

    def __init__(
        self,
        model: torch.nn.Module,
        label_names: Sequence[str],
        *,
        identity: PredictionIdentity,
        training_fingerprints: Optional[torch.Tensor] = None,
        calibration: Optional[object] = None,
    ) -> None:
        self.model = model
        self.label_names = tuple(str(value) for value in label_names)
        if len(self.label_names) != ODOR_LABEL_COUNT:
            raise ValueError(f"Legacy predictor requires exactly {ODOR_LABEL_COUNT} labels")
        self.identity = identity
        self.training_fingerprints = training_fingerprints
        self.calibration = calibration
        if self.identity.calibration_version not in {"", "uncalibrated"} and calibration is None:
            raise ValueError("A declared calibration version requires a calibration artifact")

    def predict(self, isomeric_smiles: Sequence[str]) -> PredictionBatch:
        molecules = []
        for raw in isomeric_smiles:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(str(raw), sanitize=True)
            if molecule is None:
                raise ValueError(f"Invalid SMILES: {raw}")
            molecules.append(molecule)
        if molecules:
            features = torch.stack([create_morgan_tensor(molecule) for molecule in molecules])
            with torch.inference_mode():
                logits = self.model(features.to(model_device(self.model))).cpu().numpy()
                probabilities = (
                    self.calibration.transform_logits(logits)
                    if self.calibration is not None
                    else 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
                )
            similarities = np.asarray(
                [
                    nearest_training_similarity(feature, self.training_fingerprints)
                    for feature in features
                ],
                dtype=np.float32,
            )
        else:
            probabilities = np.empty((0, len(self.label_names)), dtype=np.float32)
            similarities = np.empty((0,), dtype=np.float32)
        states = tuple(reliability_state(value if np.isfinite(value) else None) for value in similarities)
        shape = probabilities.shape
        return PredictionBatch(
            model_version=self.identity.model_version,
            dataset_version=self.identity.dataset_version,
            calibration_version=self.identity.calibration_version,
            presence_probability=probabilities,
            expected_intensity=np.full(shape, np.nan, dtype=np.float32),
            ensemble_uncertainty=np.full(shape, np.nan, dtype=np.float32),
            training_similarity=similarities,
            reliability_state=states,
            label_names=self.label_names,
        )


class EnsemblePredictor:
    """Aggregate independently trained predictors without mixing force fields.

    The ensemble reports the standard deviation of presence probabilities as
    epistemic uncertainty.  It is useful in shadow evaluation; callers still
    need a calibration bundle fitted on the dedicated calibration partition
    before promotion.
    """

    def __init__(self, predictors: Sequence[MoleculePredictor], *, model_version: str = "ensemble"):
        self.predictors = tuple(predictors)
        if len(self.predictors) < 2:
            raise ValueError("An ensemble requires at least two predictors")
        self.label_names = tuple(self.predictors[0].label_names)
        if any(tuple(item.label_names) != self.label_names for item in self.predictors[1:]):
            raise ValueError("All ensemble members must expose the same labels")
        self.model_version = model_version

    def predict(self, isomeric_smiles: Sequence[str]) -> PredictionBatch:
        batches = [predictor.predict(isomeric_smiles) for predictor in self.predictors]
        provenance = {
            (batch.dataset_version, batch.calibration_version)
            for batch in batches
        }
        if len(provenance) != 1:
            raise ValueError(
                "All ensemble members must share dataset and calibration provenance"
            )
        values = np.stack([batch.presence_probability for batch in batches], axis=0)
        intensities = np.stack([batch.expected_intensity for batch in batches], axis=0)
        similarities = np.stack([batch.training_similarity for batch in batches], axis=0)
        similarity_count = np.isfinite(similarities).sum(axis=0)
        mean_similarity = np.divide(
            np.nansum(similarities, axis=0),
            np.maximum(similarity_count, 1),
            where=similarity_count > 0,
            out=np.zeros_like(similarity_count, dtype=np.float32),
        ).astype(np.float32)
        mean_similarity[similarity_count == 0] = np.nan
        intensity_count = np.isfinite(intensities).sum(axis=0)
        mean_intensity = np.divide(
            np.nansum(intensities, axis=0),
            np.maximum(intensity_count, 1),
            where=intensity_count > 0,
            out=np.zeros_like(intensity_count, dtype=np.float32),
        ).astype(np.float32)
        mean_intensity[intensity_count == 0] = np.nan
        identity = batches[0]
        return PredictionBatch(
            model_version=self.model_version,
            dataset_version=identity.dataset_version,
            calibration_version=identity.calibration_version,
            presence_probability=values.mean(axis=0),
            expected_intensity=mean_intensity,
            ensemble_uncertainty=values.std(axis=0),
            training_similarity=mean_similarity,
            reliability_state=tuple(reliability_state(value if np.isfinite(value) else None) for value in mean_similarity),
            label_names=self.label_names,
        )


__all__ = ["PredictionBatch", "MoleculePredictor", "LegacyMorganPredictor", "EnsemblePredictor"]
