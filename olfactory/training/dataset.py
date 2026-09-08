"""Build model matrices without collapsing unknown labels into negatives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .reliability import icc_2k, krippendorff_alpha_nominal


@dataclass(frozen=True)
class MolecularTargetTable:
    smiles: Tuple[str, ...]
    label_names: Tuple[str, ...]
    presence: np.ndarray
    intensity: np.ndarray
    source: Tuple[str, ...]
    stereo_state: Tuple[str, ...]
    presence_reliability: Optional[np.ndarray] = None
    intensity_reliability: Optional[np.ndarray] = None

    @property
    def presence_mask(self) -> np.ndarray:
        return np.isfinite(self.presence)

    @property
    def intensity_mask(self) -> np.ndarray:
        return np.isfinite(self.intensity)


def load_legacy_baseline(
    csv_path: Path,
    tensor_path: Path,
) -> MolecularTargetTable:
    """Load the v1 matrix exactly as trained, explicitly marked as weak-label baseline."""
    frame = pd.read_csv(csv_path)
    dataset = torch.load(tensor_path, map_location="cpu", weights_only=False)
    labels = tuple(str(name) for name in dataset.label_names)
    _, targets = dataset.tensors
    smiles_column = next(column for column in frame.columns if column.lower() == "smiles")
    if len(frame) != len(targets):
        raise ValueError("Legacy CSV and tensor dataset have different row counts")
    return MolecularTargetTable(
        smiles=tuple(frame[smiles_column].astype(str)),
        label_names=labels,
        presence=targets.detach().cpu().numpy().astype(np.float32),
        intensity=np.full(targets.shape, np.nan, dtype=np.float32),
        source=tuple("legacy_weak_catalog" for _ in range(len(frame))),
        stereo_state=tuple("LEGACY_UNAUDITED" for _ in range(len(frame))),
    )


def _active_assessments(frame: pd.DataFrame) -> pd.DataFrame:
    if "supersedes_assessment_id" not in frame:
        return frame
    superseded = set(frame["supersedes_assessment_id"].dropna().astype(str))
    return frame[~frame["assessment_id"].astype(str).isin(superseded)].copy()


def load_versioned_snapshot(
    snapshot_path: Path,
    label_names: Sequence[str],
    *,
    strict_panel_gate: bool = True,
    minimum_assessors: int = 8,
    minimum_repeats: int = 2,
    minimum_presence_alpha: float = 0.5,
    minimum_intensity_icc: float = 0.5,
    measurement_domain: Optional[dict] = None,
) -> MolecularTargetTable:
    path = Path(snapshot_path)
    manifest_path = path if path.suffix == ".json" else path.parent / "manifest.json"
    if not manifest_path.exists() and path.with_suffix(".manifest.json").exists():
        manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata.get("status") == "AUDIT_ONLY" or metadata.get("training_eligible") is False:
            raise ValueError("AUDIT_ONLY evidence requires a reviewed training release")
        if path.suffix == ".json":
            raise ValueError("Supply the training Parquet path, not a manifest")
        if metadata.get("parquet_sha256"):
            from ..data_foundation.evidence import file_hash
            if file_hash(path) != metadata["parquet_sha256"]:
                raise ValueError("Training snapshot checksum mismatch")
    frame = _active_assessments(pd.read_parquet(snapshot_path))
    required = {"assessment_id", "study_name", "session_name", "assessor_id", "inchikey", "descriptor",
                "presence_state", "intensity", "replicate_number", "stereo_state", "isomeric_smiles"}
    if not required <= set(frame):
        raise ValueError("Snapshot is not a reviewed assessment table")
    domain_fields = [name for name in ("study_name", "source_name", "source_version", "concentration",
                                      "concentration_unit", "solvent", "temperature_c") if name in frame]
    for name, value in (measurement_domain or {}).items():
        if name not in domain_fields:
            raise ValueError(f"Unknown measurement domain field: {name}")
        frame = frame[frame[name].eq(value)]
    if len(frame[domain_fields].drop_duplicates()) > 1:
        raise ValueError("Select one measurement_domain; conditions cannot be pooled silently")
    if frame.empty:
        raise ValueError("No assessments in the selected measurement domain")
    # Unresolved stereochemistry is excluded rather than assigned a random identity.
    frame = frame[frame.stereo_state.isin(["DEFINED", "ACHIRAL"])].copy()
    reliability = {}
    for descriptor, observations in frame.groupby("descriptor"):
        assessed = observations[observations.presence_state != "UNASSESSED"].copy()
        unit_fields = ["inchikey", "session_name", "replicate_number"]
        if assessed.duplicated(unit_fields + ["assessor_id"]).any():
            raise ValueError("Duplicate active assessment in the same measurement context")
        presence_matrix = assessed.assign(numeric=assessed.presence_state.map({"ABSENT": 0., "PRESENT": 1.})).pivot(
            index=unit_fields, columns="assessor_id", values="numeric").to_numpy(dtype=float)
        intensity_matrix = assessed[assessed.presence_state == "PRESENT"].pivot(
            index=unit_fields, columns="assessor_id", values="intensity").to_numpy(dtype=float)
        reliability[descriptor] = (krippendorff_alpha_nominal(presence_matrix), icc_2k(intensity_matrix))
    labels = tuple(str(name) for name in label_names)
    label_index = {name: index for index, name in enumerate(labels)}
    grouped = list(frame.groupby("inchikey", sort=True))
    presence = np.full((len(grouped), len(labels)), np.nan, dtype=np.float32)
    intensity = np.full_like(presence, np.nan)
    presence_reliability = np.full_like(presence, np.nan)
    intensity_reliability = np.full_like(presence, np.nan)
    smiles: List[str] = []
    sources: List[str] = []
    stereo_states: List[str] = []

    for molecule_row, (_, molecule_frame) in enumerate(grouped):
        first = molecule_frame.iloc[0]
        smiles.append(str(first["isomeric_smiles"]))
        sources.append(str(first.get("source_name", first["study_name"])))
        stereo_states.append(str(first["stereo_state"]))
        if str(first["stereo_state"]) == "UNRESOLVED":
            continue
        for descriptor, observations in molecule_frame.groupby("descriptor"):
            if descriptor not in label_index:
                continue
            assessed = observations[observations["presence_state"] != "UNASSESSED"]
            if assessed.empty:
                continue
            if strict_panel_gate:
                assessor_counts = assessed.drop_duplicates(["assessor_id", "session_name", "replicate_number"]).groupby("assessor_id").size()
                if len(assessor_counts) < minimum_assessors:
                    continue
                if int((assessor_counts >= minimum_repeats).sum()) < minimum_assessors:
                    continue
                sessions = assessed.groupby("assessor_id").session_name.nunique()
                if int((sessions >= minimum_repeats).sum()) < minimum_assessors:
                    continue
                alpha = reliability[descriptor][0]
                if not np.isfinite(alpha) or alpha < minimum_presence_alpha:
                    continue
            else:
                alpha = float("nan")
            values = assessed["presence_state"].map({"ABSENT": 0.0, "PRESENT": 1.0})
            label = label_index[str(descriptor)]
            presence[molecule_row, label] = float(values.mean() >= 0.5)
            presence_reliability[molecule_row, label] = alpha
            present_intensity = assessed.loc[
                assessed["presence_state"] == "PRESENT", "intensity"
            ].dropna()
            if not present_intensity.empty:
                if strict_panel_gate:
                    intensity_icc = reliability[descriptor][1]
                    intensity_reliability[molecule_row, label] = intensity_icc
                    if np.isfinite(intensity_icc) and intensity_icc >= minimum_intensity_icc:
                        intensity[molecule_row, label] = float(present_intensity.median())
                else:
                    intensity[molecule_row, label] = float(present_intensity.median())
    return MolecularTargetTable(
        tuple(smiles),
        labels,
        presence,
        intensity,
        tuple(sources),
        tuple(stereo_states),
        presence_reliability,
        intensity_reliability,
    )
