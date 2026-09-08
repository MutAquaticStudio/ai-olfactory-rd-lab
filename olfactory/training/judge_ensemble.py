"""Five-seed Chemprop shadow ensemble, calibration, and audit artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem, rdBase

from ..features import create_morgan_tensor
from ..prediction import PredictionBatch
from ..prediction_integrity import PredictionIdentity, nearest_training_similarity, reliability_state
from ..target_matching import descriptor_evidence, descriptor_evidence_payload
from .benchmark import dataset_fingerprint
from .calibration import CalibrationBundle
from .dataset import MolecularTargetTable
from .gates import grouped_bootstrap_delta, judge_promotion_gate
from .judge_v2 import _module_class
from .metrics import intensity_metrics, multilabel_metrics
from .registry import sha256_file
from .splits import SplitManifest


ENSEMBLE_SCHEMA_VERSION = 1
ENSEMBLE_SIZE = 5


def _artifact_path(manifest_path: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    # Historical training manifests stored paths relative to the repository,
    # while portable run bundles keep referenced files beside the manifest.
    # Resolve both without trusting a non-existent, doubly-prefixed path.
    candidates = (
        manifest_path.parent / path,
        path.resolve(),
        manifest_path.parent / path.name,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[-1])


def _load_manifest(path: Path) -> Dict[str, object]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("model_version") is None and payload.get("run_id") is None:
        raise ValueError(f"Judge member manifest is missing model identity: {manifest_path.name}")
    return payload


def _verified_member_model(manifest_path: Path, label_names: Sequence[str]):
    payload = _load_manifest(manifest_path)
    weights_path = _artifact_path(manifest_path, payload["weights_path"])
    if sha256_file(weights_path) != payload.get("weights_sha256"):
        raise ValueError("Judge member weights checksum verification failed")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    member_labels = tuple(str(value) for value in checkpoint["label_names"])
    if member_labels != tuple(label_names):
        raise ValueError("Judge ensemble member label order does not match the ensemble")
    intensity_weight = float(payload.get("intensity_weight", 0.3))
    module_type = _module_class(len(member_labels), torch.ones(len(member_labels)), intensity_weight)
    model = module_type()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(torch.device("cpu"))
    return model


def _inference_loader(smiles: Sequence[str], label_count: int):
    try:
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    except ImportError as error:
        raise RuntimeError("Chemprop is required when Judge shadow mode is enabled") from error
    datapoints = []
    for raw in smiles:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(str(raw), sanitize=True)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {raw}")
        datapoints.append(
            MoleculeDatapoint.from_smi(
                str(raw),
                y=np.zeros(label_count * 2, dtype=np.float32),
                x_d=create_morgan_tensor(molecule).numpy().astype(np.float32),
                ignore_stereo=False,
            )
        )
    dataset = MoleculeDataset(datapoints)
    dataset.cache = True
    return build_dataloader(dataset, batch_size=128, shuffle=False, drop_last=False)


def _member_outputs(model, loader) -> Tuple[np.ndarray, np.ndarray]:
    logits: List[np.ndarray] = []
    intensities: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            bmg, atom_descriptors, molecule_descriptors, *_ = batch
            bmg.to(torch.device("cpu"))
            if atom_descriptors is not None:
                atom_descriptors = atom_descriptors.to(torch.device("cpu"))
            if molecule_descriptors is not None:
                molecule_descriptors = molecule_descriptors.to(torch.device("cpu"))
            presence, intensity = model(bmg, atom_descriptors, molecule_descriptors)
            logits.append(presence.cpu().numpy())
            intensities.append(intensity.cpu().numpy())
    if not logits:
        width = int(model.presence_head.out_features)
        return np.empty((0, width), dtype=np.float32), np.empty((0, width), dtype=np.float32)
    return np.concatenate(logits), np.concatenate(intensities)


@dataclass
class ChempropEnsemblePredictor:
    """Checksum-verified shadow adapter implementing ``MoleculePredictor``."""

    models: Tuple[object, ...]
    label_names: Tuple[str, ...]
    identity: PredictionIdentity
    calibration: CalibrationBundle
    training_fingerprints: torch.Tensor
    intensity_available: bool = False
    gate_summary: Optional[Dict[str, object]] = None

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "ChempropEnsemblePredictor":
        path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != ENSEMBLE_SCHEMA_VERSION:
            raise ValueError("Unsupported Judge shadow ensemble schema")
        labels = tuple(str(value) for value in payload.get("label_names", ()))
        if len(labels) != 113:
            raise ValueError("Judge shadow ensemble must expose exactly 113 labels")
        members = payload.get("members", [])
        if len(members) != ENSEMBLE_SIZE:
            raise ValueError("Judge shadow ensemble must contain exactly five members")
        models = []
        for member in members:
            member_path = _artifact_path(path, member["manifest_path"])
            if sha256_file(member_path) != member.get("manifest_sha256"):
                raise ValueError("Judge member manifest checksum verification failed")
            models.append(_verified_member_model(member_path, labels))
        calibration_path = _artifact_path(path, payload["calibration_path"])
        if sha256_file(calibration_path) != payload.get("calibration_sha256"):
            raise ValueError("Judge ensemble calibration checksum verification failed")
        calibration = CalibrationBundle.load(calibration_path)
        if calibration.label_names != labels:
            raise ValueError("Judge ensemble calibration label order mismatch")
        fingerprint_path = _artifact_path(path, payload["training_fingerprints_path"])
        if sha256_file(fingerprint_path) != payload.get("training_fingerprints_sha256"):
            raise ValueError("Judge ensemble training fingerprint checksum verification failed")
        fingerprints = torch.load(fingerprint_path, map_location="cpu", weights_only=True)
        if fingerprints.ndim != 2 or fingerprints.shape[1] != 2048:
            raise ValueError("Judge ensemble training fingerprints must have 2,048 bits")
        gate_path = _artifact_path(path, payload["gate_report_path"])
        if sha256_file(gate_path) != payload.get("gate_report_sha256"):
            raise ValueError("Judge ensemble gate-report checksum verification failed")
        gate_report = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_summary = {
            "status": gate_report.get("status"),
            "eligible": bool(gate_report.get("eligible", False)),
            "blocked_reasons": list(gate_report.get("blocked_reasons", ())),
            "external_blockers": list(gate_report.get("external_blockers", ())),
        }
        return cls(
            models=tuple(models),
            label_names=labels,
            identity=PredictionIdentity(
                model_version=str(payload["model_version"]),
                dataset_version=str(payload["dataset_version"]),
                calibration_version=str(payload["calibration_version"]),
                model_status="SHADOW_ONLY",
            ),
            calibration=calibration,
            training_fingerprints=fingerprints,
            intensity_available=bool(payload.get("intensity_available", False)),
            gate_summary=gate_summary,
        )

    def predict(self, isomeric_smiles: Sequence[str]) -> PredictionBatch:
        loader = _inference_loader(isomeric_smiles, len(self.label_names))
        member_logits: List[np.ndarray] = []
        member_intensity: List[np.ndarray] = []
        # Rebuild the deterministic, non-shuffled loader for every member.
        for index, model in enumerate(self.models):
            if index:
                loader = _inference_loader(isomeric_smiles, len(self.label_names))
            logits, intensity = _member_outputs(model, loader)
            member_logits.append(logits)
            member_intensity.append(intensity)
        stacked_logits = np.stack(member_logits, axis=0)
        calibrated_members = np.stack(
            [self.calibration.transform_logits(values) for values in stacked_logits],
            axis=0,
        )
        probabilities = self.calibration.transform_logits(stacked_logits.mean(axis=0))
        uncertainties = calibrated_members.std(axis=0)
        shape = probabilities.shape
        intensities = (
            np.stack(member_intensity, axis=0).mean(axis=0)
            if self.intensity_available
            else np.full(shape, np.nan, dtype=np.float32)
        )
        similarities = []
        for raw in isomeric_smiles:
            molecule = Chem.MolFromSmiles(str(raw), sanitize=True)
            similarity = nearest_training_similarity(
                create_morgan_tensor(molecule),
                self.training_fingerprints,
            )
            similarities.append(float("nan") if similarity is None else similarity)
        similarity_values = np.asarray(similarities, dtype=np.float32)
        return PredictionBatch(
            model_version=self.identity.model_version,
            dataset_version=self.identity.dataset_version,
            calibration_version=self.identity.calibration_version,
            presence_probability=probabilities,
            expected_intensity=intensities,
            ensemble_uncertainty=uncertainties,
            training_similarity=similarity_values,
            reliability_state=tuple(
                reliability_state(value if np.isfinite(value) else None)
                for value in similarity_values
            ),
            label_names=self.label_names,
            calibration_methods=self.calibration.methods,
        )


def _member_prediction_arrays(manifest_path: Path) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    payload = _load_manifest(manifest_path)
    predictions_path = _artifact_path(manifest_path, payload["raw_predictions_path"])
    if sha256_file(predictions_path) != payload.get("raw_predictions_sha256"):
        raise ValueError("Judge member raw-prediction checksum verification failed")
    with np.load(predictions_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return payload, arrays


def _assert_aligned(name: str, values: Sequence[np.ndarray]) -> None:
    first = values[0]
    if any(not np.array_equal(first, value, equal_nan=True) for value in values[1:]):
        raise ValueError(f"Judge ensemble members have misaligned {name}")


def _aggregate_learning_curves(member_manifests: Sequence[Path], output_dir: Path) -> Tuple[Path, Path]:
    histories: List[List[Dict[str, float]]] = []
    for manifest_path in member_manifests:
        manifest = _load_manifest(manifest_path)
        curve_path = _artifact_path(manifest_path, manifest["learning_curve_path"])
        json_path = curve_path.with_suffix(".json")
        histories.append(json.loads(json_path.read_text(encoding="utf-8")))
    maximum_epoch = max(len(history) for history in histories)
    fields = (
        "train_loss",
        "validation_loss",
        "validation_macro_ap",
        "validation_micro_ap",
        "validation_ece",
        "validation_intensity_mae",
    )
    rows: List[Dict[str, float]] = []
    for epoch in range(maximum_epoch):
        record: Dict[str, float] = {"epoch": float(epoch + 1)}
        for field in fields:
            values = np.asarray(
                [history[epoch][field] for history in histories if epoch < len(history)],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            record[f"{field}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            record[f"{field}_std"] = float(finite.std()) if len(finite) else float("nan")
        rows.append(record)
    csv_path = output_dir / "aggregate_learning_curve.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] for row in rows])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = (
        (axes[0, 0], "validation_loss", "Validation loss", "Loss"),
        (axes[0, 1], "validation_macro_ap", "Macro average precision", "AP"),
        (axes[1, 0], "validation_micro_ap", "Micro average precision", "AP"),
        (axes[1, 1], "validation_ece", "Calibration error", "ECE"),
    )
    for axis, field, title, ylabel in plots:
        mean = np.asarray([row[f"{field}_mean"] for row in rows])
        deviation = np.asarray([row[f"{field}_std"] for row in rows])
        axis.plot(epochs, mean, color="#008080")
        axis.fill_between(epochs, mean - deviation, mean + deviation, color="#00A896", alpha=0.18)
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
    figure.suptitle("Judge v2 five-seed shadow ensemble — mean ± standard deviation")
    figure.tight_layout()
    png_path = output_dir / "aggregate_learning_curve.png"
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return csv_path, png_path


def build_chemprop_ensemble_artifact(
    table: MolecularTargetTable,
    split: SplitManifest,
    member_manifest_paths: Sequence[Path],
    artifact_root: Path,
    *,
    dataset_version: str,
    baseline_manifest_path: Optional[Path] = None,
    bootstrap_iterations: int = 10_000,
) -> Dict[str, object]:
    """Combine five immutable members without changing the production registry."""
    members = tuple(Path(path).expanduser().resolve() for path in member_manifest_paths)
    if len(members) != ENSEMBLE_SIZE:
        raise ValueError("The scientific Judge ensemble requires exactly five members")
    loaded = [_member_prediction_arrays(path) for path in members]
    manifests = [item[0] for item in loaded]
    arrays = [item[1] for item in loaded]
    if any(item.get("dataset_sha256") != dataset_fingerprint(table) for item in manifests):
        raise ValueError("Judge member dataset checksum does not match the supplied table")
    if any(item.get("split_hash") != split.split_hash for item in manifests):
        raise ValueError("Judge members were not trained on the same immutable split")
    seeds = [int(item["seed"]) for item in manifests]
    if len(set(seeds)) != ENSEMBLE_SIZE:
        raise ValueError("Judge ensemble members must use five distinct seeds")

    for prefix in ("calibration", "validation", "locked_test"):
        _assert_aligned(f"{prefix} indices", [item[f"{prefix}_indices"] for item in arrays])
        _assert_aligned(f"{prefix} targets", [item[f"{prefix}_presence"] for item in arrays])
    calibration_logits = np.stack([item["calibration_logits"] for item in arrays]).mean(axis=0)
    validation_logits = np.stack([item["validation_logits"] for item in arrays]).mean(axis=0)
    test_member_logits = np.stack([item["locked_test_logits"] for item in arrays])
    test_logits = test_member_logits.mean(axis=0)
    calibration_targets = arrays[0]["calibration_presence"]
    validation_targets = arrays[0]["validation_presence"]
    test_targets = arrays[0]["locked_test_presence"]
    calibration = CalibrationBundle.fit(calibration_logits, calibration_targets, table.label_names)
    calibration.calibration_version = "ensemble-platt-tier-v1"
    calibration_probabilities = calibration.transform_logits(calibration_logits)
    validation_probabilities = calibration.transform_logits(validation_logits)
    test_probabilities = calibration.transform_logits(test_logits)
    test_member_probabilities = np.stack(
        [calibration.transform_logits(values) for values in test_member_logits]
    )
    intensity_available = bool(np.isfinite(table.intensity[list(split.train_indices)]).any())
    averaged_intensity = {
        prefix: np.stack([item[f"{prefix}_intensity"] for item in arrays]).mean(axis=0)
        for prefix in ("calibration", "validation", "locked_test")
    }
    metrics = {
        "calibration": {
            "presence": multilabel_metrics(calibration_targets, calibration_probabilities),
            "intensity": intensity_metrics(
                arrays[0]["calibration_intensity_targets"], averaged_intensity["calibration"]
            ),
        },
        "validation": {
            "presence": multilabel_metrics(validation_targets, validation_probabilities),
            "intensity": intensity_metrics(
                arrays[0]["validation_intensity_targets"], averaged_intensity["validation"]
            ),
        },
        "locked_test": {
            "presence": multilabel_metrics(test_targets, test_probabilities),
            "intensity": intensity_metrics(
                arrays[0]["locked_test_intensity_targets"], averaged_intensity["locked_test"]
            ),
        },
    }

    run_id = f"judge-v2-shadow-ensemble-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = Path(artifact_root) / "judge" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    calibration_path = run_dir / "calibration.json"
    calibration.save(calibration_path)
    descriptor_path = run_dir / "descriptor_evidence.json"
    descriptor_records = descriptor_evidence(
        table.label_names,
        np.sum(calibration_targets == 1, axis=0).astype(int).tolist(),
        np.sum(calibration_targets == 0, axis=0).astype(int).tolist(),
        calibration.thresholds,
        calibration.methods,
    )
    descriptor_path.write_text(
        json.dumps(descriptor_evidence_payload(descriptor_records), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    fingerprints_path = run_dir / "training_fingerprints.pt"
    fingerprints = torch.stack(
        [
            create_morgan_tensor(Chem.MolFromSmiles(table.smiles[index])).to(torch.uint8)
            for index in split.train_indices
        ]
    )
    torch.save(fingerprints, fingerprints_path)
    predictions_path = run_dir / "raw_predictions.npz"
    np.savez_compressed(
        predictions_path,
        locked_test_indices=arrays[0]["locked_test_indices"],
        locked_test_targets=test_targets.astype(np.float32),
        mean_logits=test_logits.astype(np.float32),
        calibrated_probabilities=test_probabilities.astype(np.float32),
        calibrated_member_probabilities=test_member_probabilities.astype(np.float32),
        ensemble_uncertainty=test_member_probabilities.std(axis=0).astype(np.float32),
    )
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    curve_csv, curve_png = _aggregate_learning_curves(members, run_dir)

    bootstrap = {"mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    baseline_metrics: Dict[str, float] = {
        "macro_average_precision_supported": float("nan"),
        "micro_average_precision": float("nan"),
        "mean_label_ece": float("nan"),
    }
    baseline_intensity_mae = float("nan")
    if baseline_manifest_path is not None:
        baseline_path = Path(baseline_manifest_path).expanduser().resolve()
        baseline_manifest = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline_manifest.get("split_hash") != split.split_hash:
            raise ValueError("Baseline and Judge ensemble split hashes do not match")
        raw_path = _artifact_path(baseline_path, baseline_manifest["raw_predictions_path"])
        if sha256_file(raw_path) != baseline_manifest.get("raw_predictions_sha256"):
            raise ValueError("Baseline raw-prediction checksum verification failed")
        with np.load(raw_path) as archive:
            if not np.array_equal(archive["locked_test_indices"], arrays[0]["locked_test_indices"]):
                raise ValueError("Baseline and Judge ensemble test rows are misaligned")
            baseline_probabilities = archive["morgan_probabilities"]
        baseline_metrics = dict(baseline_manifest["morgan_mlp"]["locked_test_metrics"])
        test_indices = arrays[0]["locked_test_indices"].astype(int)
        group_ids = [split.group_ids[index] for index in test_indices]
        bootstrap = grouped_bootstrap_delta(
            test_targets,
            baseline_probabilities,
            test_probabilities,
            group_ids,
            iterations=bootstrap_iterations,
            seed=42,
        )

    candidate_metrics = metrics["locked_test"]["presence"]
    candidate_intensity_mae = float(metrics["locked_test"]["intensity"]["masked_mae"])
    statistical_gate = judge_promotion_gate(
        baseline_metrics,
        candidate_metrics,
        bootstrap_macro_delta_lower=float(bootstrap["lower_95"]),
        baseline_intensity_mae=baseline_intensity_mae,
        candidate_intensity_mae=candidate_intensity_mae,
    )
    gate_report = {
        **statistical_gate.to_dict(),
        "eligible": False,
        "status": "SHADOW_ONLY",
        "external_blockers": [
            "locked_test_exposed",
            "prospective_panel_missing",
            "intensity_panel_missing" if not intensity_available else "prospective_intensity_gate_pending",
            "legacy_weak_labels_treat_unmentioned_terms_as_negatives",
        ],
        "bootstrap": bootstrap,
    }
    gate_path = run_dir / "gate_report.json"
    gate_path.write_text(json.dumps(gate_report, indent=2, sort_keys=True), encoding="utf-8")

    member_records = [
        {
            "seed": int(payload["seed"]),
            "model_version": str(payload.get("model_version", payload["run_id"])),
            "manifest_path": str(Path("..") / path.parent.name / path.name),
            "manifest_sha256": sha256_file(path),
        }
        for path, payload in zip(members, manifests)
    ]
    manifest: Dict[str, object] = {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "run_id": run_id,
        "model_family": "chemprop-chiral-dmpnn-morgan-ensemble",
        "model_version": run_id,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_fingerprint(table),
        "split_hash": split.split_hash,
        "locked_test_status": "EXPOSED_RETROSPECTIVE_TEST",
        "label_names": list(table.label_names),
        "members": member_records,
        "calibration_version": calibration.calibration_version,
        "calibration_path": calibration_path.name,
        "calibration_sha256": sha256_file(calibration_path),
        "descriptor_evidence_path": descriptor_path.name,
        "descriptor_evidence_sha256": sha256_file(descriptor_path),
        "training_fingerprints_path": fingerprints_path.name,
        "training_fingerprints_sha256": sha256_file(fingerprints_path),
        "raw_predictions_path": predictions_path.name,
        "raw_predictions_sha256": sha256_file(predictions_path),
        "metrics_path": metrics_path.name,
        "metrics_sha256": sha256_file(metrics_path),
        "aggregate_learning_curve_path": curve_png.name,
        "aggregate_learning_curve_sha256": sha256_file(curve_png),
        "aggregate_learning_curve_csv_path": curve_csv.name,
        "aggregate_learning_curve_csv_sha256": sha256_file(curve_csv),
        "gate_report_path": gate_path.name,
        "gate_report_sha256": sha256_file(gate_path),
        "intensity_available": intensity_available,
        "status": "SHADOW_ONLY",
        "production_promoted": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = run_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_checksum = sha256_file(manifest_path)
    manifest_path.with_suffix(".sha256").write_text(
        f"{manifest_checksum}  {manifest_path.name}\n",
        encoding="utf-8",
    )
    result = dict(manifest)
    result.update(
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_checksum,
            "calibration_path": str(calibration_path),
            "descriptor_evidence_path": str(descriptor_path),
            "training_fingerprints_path": str(fingerprints_path),
            "raw_predictions_path": str(predictions_path),
            "metrics_path": str(metrics_path),
            "aggregate_learning_curve_path": str(curve_png),
            "aggregate_learning_curve_csv_path": str(curve_csv),
            "gate_report_path": str(gate_path),
        }
    )
    return result


def load_chemprop_ensemble_predictor(manifest_path: Path) -> ChempropEnsemblePredictor:
    return ChempropEnsemblePredictor.from_manifest(manifest_path)


__all__ = [
    "ChempropEnsemblePredictor",
    "build_chemprop_ensemble_artifact",
    "load_chemprop_ensemble_predictor",
]
