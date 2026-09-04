"""Chemprop D-MPNN with masked presence and conditional-intensity heads."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F

from ..features import create_morgan_tensor
from ..target_matching import descriptor_evidence, descriptor_evidence_payload
from .calibration import CalibrationBundle
from .dataset import MolecularTargetTable
from .metrics import intensity_metrics, multilabel_metrics
from .benchmark import dataset_fingerprint
from .registry import sha256_file
from .splits import SplitManifest
from .tracking import log_manifest_to_mlflow


JUDGE_V2_ARCHITECTURE = {
    "message_passing": "chemprop-v2-bond-dmpnn",
    "message_hidden": 300,
    "message_depth": 3,
    "morgan_bits": 2048,
    "morgan_radius": 2,
    "morgan_chirality": True,
    "shared_hidden": 512,
    "dropout": 0.2,
    "presence_labels": 113,
    "intensity_labels": 113,
}


def effective_positive_weights(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.999,
    maximum: float = 20.0,
) -> torch.Tensor:
    positives = ((targets == 1) & mask).sum(dim=0).float()
    negatives = ((targets == 0) & mask).sum(dim=0).float()
    positive_effective = 1.0 - torch.pow(torch.tensor(beta), positives.clamp_min(1.0))
    negative_effective = 1.0 - torch.pow(torch.tensor(beta), negatives.clamp_min(1.0))
    weights = negative_effective / positive_effective.clamp_min(1e-8)
    weights = torch.where((positives > 0) & (negatives > 0), weights, torch.ones_like(weights))
    return weights.clamp(1.0, maximum)


def masked_multitask_loss(
    presence_logits: torch.Tensor,
    intensity_predictions: torch.Tensor,
    presence_targets: torch.Tensor,
    intensity_targets: torch.Tensor,
    positive_weights: torch.Tensor,
    intensity_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    presence_mask = torch.isfinite(presence_targets)
    intensity_mask = torch.isfinite(intensity_targets) & (presence_targets == 1)
    safe_presence = torch.nan_to_num(presence_targets, nan=0.0)
    safe_intensity = torch.nan_to_num(intensity_targets, nan=0.0)
    raw_presence = F.binary_cross_entropy_with_logits(
        presence_logits,
        safe_presence,
        pos_weight=positive_weights.to(presence_logits.device),
        reduction="none",
    )
    presence_loss = (
        raw_presence[presence_mask].mean()
        if presence_mask.any()
        else presence_logits.sum() * 0.0
    )
    raw_intensity = F.huber_loss(
        intensity_predictions,
        safe_intensity,
        reduction="none",
        delta=1.0,
    )
    intensity_loss = (
        raw_intensity[intensity_mask].mean()
        if intensity_mask.any()
        else intensity_predictions.sum() * 0.0
    )
    total = presence_loss + intensity_weight * intensity_loss
    return total, {
        "presence_loss": presence_loss.detach(),
        "intensity_loss": intensity_loss.detach(),
    }


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_datapoints(table: MolecularTargetTable, indices: Sequence[int]):
    try:
        from chemprop.data import MoleculeDatapoint
    except ImportError as error:
        raise RuntimeError(
            "Judge v2 training requires Python 3.11–3.12 and requirements-training.txt"
        ) from error
    datapoints = []
    for index in indices:
        molecule = Chem.MolFromSmiles(table.smiles[index])
        if molecule is None:
            raise ValueError(f"Invalid training SMILES at row {index}")
        fingerprint = create_morgan_tensor(molecule).numpy().astype(np.float32)
        targets = np.concatenate(
            [table.presence[index], table.intensity[index]]
        ).astype(np.float32)
        datapoints.append(
            MoleculeDatapoint.from_smi(
                table.smiles[index],
                y=targets,
                x_d=fingerprint,
                ignore_stereo=False,
                name=str(index),
            )
        )
    return datapoints


def _module_class(label_count: int, positive_weights: torch.Tensor, intensity_weight: float):
    try:
        from lightning import pytorch as pl
        from chemprop.nn import BondMessagePassing, MeanAggregation
    except ImportError as error:
        raise RuntimeError(
            "Judge v2 training requires Python 3.11–3.12 and requirements-training.txt"
        ) from error

    class ChempropJudgeV2(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(
                {
                    "label_count": label_count,
                    "intensity_weight": intensity_weight,
                    "architecture": JUDGE_V2_ARCHITECTURE,
                }
            )
            self.message_passing = BondMessagePassing(
                d_h=JUDGE_V2_ARCHITECTURE["message_hidden"],
                depth=JUDGE_V2_ARCHITECTURE["message_depth"],
                dropout=JUDGE_V2_ARCHITECTURE["dropout"],
            )
            self.aggregation = MeanAggregation()
            encoder_size = self.message_passing.output_dim + JUDGE_V2_ARCHITECTURE["morgan_bits"]
            self.shared = nn.Sequential(
                nn.Linear(encoder_size, JUDGE_V2_ARCHITECTURE["shared_hidden"]),
                nn.ReLU(),
                nn.Dropout(JUDGE_V2_ARCHITECTURE["dropout"]),
            )
            self.presence_head = nn.Linear(JUDGE_V2_ARCHITECTURE["shared_hidden"], label_count)
            self.intensity_head = nn.Linear(JUDGE_V2_ARCHITECTURE["shared_hidden"], label_count)
            self.register_buffer("positive_weights", positive_weights.float())

        def forward(self, bmg, atom_descriptors=None, molecule_descriptors=None):
            atom_states = self.message_passing(bmg, atom_descriptors)
            molecular = self.aggregation(atom_states, bmg.batch)
            if molecule_descriptors is None:
                raise RuntimeError("Chiral Morgan descriptors are required by Judge v2")
            encoded = self.shared(torch.cat([molecular, molecule_descriptors], dim=1))
            return self.presence_head(encoded), self.intensity_head(encoded)

        def _step(self, batch, stage: str):
            bmg, atom_descriptors, molecule_descriptors, targets, _, _, _ = batch
            presence_targets = targets[:, :label_count]
            intensity_targets = targets[:, label_count:]
            presence_logits, intensity_predictions = self(
                bmg,
                atom_descriptors,
                molecule_descriptors,
            )
            loss, parts = masked_multitask_loss(
                presence_logits,
                intensity_predictions,
                presence_targets,
                intensity_targets,
                self.positive_weights,
                intensity_weight,
            )
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, batch_size=len(targets))
            self.log(f"{stage}_presence_loss", parts["presence_loss"], on_epoch=True, batch_size=len(targets))
            self.log(f"{stage}_intensity_loss", parts["intensity_loss"], on_epoch=True, batch_size=len(targets))
            return loss

        def training_step(self, batch, batch_index):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_index):
            return self._step(batch, "validation")

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)

    return ChempropJudgeV2


def _predict(model, loader, label_count: int, device: torch.device):
    logits: List[np.ndarray] = []
    intensities: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            bmg, atom_descriptors, molecule_descriptors, batch_targets, _, _, _ = batch
            bmg = bmg.to(device)
            atom_descriptors = atom_descriptors.to(device) if atom_descriptors is not None else None
            molecule_descriptors = molecule_descriptors.to(device) if molecule_descriptors is not None else None
            presence_logits, intensity = model(bmg, atom_descriptors, molecule_descriptors)
            logits.append(presence_logits.cpu().numpy())
            intensities.append(intensity.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())
    matrix = np.concatenate(targets)
    return np.concatenate(logits), np.concatenate(intensities), matrix[:, :label_count], matrix[:, label_count:]


def train_judge_v2(
    table: MolecularTargetTable,
    split: SplitManifest,
    artifact_root: Path,
    *,
    dataset_version: str,
    seed: int,
    intensity_weight: float = 0.3,
    max_epochs: int = 100,
    patience: int = 20,
) -> Dict[str, object]:
    try:
        from lightning import pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from chemprop.data import MoleculeDataset, build_dataloader
    except ImportError as error:
        raise RuntimeError(
            "Judge v2 training requires Python 3.11–3.12 and requirements-training.txt"
        ) from error
    if len(table.label_names) != 113:
        raise ValueError("Judge v2 requires exactly 113 odor descriptors")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

    train_targets = torch.from_numpy(table.presence[list(split.train_indices)])
    positive_weights = effective_positive_weights(
        train_targets,
        torch.isfinite(train_targets),
    )
    train_dataset = MoleculeDataset(_build_datapoints(table, split.train_indices))
    calibration_indices = split.calibration_indices or split.validation_indices
    calibration_dataset = MoleculeDataset(_build_datapoints(table, calibration_indices))
    validation_dataset = MoleculeDataset(_build_datapoints(table, split.validation_indices))
    test_dataset = MoleculeDataset(_build_datapoints(table, split.test_indices))
    train_dataset.cache = True
    calibration_dataset.cache = True
    validation_dataset.cache = True
    test_dataset.cache = True
    train_loader = build_dataloader(train_dataset, batch_size=64, seed=seed, shuffle=True)
    calibration_loader = build_dataloader(calibration_dataset, batch_size=128, shuffle=False)
    validation_loader = build_dataloader(validation_dataset, batch_size=128, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size=128, shuffle=False)

    module_type = _module_class(len(table.label_names), positive_weights, intensity_weight)
    model = module_type()
    run_id = f"judge-v2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-s{seed}"
    run_dir = Path(artifact_root) / "judge" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best",
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="validation_loss",
        mode="min",
        patience=patience,
    )

    learning_history: List[Dict[str, float]] = []

    class LearningCurveCallback(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking:
                return
            current_device = pl_module.device
            logits, intensity_values, presence_values, intensity_targets = _predict(
                pl_module,
                validation_loader,
                len(table.label_names),
                current_device,
            )
            presence_metrics = multilabel_metrics(
                presence_values,
                1.0 / (1.0 + np.exp(-logits)),
            )
            current_intensity = intensity_metrics(intensity_targets, intensity_values)

            def metric_value(name: str) -> float:
                value = trainer.callback_metrics.get(name, float("nan"))
                if isinstance(value, torch.Tensor):
                    return float(value.detach().cpu().item())
                return float(value)

            learning_history.append(
                {
                    "epoch": float(trainer.current_epoch + 1),
                    "train_loss": metric_value("train_loss"),
                    "validation_loss": metric_value("validation_loss"),
                    "validation_macro_ap": float(presence_metrics["macro_average_precision_supported"]),
                    "validation_micro_ap": float(presence_metrics["micro_average_precision"]),
                    "validation_ece": float(presence_metrics["mean_label_ece"]),
                    "validation_intensity_mae": float(current_intensity["masked_mae"]),
                }
            )
    if torch.cuda.is_available():
        accelerator = "gpu"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    else:
        accelerator = "cpu"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        deterministic=True,
        logger=False,
        callbacks=[checkpoint, early_stopping, LearningCurveCallback()],
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, validation_loader)
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    calibration_logits, calibration_intensity, calibration_presence, calibration_intensity_target = _predict(
        model, calibration_loader, len(table.label_names), device
    )
    calibration = CalibrationBundle.fit(
        calibration_logits,
        calibration_presence,
        table.label_names,
    )
    calibration_path = run_dir / "calibration.json"
    calibration.save(calibration_path)
    descriptor_path = run_dir / "descriptor_evidence.json"
    descriptor_records = descriptor_evidence(
        table.label_names,
        np.sum(table.presence == 1, axis=0).astype(int).tolist(),
        np.sum(table.presence == 0, axis=0).astype(int).tolist(),
        calibration.thresholds,
        calibration.methods,
    )
    descriptor_path.write_text(
        json.dumps(
            descriptor_evidence_payload(descriptor_records),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    calibration_probabilities = calibration.transform_logits(calibration_logits)
    validation_logits, validation_intensity, validation_presence, validation_intensity_target = _predict(
        model, validation_loader, len(table.label_names), device
    )
    validation_probabilities = calibration.transform_logits(validation_logits)
    test_logits, test_intensity, test_presence, test_intensity_target = _predict(
        model, test_loader, len(table.label_names), device
    )
    test_probabilities = calibration.transform_logits(test_logits)
    metrics = {
        "calibration": {
            "presence": multilabel_metrics(calibration_presence, calibration_probabilities),
            "intensity": intensity_metrics(calibration_intensity_target, calibration_intensity),
        },
        "validation": {
            "presence": multilabel_metrics(validation_presence, validation_probabilities),
            "intensity": intensity_metrics(validation_intensity_target, validation_intensity),
        },
        "locked_test": {
            "presence": multilabel_metrics(test_presence, test_probabilities),
            "intensity": intensity_metrics(test_intensity_target, test_intensity),
        },
    }
    weights_path = run_dir / "judge_v2_weights.pth"
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "architecture": JUDGE_V2_ARCHITECTURE,
            "label_names": list(table.label_names),
        },
        weights_path,
    )
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architecture": JUDGE_V2_ARCHITECTURE,
                "seed": seed,
                "intensity_weight": intensity_weight,
                "max_epochs": max_epochs,
                "patience": patience,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_path = run_dir / "split.json"
    split_path.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    learning_json_path = run_dir / "learning_curve.json"
    learning_csv_path = run_dir / "learning_curve.csv"
    learning_png_path = run_dir / "learning_curve.png"
    learning_json_path.write_text(json.dumps(learning_history, indent=2, sort_keys=True), encoding="utf-8")
    with learning_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(learning_history[0]))
        writer.writeheader()
        writer.writerows(learning_history)
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in learning_history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, [record["train_loss"] for record in learning_history], label="Train")
    axes[0, 0].plot(epochs, [record["validation_loss"] for record in learning_history], label="Validation")
    axes[0, 0].set(title="Masked multitask loss", xlabel="Epoch", ylabel="Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [record["validation_macro_ap"] for record in learning_history], label="Macro AP")
    axes[0, 1].plot(epochs, [record["validation_micro_ap"] for record in learning_history], label="Micro AP")
    axes[0, 1].set(title="Presence ranking", xlabel="Epoch", ylabel="Average precision")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, [record["validation_ece"] for record in learning_history])
    axes[1, 0].set(title="Calibration error (uncalibrated epoch output)", xlabel="Epoch", ylabel="Mean label ECE")
    axes[1, 1].plot(epochs, [record["validation_intensity_mae"] for record in learning_history])
    axes[1, 1].set(title="Conditional intensity", xlabel="Epoch", ylabel="Masked MAE")
    figure.tight_layout()
    figure.savefig(learning_png_path, dpi=180)
    plt.close(figure)
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "run_id": run_id,
        "model_version": run_id,
        "architecture": JUDGE_V2_ARCHITECTURE,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_fingerprint(table),
        "split_hash": split.split_hash,
        "seed": seed,
        "intensity_weight": intensity_weight,
        "git_commit": _git_commit(root),
        "calibration_version": calibration.calibration_version,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "descriptor_evidence_path": str(descriptor_path),
        "descriptor_evidence_sha256": sha256_file(descriptor_path),
        "config_path": str(config_path),
        "split_path": str(split_path),
        "metrics_path": str(metrics_path),
        "checksums": {
            str(path.name): sha256_file(path)
            for path in (
                weights_path,
                calibration_path,
                descriptor_path,
                config_path,
                split_path,
                metrics_path,
                learning_json_path,
                learning_csv_path,
                learning_png_path,
            )
        },
        "metrics": metrics,
        "learning_curve_path": str(learning_png_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["mlflow_run_id"] = log_manifest_to_mlflow(
        manifest,
        manifest_path,
        Path(artifact_root) / "tracking",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def select_intensity_weight(manifests: Sequence[Dict[str, object]]) -> float:
    """Choose MAE winner among runs within 2% of the best validation macro AP."""
    if not manifests:
        raise ValueError("No Judge v2 tuning runs were supplied")
    scored = []
    for manifest in manifests:
        validation = manifest["metrics"]["validation"]
        macro_ap = float(validation["presence"]["macro_average_precision_supported"])
        mae = float(validation["intensity"]["masked_mae"])
        weight = float(manifest.get("intensity_weight", 0.3))
        scored.append((macro_ap, mae, weight))
    best_ap = max(item[0] for item in scored if math.isfinite(item[0]))
    eligible = [item for item in scored if item[0] >= best_ap * 0.98]
    finite_mae = [item for item in eligible if math.isfinite(item[1])]
    return min(finite_mae or eligible, key=lambda item: item[1] if math.isfinite(item[1]) else float("inf"))[2]
