#!/usr/bin/env python3
"""Create a reproducible legacy-data/model audit before Judge v2 development."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

from olfactory.models import OdorPredictor
from olfactory.training.metrics import multilabel_metrics
from olfactory.training.splits import chemical_group_split


ROOT = Path(__file__).resolve().parent


def f1_at(targets: np.ndarray, probabilities: np.ndarray, threshold: float):
    predicted = probabilities >= threshold
    tp = float(((targets == 1) & predicted).sum())
    fp = float(((targets == 0) & predicted).sum())
    fn = float(((targets == 1) & ~predicted).sum())
    micro = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    label_scores = []
    for label in range(targets.shape[1]):
        y = targets[:, label]
        p = predicted[:, label]
        label_tp = float(((y == 1) & p).sum())
        label_fp = float(((y == 0) & p).sum())
        label_fn = float(((y == 1) & ~p).sum())
        denominator = 2 * label_tp + label_fp + label_fn
        if denominator:
            label_scores.append(2 * label_tp / denominator)
    return {
        "micro_f1": micro,
        "macro_f1_supported": float(np.mean(label_scores)),
        "predicted_labels_per_molecule": float(predicted.sum(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "legacy-audit.json")
    args = parser.parse_args()
    frame = pd.read_csv(ROOT / "clean_dataset.csv")
    dataset = torch.load(ROOT / "odor_morgan_tensor_dataset.pt", map_location="cpu", weights_only=False)
    features, targets = dataset.tensors
    labels = tuple(str(name) for name in dataset.label_names)
    smiles_column = next(column for column in frame if column.lower() == "smiles")
    odor_column = next(column for column in frame if column.lower() in {"odor", "odors"})

    canonical_isomeric = []
    connectivity = []
    unresolved_stereo = 0
    with rdBase.BlockLogs():
        for value in frame[smiles_column].astype(str):
            molecule = Chem.MolFromSmiles(value)
            if molecule is None:
                canonical_isomeric.append(None)
                connectivity.append(None)
                continue
            canonical_isomeric.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
            connectivity.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
            unresolved_stereo += sum(
                item.specified == Chem.StereoSpecified.Unspecified
                for item in Chem.FindPotentialStereo(molecule)
            ) > 0

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(features))
    permutation = torch.randperm(len(features), generator=generator)
    train_indices = permutation[:train_size]
    test_indices = permutation[train_size:]
    model = OdorPredictor()
    model.load_state_dict(torch.load(ROOT / "odor_predictor_weights.pth", map_location="cpu", weights_only=True))
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(features[test_indices].float())).numpy()
    test_targets = targets[test_indices].numpy()
    metrics = multilabel_metrics(test_targets, probabilities)

    train_connectivity = {connectivity[index] for index in train_indices.tolist()}
    connectivity_leakage = sum(connectivity[index] in train_connectivity for index in test_indices.tolist())
    chemical_split = chemical_group_split(
        frame[smiles_column].astype(str).tolist(),
        targets.numpy(),
        seed=42,
    )
    positive_counts = targets.sum(dim=0).int().tolist()
    odorless_rows = frame[odor_column].fillna("").astype(str).str.split(";").apply(
        lambda values: [value.strip() for value in values if value.strip()]
    )
    odorless_conflicts = sum("odorless" in values and len(values) > 1 for values in odorless_rows)
    cid_column = next((column for column in frame if column.lower() == "cid"), None)
    negative_cids = int((pd.to_numeric(frame[cid_column], errors="coerce") < 0).sum()) if cid_column else None

    payload = {
        "audit_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "clean_dataset_sha256": hashlib.sha256((ROOT / "clean_dataset.csv").read_bytes()).hexdigest(),
            "tensor_dataset_sha256": hashlib.sha256((ROOT / "odor_morgan_tensor_dataset.pt").read_bytes()).hexdigest(),
            "weights_sha256": hashlib.sha256((ROOT / "odor_predictor_weights.pth").read_bytes()).hexdigest(),
        },
        "data": {
            "molecules": len(frame),
            "labels": len(labels),
            "unique_isomeric_smiles": len(set(canonical_isomeric)),
            "unique_connectivity_smiles": len(set(connectivity)),
            "unresolved_stereo_molecules": int(unresolved_stereo),
            "labels_with_at_most_50_positive": sum(value <= 50 for value in positive_counts),
            "minimum_label_support": min(positive_counts),
            "median_label_support": float(np.median(positive_counts)),
            "odorless_conflicts": int(odorless_conflicts),
            "negative_source_identifiers_in_cid_column": negative_cids,
        },
        "legacy_random_split": {
            "seed": 42,
            "test_rows": len(test_indices),
            "connectivity_leakage_rows": connectivity_leakage,
            "metrics": metrics,
            "threshold_sweep": {
                str(threshold): f1_at(test_targets, probabilities, threshold)
                for threshold in (0.1, 0.2, 0.3, 0.5)
            },
            "status": "DIAGNOSTIC_ONLY",
        },
        "locked_split_candidate": {
            "split_hash": chemical_split.split_hash,
            "train_rows": len(chemical_split.train_indices),
            "validation_rows": len(chemical_split.validation_indices),
            "test_rows": len(chemical_split.test_indices),
            "connectivity_leakage_rows": 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Accuracy and data-integrity audit written to {args.output}")


if __name__ == "__main__":
    main()
