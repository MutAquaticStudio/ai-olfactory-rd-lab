#!/usr/bin/env python3
"""Benchmark one-vs-rest logistic and Morgan MLP on the shared locked split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from olfactory.features import create_morgan_tensor
from olfactory.training.baselines import fit_ovr_logistic, train_masked_morgan_mlp
from olfactory.training.calibration import CalibrationBundle
from olfactory.training.dataset import load_legacy_baseline, load_versioned_snapshot
from olfactory.training.metrics import multilabel_metrics
from olfactory.training.registry import sha256_file
from olfactory.training.splits import chemical_group_split
from olfactory.training.benchmark import dataset_fingerprint, load_immutable_manifest, split_from_payload
from olfactory.resources import validate_resource_bundle


ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = validate_resource_bundle()


def logits(probabilities):
    values = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--legacy-baseline", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--dataset-version")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-pre-panel-data", action="store_true")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Use an existing immutable split manifest")
    args = parser.parse_args()
    if args.legacy_baseline:
        table = load_legacy_baseline(ROOT / "clean_dataset.csv", RESOURCE_DIR / "odor_morgan_tensor_dataset.pt")
        dataset_version = args.dataset_version or "legacy-clean-3522"
    else:
        legacy = torch.load(RESOURCE_DIR / "odor_morgan_tensor_dataset.pt", map_location="cpu", weights_only=False)
        labels = tuple(str(value) for value in legacy.label_names)
        table = load_versioned_snapshot(args.snapshot, labels, strict_panel_gate=not args.allow_pre_panel_data)
        dataset_version = args.dataset_version or args.snapshot.stem
    features = torch.stack([create_morgan_tensor(Chem.MolFromSmiles(value)) for value in table.smiles]).float()
    if args.split_manifest:
        payload = load_immutable_manifest(args.split_manifest, table=table)
        split = split_from_payload(payload)
    else:
        split = chemical_group_split(table.smiles, np.nan_to_num(table.presence, nan=0.0), seed=args.seed)
    train, validation, test = map(list, (split.train_indices, split.validation_indices, split.test_indices))

    logistic_validation, estimators = fit_ovr_logistic(features[train].numpy(), table.presence[train], features[validation].numpy())
    logistic_test, _ = fit_ovr_logistic(features[train].numpy(), table.presence[train], features[test].numpy())
    logistic_calibration = CalibrationBundle.fit(logits(logistic_validation), table.presence[validation], table.label_names)
    logistic_metrics = multilabel_metrics(table.presence[test], logistic_calibration.transform_logits(logits(logistic_test)))

    morgan = train_masked_morgan_mlp(features, torch.from_numpy(table.presence), train, validation, seed=args.seed)
    with torch.inference_mode():
        validation_logits = morgan.model(features[validation]).numpy()
        test_logits = morgan.model(features[test]).numpy()
    morgan_calibration = CalibrationBundle.fit(validation_logits, table.presence[validation], table.label_names)
    morgan_metrics = multilabel_metrics(table.presence[test], morgan_calibration.transform_logits(test_logits))

    run_id = f"baseline-ladder-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-s{args.seed}"
    run_dir = args.artifact_root / "baselines" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights_path = run_dir / "morgan_baseline_weights.pth"
    torch.save(morgan.model.state_dict(), weights_path)
    manifest = {
        "run_id": run_id,
        "model_family": "judge-baseline-ladder",
        "dataset_version": dataset_version,
        "split_hash": split.split_hash,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "dataset_sha256": dataset_fingerprint(table),
        "seed": args.seed,
        "features": {"radius": 2, "bits": 2048, "use_chirality": True},
        "logistic": {"metrics": logistic_metrics, "label_models": len(estimators)},
        "morgan_mlp": {
            "metrics": morgan_metrics,
            "best_validation_loss": morgan.best_validation_loss,
            "epochs": morgan.epochs,
            "weights_path": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
        },
        "status": "BASELINE_BENCHMARK",
    }
    output = run_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Baseline ladder benchmark written to {output}")


if __name__ == "__main__":
    main()
