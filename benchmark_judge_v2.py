#!/usr/bin/env python3
"""Run grouped five-fold Judge v2 CV on development data with three seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from olfactory.training.dataset import load_legacy_baseline, load_versioned_snapshot
from olfactory.training.judge_v2 import train_judge_v2
from olfactory.training.splits import (
    SplitManifest,
    chemical_group_calibrated_split,
    chemical_group_folds,
    chemical_group_split,
)
from olfactory.training.benchmark import dataset_fingerprint, load_immutable_manifest, split_from_payload
from olfactory.training.registry import sha256_file
from olfactory.resources import validate_resource_bundle


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--legacy-baseline", action="store_true")
    parser.add_argument("--dataset-version")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seeds", default="11,17,23")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--intensity-weight", type=float, choices=(0.1, 0.3, 1.0), default=0.3)
    parser.add_argument("--allow-pre-panel-data", action="store_true")
    parser.add_argument("--accelerator", choices=("cpu", "gpu", "mps"), default="cpu")
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Use an existing immutable split manifest")
    parser.add_argument(
        "--resume-manifests",
        type=Path,
        nargs="+",
        default=None,
        help="Assemble the CV summary from completed member manifests without retraining",
    )
    return parser.parse_args()


def fold_split(table, development_indices, folds, fold_index, seed, group_ids=None):
    holdout = [development_indices[index] for index in folds[fold_index]]
    holdout_local = set(folds[fold_index])
    remaining = [
        development_indices[index]
        for index in range(len(development_indices))
        if index not in holdout_local
    ]
    # Outer CV keeps one 20% chemical-group fold untouched. The remaining 80%
    # is split again at group level into 60% train, 10% calibration and 10%
    # validation of total development rows.
    inner = chemical_group_split(
        [table.smiles[index] for index in remaining],
        np.nan_to_num(table.presence[remaining], nan=0.0),
        ratios=(0.75, 0.125, 0.125),
        seed=seed * 100 + fold_index,
        fixed_group_ids=[group_ids[index] for index in remaining] if group_ids else None,
    )
    train = [remaining[index] for index in inner.train_indices]
    calibration = [remaining[index] for index in inner.validation_indices]
    validation = [remaining[index] for index in inner.test_indices]
    content = {
        "train": sorted(train),
        "calibration": sorted(calibration),
        "validation": sorted(validation),
        "cv_holdout": sorted(holdout),
        "seed": seed,
        "fold": fold_index,
    }
    split_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()
    return SplitManifest(
        train_indices=tuple(content["train"]),
        validation_indices=tuple(content["validation"]),
        test_indices=tuple(content["cv_holdout"]),
        group_ids=tuple(),
        seed=seed,
        ratios=(
            len(train) / len(development_indices),
            len(calibration) / len(development_indices),
            len(validation) / len(development_indices),
            len(holdout) / len(development_indices),
        ),
        similarity_threshold=0.6,
        split_hash=split_hash,
        calibration_indices=tuple(content["calibration"]),
    )


def load_completed_runs(paths, table, development, folds, seeds, group_ids=None):
    expected_dataset_sha256 = dataset_fingerprint(table)
    expected = {}
    for seed in seeds:
        for fold_index in range(len(folds)):
            split = fold_split(table, development, folds, fold_index, seed, group_ids)
            expected[split.split_hash] = (seed, fold_index)

    completed = {}
    for path in paths:
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("dataset_sha256") != expected_dataset_sha256:
            raise ValueError(f"Dataset checksum mismatch in completed CV manifest: {manifest_path}")
        split_hash = str(payload.get("split_hash", ""))
        if split_hash not in expected:
            raise ValueError(f"Manifest does not belong to this CV contract: {manifest_path}")
        seed, fold_index = expected[split_hash]
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Seed mismatch in completed CV manifest: {manifest_path}")
        if payload.get("locked_test_status") != "DEVELOPMENT_CV_HOLDOUT":
            raise ValueError(f"Completed manifest is not a development CV holdout: {manifest_path}")
        if split_hash in completed:
            raise ValueError(f"Duplicate completed CV split: {split_hash}")
        metrics_path = Path(str(payload.get("metrics_path", "metrics.json")))
        if not metrics_path.is_absolute():
            metrics_path = manifest_path.parent / metrics_path
        expected_metrics_sha256 = payload.get("checksums", {}).get("metrics.json")
        if not expected_metrics_sha256 or sha256_file(metrics_path) != expected_metrics_sha256:
            raise ValueError(f"Metrics checksum mismatch in completed CV manifest: {manifest_path}")
        payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["manifest_path"] = str(manifest_path)
        payload["manifest_sha256"] = sha256_file(manifest_path)
        payload["evaluation_partition"] = "DEVELOPMENT_GROUPED_CV"
        payload["cv_fold"] = fold_index
        completed[split_hash] = payload

    missing = sorted(set(expected) - set(completed))
    if missing:
        raise ValueError(f"Missing {len(missing)} completed CV member manifest(s)")
    return [
        completed[fold_split(table, development, folds, fold_index, seed, group_ids).split_hash]
        for seed in seeds
        for fold_index in range(len(folds))
    ]


def main() -> None:
    args = parse_args()
    resource_dir = validate_resource_bundle()
    if args.legacy_baseline:
        table = load_legacy_baseline(ROOT / "clean_dataset.csv", resource_dir / "odor_morgan_tensor_dataset.pt")
        dataset_version = args.dataset_version or "legacy-clean-3522"
    else:
        legacy = torch.load(resource_dir / "odor_morgan_tensor_dataset.pt", map_location="cpu", weights_only=False)
        labels = tuple(str(value) for value in legacy.label_names)
        table = load_versioned_snapshot(args.snapshot, labels, strict_panel_gate=not args.allow_pre_panel_data)
        dataset_version = args.dataset_version or args.snapshot.stem
    target_matrix = np.nan_to_num(table.presence, nan=0.0)
    if args.split_manifest:
        locked = split_from_payload(load_immutable_manifest(args.split_manifest, table=table))
    else:
        locked = chemical_group_calibrated_split(table.smiles, target_matrix, seed=42)
    development = [
        *locked.train_indices,
        *locked.validation_indices,
    ]
    folds = chemical_group_folds(
        [table.smiles[index] for index in development],
        target_matrix[development],
        fold_count=args.folds,
        seed=42,
        fixed_group_ids=[locked.group_ids[index] for index in development],
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != 3:
        raise SystemExit("The scientific benchmark contract requires exactly three seeds.")
    if args.resume_manifests:
        runs = load_completed_runs(
            args.resume_manifests,
            table,
            development,
            folds.folds,
            seeds,
            locked.group_ids,
        )
    else:
        runs = []
        for seed in seeds:
            for fold_index in range(args.folds):
                split = fold_split(table, development, folds.folds, fold_index, seed, locked.group_ids)
                manifest = train_judge_v2(
                    table,
                    split,
                    args.artifact_root,
                    dataset_version=dataset_version,
                    seed=seed,
                    intensity_weight=args.intensity_weight,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    accelerator=args.accelerator,
                    evaluation_partition_status="DEVELOPMENT_CV_HOLDOUT",
                )
                manifest["evaluation_partition"] = "DEVELOPMENT_GROUPED_CV"
                manifest["cv_fold"] = fold_index
                runs.append(manifest)
    metric_names = (
        "macro_average_precision_supported",
        "micro_average_precision",
        "mean_label_ece",
        "mean_brier",
        "precision_at_5",
        "recall_at_5",
    )
    aggregate = {}
    for name in metric_names:
        values = np.asarray(
            [run["metrics"]["locked_test"]["presence"][name] for run in runs],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        aggregate[name] = {
            "mean": float(finite.mean()) if len(finite) else float("nan"),
            "std": float(finite.std()) if len(finite) else float("nan"),
            "runs": int(len(finite)),
        }
    summary = {
        "benchmark": "judge-v2-grouped-five-fold-three-seed",
        "dataset_version": dataset_version,
        "locked_test_split_hash": locked.split_hash,
        "cv_fold_hash": folds.fold_hash,
        "locked_test_was_used_for_tuning": False,
        "seeds": seeds,
        "folds": args.folds,
        "runs": runs,
        "aggregate_holdout_metrics": aggregate,
        "status": "DEVELOPMENT_BENCHMARK",
    }
    output = args.artifact_root / "benchmarks" / f"judge-v2-cv-{folds.fold_hash[:12]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    checksum_path = output.with_suffix(".sha256")
    checksum_path.write_text(f"{sha256_file(output)}  {output.name}\n", encoding="utf-8")
    print(f"Grouped CV benchmark written to {output}")


if __name__ == "__main__":
    main()
