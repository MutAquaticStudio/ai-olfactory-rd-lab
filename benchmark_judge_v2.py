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
from olfactory.training.splits import SplitManifest, chemical_group_folds, chemical_group_split


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
    return parser.parse_args()


def fold_split(development_indices, folds, fold_index, seed):
    holdout = [development_indices[index] for index in folds[fold_index]]
    validation_fold = (fold_index + 1) % len(folds)
    validation = [development_indices[index] for index in folds[validation_fold]]
    excluded = set(folds[fold_index]) | set(folds[validation_fold])
    train = [development_indices[index] for index in range(len(development_indices)) if index not in excluded]
    content = {"train": sorted(train), "validation": sorted(validation), "cv_holdout": sorted(holdout), "seed": seed, "fold": fold_index}
    split_hash = hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()
    return SplitManifest(
        tuple(content["train"]),
        tuple(content["validation"]),
        tuple(content["cv_holdout"]),
        tuple(),
        seed,
        (len(train) / len(development_indices), len(validation) / len(development_indices), len(holdout) / len(development_indices)),
        0.6,
        split_hash,
    )


def main() -> None:
    args = parse_args()
    if args.legacy_baseline:
        table = load_legacy_baseline(ROOT / "clean_dataset.csv", ROOT / "odor_morgan_tensor_dataset.pt")
        dataset_version = args.dataset_version or "legacy-clean-3522"
    else:
        legacy = torch.load(ROOT / "odor_morgan_tensor_dataset.pt", map_location="cpu", weights_only=False)
        labels = tuple(str(value) for value in legacy.label_names)
        table = load_versioned_snapshot(args.snapshot, labels, strict_panel_gate=not args.allow_pre_panel_data)
        dataset_version = args.dataset_version or args.snapshot.stem
    target_matrix = np.nan_to_num(table.presence, nan=0.0)
    locked = chemical_group_split(table.smiles, target_matrix, seed=42)
    development = [*locked.train_indices, *locked.validation_indices]
    folds = chemical_group_folds(
        [table.smiles[index] for index in development],
        target_matrix[development],
        fold_count=args.folds,
        seed=42,
    )
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) != 3:
        raise SystemExit("The scientific benchmark contract requires exactly three seeds.")
    runs = []
    for seed in seeds:
        for fold_index in range(args.folds):
            split = fold_split(development, folds.folds, fold_index, seed)
            manifest = train_judge_v2(
                table,
                split,
                args.artifact_root,
                dataset_version=dataset_version,
                seed=seed,
                intensity_weight=args.intensity_weight,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
            manifest["evaluation_partition"] = "DEVELOPMENT_GROUPED_CV"
            manifest["cv_fold"] = fold_index
            runs.append(manifest)
    summary = {
        "benchmark": "judge-v2-grouped-five-fold-three-seed",
        "dataset_version": dataset_version,
        "locked_test_split_hash": locked.split_hash,
        "cv_fold_hash": folds.fold_hash,
        "locked_test_was_used_for_tuning": False,
        "seeds": seeds,
        "folds": args.folds,
        "runs": runs,
        "status": "DEVELOPMENT_BENCHMARK",
    }
    output = args.artifact_root / "benchmarks" / f"judge-v2-cv-{folds.fold_hash[:12]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Grouped CV benchmark written to {output}")


if __name__ == "__main__":
    main()
