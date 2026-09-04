#!/usr/bin/env python3
"""Train a leakage-resistant five-seed Judge v2 candidate ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from olfactory.training.dataset import load_legacy_baseline, load_versioned_snapshot
from olfactory.training.judge_v2 import train_judge_v2
from olfactory.training.splits import chemical_group_split
from olfactory.resources import validate_resource_bundle


ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = validate_resource_bundle()


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="Versioned Parquet snapshot from Data intake")
    source.add_argument("--legacy-baseline", action="store_true", help="Reproduce the weak-label v1 baseline only")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--dataset-version")
    parser.add_argument("--seeds", default="11,17,23,31,43")
    parser.add_argument("--intensity-weight", type=float, choices=(0.1, 0.3, 1.0), default=0.3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--allow-pre-panel-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.legacy_baseline:
        table = load_legacy_baseline(
            ROOT / "clean_dataset.csv",
            RESOURCE_DIR / "odor_morgan_tensor_dataset.pt",
        )
        dataset_version = args.dataset_version or "legacy-clean-3522"
    else:
        dataset = torch.load(
            RESOURCE_DIR / "odor_morgan_tensor_dataset.pt",
            map_location="cpu",
            weights_only=False,
        )
        labels = tuple(str(name) for name in dataset.label_names)
        table = load_versioned_snapshot(
            args.snapshot,
            labels,
            strict_panel_gate=not args.allow_pre_panel_data,
        )
        dataset_version = args.dataset_version or args.snapshot.stem
    if not np.isfinite(table.presence).any():
        raise SystemExit(
            "No assessed targets passed the configured panel/stereo gates. "
            "Collect replicated panel observations or use --allow-pre-panel-data for a diagnostic run."
        )
    split = chemical_group_split(
        table.smiles,
        np.nan_to_num(table.presence, nan=0.0),
        seed=42,
    )
    split_dir = args.artifact_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{dataset_version}-{split.split_hash[:12]}.json"
    split_path.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    manifests = []
    for seed in [int(value) for value in args.seeds.split(",") if value.strip()]:
        manifests.append(
            train_judge_v2(
                table,
                split,
                args.artifact_root,
                dataset_version=dataset_version,
                seed=seed,
                intensity_weight=args.intensity_weight,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
        )
    ensemble = {
        "model_family": "judge-v2-ensemble",
        "dataset_version": dataset_version,
        "split_hash": split.split_hash,
        "intensity_weight": args.intensity_weight,
        "members": manifests,
        "status": "CANDIDATE",
        "promotion_note": "Locked-test and prospective-panel gates must pass before registry promotion.",
    }
    output = args.artifact_root / "judge" / f"ensemble-{split.split_hash[:12]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ensemble, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Judge v2 candidate ensemble written to {output}")


if __name__ == "__main__":
    main()
