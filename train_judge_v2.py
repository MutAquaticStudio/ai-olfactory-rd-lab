#!/usr/bin/env python3
"""Train a leakage-resistant five-seed Judge v2 candidate ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from olfactory.training.benchmark import (
    build_benchmark_manifest,
    load_immutable_manifest,
    save_immutable_manifest,
    split_from_payload,
)
from olfactory.training.dataset import load_legacy_baseline, load_versioned_snapshot
from olfactory.training.judge_v2 import train_judge_v2
from olfactory.training.judge_ensemble import build_chemprop_ensemble_artifact
from olfactory.resources import validate_resource_bundle


ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = validate_resource_bundle()


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="Versioned Parquet snapshot from Data intake")
    source.add_argument("--legacy-baseline", action="store_true", help="Reproduce the weak-label v1 baseline only")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--dataset-version")
    parser.add_argument("--seeds", default="11,17,23,31,43")
    parser.add_argument("--intensity-weight", type=float, choices=(0.1, 0.3, 1.0), default=0.3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--accelerator",
        choices=("cpu", "gpu", "mps"),
        default="cpu",
        help=(
            "Deterministic Chemprop training defaults to CPU. MPS is rejected "
            "because its scatter_reduce operation is not deterministic."
        ),
    )
    parser.add_argument("--allow-pre-panel-data", action="store_true")
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        help="Baseline ladder manifest with row-aligned locked-test predictions",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
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
    if args.split_manifest:
        split_payload = load_immutable_manifest(args.split_manifest, table=table)
    else:
        split_payload = build_benchmark_manifest(
            table,
            dataset_version=dataset_version,
            seed=42,
        )
        split_path = args.artifact_root / "splits" / f"{dataset_version}-60-10-15-15.json"
        save_immutable_manifest(split_path, split_payload)
    split = split_from_payload(split_payload)

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
                accelerator=args.accelerator,
            )
        )
    ensemble = build_chemprop_ensemble_artifact(
        table,
        split,
        [Path(str(item["manifest_path"])) for item in manifests],
        args.artifact_root,
        dataset_version=dataset_version,
        baseline_manifest_path=args.baseline_manifest,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(f"Judge v2 shadow ensemble written to {ensemble['manifest_path']}")


if __name__ == "__main__":
    main()
