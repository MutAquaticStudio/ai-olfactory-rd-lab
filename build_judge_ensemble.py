#!/usr/bin/env python3
"""Assemble five completed Judge members without retraining them."""

from __future__ import annotations

import argparse
from pathlib import Path

from olfactory.resources import validate_resource_bundle
from olfactory.training.benchmark import load_immutable_manifest, split_from_payload
from olfactory.training.dataset import load_legacy_baseline
from olfactory.training.judge_ensemble import build_chemprop_ensemble_artifact


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path, action="append", required=True)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--dataset-version", default="legacy-clean-3522-shadow-audit-v1")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()
    resource_dir = validate_resource_bundle()
    table = load_legacy_baseline(
        ROOT / "clean_dataset.csv",
        resource_dir / "odor_morgan_tensor_dataset.pt",
    )
    split = split_from_payload(load_immutable_manifest(args.split_manifest, table=table))
    artifact = build_chemprop_ensemble_artifact(
        table,
        split,
        args.member_manifest,
        args.artifact_root,
        dataset_version=args.dataset_version,
        baseline_manifest_path=args.baseline_manifest,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    print(f"Judge v2 shadow ensemble written to {artifact['manifest_path']}")


if __name__ == "__main__":
    main()
