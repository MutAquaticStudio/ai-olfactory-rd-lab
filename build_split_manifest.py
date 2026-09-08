#!/usr/bin/env python3
"""Create the immutable leakage-resistant split used by every Judge benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from olfactory.training.benchmark import build_benchmark_manifest, save_immutable_manifest
from olfactory.training.dataset import load_legacy_baseline, load_versioned_snapshot
from olfactory.resources import validate_resource_bundle


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--legacy-baseline", action="store_true")
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--evidence-snapshot", type=Path, help="Manifest of an audit-only evidence snapshot")
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmarks" / "split_manifest.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    args = parser.parse_args()

    if args.evidence_snapshot:
        from olfactory.training.data_audit import build_release_audit
        report_dir = args.output.parent / args.output.stem
        build_release_audit(args.evidence_snapshot, report_dir, args.seed)
        print(f"AUDIT_ONLY release report written to {report_dir}")
        return
    resource_dir = validate_resource_bundle()
    if args.legacy_baseline:
        table = load_legacy_baseline(ROOT / "clean_dataset.csv", resource_dir / "odor_morgan_tensor_dataset.pt")
        version = args.dataset_version or "legacy-clean-3522"
    else:
        legacy = torch.load(resource_dir / "odor_morgan_tensor_dataset.pt", map_location="cpu", weights_only=False)
        labels = tuple(str(value) for value in legacy.label_names)
        table = load_versioned_snapshot(args.snapshot, labels)
        version = args.dataset_version or args.snapshot.stem
    payload = build_benchmark_manifest(
        table,
        dataset_version=version,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
    )
    path = save_immutable_manifest(args.output, payload)
    print(f"Immutable split manifest written to {path}")


if __name__ == "__main__":
    main()
