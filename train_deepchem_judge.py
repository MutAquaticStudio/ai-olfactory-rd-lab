#!/usr/bin/env python3
"""Train one optional DeepChem graph Judge v2 candidate artifact.

Run this from the Python 3.11 training environment.  The command never
promotes the result; compare its manifest with the locked baseline first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from olfactory.training.benchmark import load_immutable_manifest, split_from_payload
from olfactory.training.dataset import load_legacy_baseline, load_versioned_snapshot
from olfactory.training.deepchem_judge import train_deepchem_judge


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--legacy-baseline", action="store_true")
    source.add_argument("--snapshot", type=Path)
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--intensity-weight", type=float, choices=(0.1, 0.3, 1.0), default=0.3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    if args.legacy_baseline:
        table = load_legacy_baseline(ROOT / "clean_dataset.csv", ROOT / "odor_morgan_tensor_dataset.pt")
        version = args.dataset_version or "legacy-clean-3522"
    else:
        dataset = torch.load(ROOT / "odor_morgan_tensor_dataset.pt", map_location="cpu", weights_only=False)
        labels = tuple(str(value) for value in dataset.label_names)
        table = load_versioned_snapshot(args.snapshot, labels)
        version = args.dataset_version or args.snapshot.stem
    if args.split_manifest:
        split = split_from_payload(load_immutable_manifest(args.split_manifest, table=table))
    else:
        from olfactory.training.splits import chemical_group_split
        import numpy as np
        split = chemical_group_split(table.smiles, np.nan_to_num(table.presence, nan=0.0), seed=42)
    try:
        manifest = train_deepchem_judge(
            table,
            split,
            args.artifact_root,
            dataset_version=version,
            seed=args.seed,
            intensity_weight=args.intensity_weight,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
    except RuntimeError as error:
        parser.error(str(error))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
