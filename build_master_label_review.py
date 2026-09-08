#!/usr/bin/env python3
"""Build a non-trainable 254→113 draft mapping and review queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from olfactory.training.label_mapping import build_master_label_review


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "data" / "odor_taxonomy_mapping_v1_2.json",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=ROOT / "data" / "odor_label_aliases_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "data" / "clean-master-113-draft-v1",
    )
    parser.add_argument("--review-policy", type=Path)
    parser.add_argument(
        "--source-taxonomy",
        type=Path,
        default=ROOT / "data" / "osmo_taxonomy_v1_2.json",
    )
    parser.add_argument("--dataset-version", default="clean-master-113-draft-v1")
    args = parser.parse_args()
    taxonomy = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = tuple(taxonomy["labels"].keys())
    result = build_master_label_review(
        args.data,
        labels,
        args.aliases,
        args.output,
        dataset_version=args.dataset_version,
        review_policy_path=args.review_policy,
        source_taxonomy_path=args.source_taxonomy if args.review_policy else None,
    )
    print(f"Draft mapping manifest written to {result['manifest_path']}")
    print(json.dumps(result["report"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
