#!/usr/bin/env python3
"""Publish reviewed catalog evidence in the existing Data Foundation, then audit."""
import argparse
import json
from pathlib import Path

from olfactory.data_foundation import DataFoundationService
from olfactory.data_foundation.evidence import load_evidence_snapshot
from olfactory.training.data_audit import build_release_audit

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Verify an existing snapshot and create missing audit output")
    args = parser.parse_args()
    mapping = json.loads((ROOT / "data/odor_taxonomy_mapping_v1_2.json").read_text())
    service = DataFoundationService(args.data_root, tuple(mapping["labels"]))
    source_id = service.import_reviewed_catalog(
        args.reviewed_artifact, csv_path=args.data, alias_path=ROOT / "data/odor_label_aliases_v1.json",
        review_path=ROOT / "data/odor_label_review_v1.json", taxonomy_path=ROOT / "data/osmo_taxonomy_v1_2.json")
    snapshot_path = service.snapshot_root / args.dataset_version / "manifest.json"
    if args.resume and snapshot_path.exists():
        manifest, _, _ = load_evidence_snapshot(snapshot_path)
        if [item["source_id"] for item in manifest["source_manifest"]] != [source_id]:
            raise ValueError("Resume snapshot source mismatch")
    else:
        service.create_evidence_snapshot(args.dataset_version, [source_id])
    audit = build_release_audit(snapshot_path, args.report_dir)
    print(json.dumps({"snapshot": str(snapshot_path), "report": str(args.report_dir),
                      "training_eligible": audit["training_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
