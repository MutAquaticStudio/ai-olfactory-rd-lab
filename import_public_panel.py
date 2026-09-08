#!/usr/bin/env python3
"""Normalize an already mirrored, licensed panel archive into the existing data root."""
import argparse
import json
from pathlib import Path

from olfactory.data_foundation import DataFoundationService
from olfactory.data_foundation.public_panels import import_public_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--original-workbook", help="Pinned Keller XLSX filename inside --archive-dir; restores source trial identifiers")
    parser.add_argument("--cas-correction", type=Path, help="Explicitly authorized, parent-checksummed CAS-only correction JSON")
    args = parser.parse_args()
    labels = tuple(json.loads((Path(__file__).resolve().parent / "data/odor_taxonomy_mapping_v1_2.json").read_text())["labels"])
    service = DataFoundationService(args.data_root, labels)
    print(json.dumps(import_public_panel(service, args.archive_dir, dataset_version=args.dataset_version,
                                        original_workbook=args.original_workbook,
                                        cas_correction=json.loads(args.cas_correction.read_text()) if args.cas_correction else None), indent=2))


if __name__ == "__main__":
    main()
