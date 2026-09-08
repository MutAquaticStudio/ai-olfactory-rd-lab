#!/usr/bin/env python3
"""Publish source harmonization and identity decisions; never retrain models."""
import argparse
import json
from pathlib import Path

from olfactory.data_foundation import DataFoundationService
from olfactory.data_foundation.keller_review import finalize_review


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest', type=Path, required=True)
    parser.add_argument('--identity-evidence-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, default=root / 'data/keller_descriptor_policy_v1.json')
    parser.add_argument('--dataset-version', required=True)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    labels = tuple(json.loads((root / 'data/odor_taxonomy_mapping_v1_2.json').read_text())['labels'])
    service = DataFoundationService(args.data_root, labels)
    print(json.dumps(finalize_review(service, parent_path=args.parent_manifest,
        evidence_dir=args.identity_evidence_dir, policy_path=args.policy,
        dataset_version=args.dataset_version), indent=2))


if __name__ == '__main__':
    main()
