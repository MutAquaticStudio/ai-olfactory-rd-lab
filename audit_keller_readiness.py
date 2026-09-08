#!/usr/bin/env python3
"""Generate local-only, source-separated evidence readiness reports; no training."""
import argparse
import json
from pathlib import Path

from olfactory.data_foundation.keller_readiness import build_readiness_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-manifest', type=Path, required=True)
    parser.add_argument('--catalog-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_readiness_report(args.panel_manifest, args.catalog_manifest, args.output), indent=2))
