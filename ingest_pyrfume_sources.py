#!/usr/bin/env python3
"""Mirror a reviewed Pyrfume source without merging its measurement domain."""

from __future__ import annotations

import argparse
import json

from olfactory.data_foundation.service import default_data_root
from olfactory.data_foundation.sources import load_source_registry, mirror_pyrfume_archive


def main() -> None:
    registry = load_source_registry()
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", choices=sorted(registry["archives"]))
    parser.add_argument(
        "--license-approval-ticket",
        help="Internal review reference; required while registry status is REVIEW_REQUIRED",
    )
    parser.add_argument("--destination", default=str(default_data_root() / "raw"))
    args = parser.parse_args()
    try:
        result = mirror_pyrfume_archive(
            args.archive,
            args.destination,
            license_approval_ticket=args.license_approval_ticket,
        )
    except PermissionError as error:
        raise SystemExit(str(error)) from None
    print(
        json.dumps(
            {
                "archive": result.archive,
                "commit": result.commit,
                "quality_tier": result.quality_tier,
                "manifest_path": str(result.manifest_path),
                "file_count": len(result.files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
