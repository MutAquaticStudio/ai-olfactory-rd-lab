#!/usr/bin/env python3
"""Copy private model resources into a checksum-verified local bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

# Allow ``python scripts/prepare_resource_bundle.py`` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from olfactory.resources import (
    PRIVATE_RESOURCE_FILES,
    RESOURCE_MANIFEST_NAME,
    default_resource_dir,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest(target: Path, checksums: dict[str, str]) -> None:
    payload = {
        "schema_version": 1,
        "files": checksums,
    }
    temporary = target / f".{RESOURCE_MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target / RESOURCE_MANIFEST_NAME)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing the three current private resources",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=default_resource_dir(),
        help="Private bundle destination",
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if source == target:
        raise SystemExit("Source and target must be different directories.")

    missing = [name for name in PRIVATE_RESOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit("Missing private resources: " + ", ".join(missing))

    target.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for name in PRIVATE_RESOURCE_FILES:
        destination = target / name
        atomic_copy(source / name, destination)
        checksums[name] = sha256_file(destination)
    write_manifest(target, checksums)
    print(f"Prepared private resource bundle: {target}")
    for name, checksum in checksums.items():
        print(f"{name}: {checksum}")


if __name__ == "__main__":
    main()
