"""Atomic model-registry manifest with checksum verification."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> Dict[str, object]:
        if not self.path.exists():
            return {"schema_version": 1, "production": {}, "history": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported model registry schema")
        return payload

    def production(self, family: str) -> Optional[Dict[str, object]]:
        entry = self.read().get("production", {}).get(family)
        return dict(entry) if entry else None

    def verify_entry(self, entry: Dict[str, object], root: Optional[Path] = None) -> bool:
        artifact = Path(str(entry["weights_path"]))
        if not artifact.is_absolute() and root is not None:
            artifact = root / artifact
        return artifact.exists() and sha256_file(artifact) == entry.get("weights_sha256")

    def promote(self, family: str, entry: Dict[str, object]) -> Dict[str, object]:
        payload = self.read()
        previous = payload.setdefault("production", {}).get(family)
        if previous:
            payload.setdefault("history", []).append(previous)
        payload["production"][family] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, self.path)
        return payload
