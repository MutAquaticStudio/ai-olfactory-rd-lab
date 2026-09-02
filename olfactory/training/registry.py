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
        if not artifact.exists() or sha256_file(artifact) != entry.get("weights_sha256"):
            return False
        # Candidate manifests may include an additional calibration checksum;
        # production v1 entries remain valid without it.
        calibration_path = entry.get("calibration_path")
        calibration_sha = entry.get("calibration_sha256")
        if calibration_path and calibration_sha:
            calibration = Path(str(calibration_path))
            if not calibration.is_absolute() and root is not None:
                calibration = root / calibration
            if not calibration.exists() or sha256_file(calibration) != calibration_sha:
                return False
        return True

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

    def promote_after_gate(
        self,
        family: str,
        entry: Dict[str, object],
        decision: object,
    ) -> Dict[str, object]:
        """Promote only an artifact whose explicit scientific gate passed."""
        if not bool(getattr(decision, "eligible", False)):
            reasons = getattr(decision, "blocked_reasons", ())
            raise ValueError(f"Model promotion blocked by quality gate: {', '.join(map(str, reasons))}")
        if not self.verify_entry(entry, self.path.parent):
            raise ValueError("Cannot promote an artifact with an invalid weights checksum")
        promoted = dict(entry)
        promoted["status"] = "PRODUCTION"
        promoted["promoted_after_gate"] = True
        return self.promote(family, promoted)


def verify_artifact_manifest(manifest: Dict[str, object], root: Optional[Path] = None) -> bool:
    """Verify all checksummed files referenced by a candidate artifact."""
    base = Path(root) if root is not None else None
    files = manifest.get("checksums", {})
    if isinstance(files, dict):
        for raw_path, expected in files.items():
            path = Path(str(raw_path))
            if not path.is_absolute() and base is not None:
                path = base / path
            if not path.exists() or sha256_file(path) != expected:
                return False
    weights = manifest.get("weights_path")
    checksum = manifest.get("weights_sha256")
    if weights and checksum:
        path = Path(str(weights))
        if not path.is_absolute() and base is not None:
            path = base / path
        if not path.exists() or sha256_file(path) != checksum:
            return False
    return True


__all__ = ["ModelRegistry", "sha256_file", "verify_artifact_manifest"]
