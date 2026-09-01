"""Pinned, license-gated Pyrfume archive mirroring into the private data root."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "source_registry.json"
RAW_BASE = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/{commit}/{archive}/{filename}"


@dataclass(frozen=True)
class MirroredArchive:
    archive: str
    commit: str
    quality_tier: str
    license_approval_ticket: str
    files: List[Dict[str, object]]
    manifest_path: Path


def load_source_registry(path: Path = REGISTRY_PATH) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported source registry schema")
    return payload


def mirror_pyrfume_archive(
    archive: str,
    destination_root: Path,
    *,
    license_approval_ticket: Optional[str],
    registry_path: Path = REGISTRY_PATH,
    timeout_seconds: float = 30.0,
) -> MirroredArchive:
    """Mirror only approved, explicitly listed files at an immutable Git commit."""
    registry = load_source_registry(registry_path)
    archive_config = registry["archives"].get(archive)
    if archive_config is None:
        raise ValueError("UNKNOWN_ARCHIVE")
    if archive_config["license_status"] != "APPROVED" and not license_approval_ticket:
        raise PermissionError(
            "LICENSE_REVIEW_REQUIRED: supply an internal approval ticket before downloading this archive"
        )
    ticket = str(license_approval_ticket or archive_config.get("license_ticket", "REGISTRY_APPROVED"))
    commit = str(registry["pyrfume_data_commit"])
    target = Path(destination_root) / "pyrfume" / archive / commit
    target.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for filename in archive_config["files"]:
        url = RAW_BASE.format(commit=commit, archive=archive, filename=filename)
        request = urllib.request.Request(url, headers={"User-Agent": "Scent-Molecule-Studio/0.5"})
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                content = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"SOURCE_DOWNLOAD_FAILED:{archive}:{filename}") from error
        output = target / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        records.append(
            {
                "filename": filename,
                "url": url,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )
    manifest = {
        "schema_version": 1,
        "archive": archive,
        "pyrfume_data_commit": commit,
        "pyrfume_code_commit": registry["pyrfume_code_commit"],
        "quality_tier": archive_config["quality_tier"],
        "license_status_at_registry": archive_config["license_status"],
        "license_approval_ticket": ticket,
        "training_semantics": archive_config["training_semantics"],
        "files": records,
        "mirrored_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = target / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return MirroredArchive(
        archive=archive,
        commit=commit,
        quality_tier=str(archive_config["quality_tier"]),
        license_approval_ticket=ticket,
        files=records,
        manifest_path=manifest_path,
    )
