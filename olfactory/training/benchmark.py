"""Immutable leakage-resistant benchmark manifests and model comparison helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .dataset import MolecularTargetTable
from .splits import (
    SplitManifest,
    chemical_group_calibrated_split,
    chemical_group_folds,
)


def dataset_fingerprint(table: MolecularTargetTable) -> str:
    """Hash molecule identity and masked targets, including label order."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(table.label_names), separators=(",", ":")).encode())
    for index, smiles in enumerate(table.smiles):
        digest.update(str(smiles).encode("utf-8"))
        digest.update(np.asarray(table.presence[index], dtype=np.float32).tobytes())
        digest.update(np.asarray(table.intensity[index], dtype=np.float32).tobytes())
    return digest.hexdigest()


def split_manifest_payload(split: SplitManifest, *, table: MolecularTargetTable, dataset_version: str) -> Dict[str, object]:
    payload = split.to_dict()
    payload.update(
        {
            "schema_version": 2 if len(split.ratios) == 4 else 1,
            "dataset_version": dataset_version,
            "dataset_sha256": dataset_fingerprint(table),
            "label_names": list(table.label_names),
            "locked_test": True,
        }
    )
    return payload


def build_benchmark_manifest(
    table: MolecularTargetTable,
    *,
    dataset_version: str,
    seed: int = 42,
    similarity_threshold: float = 0.6,
    fold_count: int = 5,
) -> Dict[str, object]:
    """Create the fixed 60/10/15/15 split plus development CV folds."""
    labels = np.nan_to_num(table.presence, nan=0.0)
    split = chemical_group_calibrated_split(
        table.smiles,
        labels,
        seed=seed,
        similarity_threshold=similarity_threshold,
    )
    development = (
        list(split.train_indices)
        + list(split.calibration_indices)
        + list(split.validation_indices)
    )
    folds = chemical_group_folds(
        [table.smiles[index] for index in development],
        labels[development],
        fold_count=fold_count,
        seed=seed,
        similarity_threshold=similarity_threshold,
    )
    payload = split_manifest_payload(split, table=table, dataset_version=dataset_version)
    payload["development_indices"] = development
    payload["development_folds"] = [
        [development[local_index] for local_index in fold] for fold in folds.folds
    ]
    payload["development_fold_hash"] = folds.fold_hash
    payload["cv_seeds"] = [11, 17, 23]
    payload["locked_test_used_for_tuning"] = False
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def save_immutable_manifest(path: Path, payload: Mapping[str, object]) -> Path:
    """Write a split once; refuse accidental mixing of datasets or seeds."""
    destination = Path(path)
    normalized = json.loads(json.dumps(dict(payload), sort_keys=True))
    if destination.exists():
        current = json.loads(destination.read_text(encoding="utf-8"))
        if current != normalized:
            raise FileExistsError(f"Immutable split manifest already exists: {destination}")
        return destination
    declared_hash = normalized.get("manifest_sha256")
    if declared_hash:
        unhashed = dict(normalized)
        unhashed.pop("manifest_sha256", None)
        calculated = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if calculated != declared_hash:
            raise ValueError("Split manifest checksum is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def load_immutable_manifest(path: Path, *, table: MolecularTargetTable | None = None) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "dataset_version", "dataset_sha256", "split_hash", "train_indices", "validation_indices", "test_indices"}
    missing = required - set(payload)
    if int(payload.get("schema_version", 1)) >= 2 and "calibration_indices" not in payload:
        missing.add("calibration_indices")
    if missing:
        raise ValueError(f"Split manifest missing fields: {sorted(missing)}")
    declared_hash = payload.get("manifest_sha256")
    if declared_hash:
        unhashed = dict(payload)
        unhashed.pop("manifest_sha256", None)
        calculated = hashlib.sha256(
            json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if calculated != declared_hash:
            raise ValueError("Split manifest checksum is invalid")
    if table is not None and payload["dataset_sha256"] != dataset_fingerprint(table):
        raise ValueError("Split manifest dataset checksum does not match the supplied table")
    assert_no_leakage(payload)
    return payload


def split_from_payload(payload: Mapping[str, object]) -> SplitManifest:
    """Convert a validated JSON manifest to the training split value object."""
    assert_no_leakage(payload)
    has_calibration = "calibration_indices" in payload
    default_ratios = (0.60, 0.10, 0.15, 0.15) if has_calibration else (0.70, 0.15, 0.15)
    return SplitManifest(
        train_indices=tuple(int(value) for value in payload["train_indices"]),
        validation_indices=tuple(int(value) for value in payload["validation_indices"]),
        test_indices=tuple(int(value) for value in payload["test_indices"]),
        group_ids=tuple(str(value) for value in payload.get("group_ids", [])),
        seed=int(payload.get("seed", 42)),
        ratios=tuple(float(value) for value in payload.get("ratios", default_ratios)),
        similarity_threshold=float(payload.get("similarity_threshold", 0.6)),
        split_hash=str(payload["split_hash"]),
        calibration_indices=tuple(int(value) for value in payload.get("calibration_indices", [])),
    )


def assert_no_leakage(payload: Mapping[str, object]) -> None:
    """Validate disjoint indices and connectivity groups before a run."""
    partition_keys = ["train_indices"]
    if "calibration_indices" in payload:
        partition_keys.append("calibration_indices")
    partition_keys.extend(("validation_indices", "test_indices"))
    partitions = [set(payload[key]) for key in partition_keys]
    if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
        raise ValueError("Benchmark partitions overlap")
    group_ids = payload.get("group_ids", [])
    if not group_ids:
        return
    all_indices = set().union(*partitions)
    if any(int(index) < 0 or int(index) >= len(group_ids) for index in all_indices):
        raise ValueError("Benchmark manifest group_ids do not cover all indices")
    if all_indices != set(range(len(group_ids))):
        raise ValueError("Benchmark manifest must assign every row exactly once")
    split_by_group: Dict[str, int] = {}
    for split_number, key in enumerate(partition_keys):
        for index in payload[key]:
            group = str(group_ids[index])
            previous = split_by_group.setdefault(group, split_number)
            if previous != split_number:
                raise ValueError("Chemical group leakage detected in benchmark manifest")


__all__ = [
    "dataset_fingerprint",
    "build_benchmark_manifest",
    "save_immutable_manifest",
    "load_immutable_manifest",
    "assert_no_leakage",
    "split_from_payload",
]
