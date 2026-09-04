import hashlib
import json
from pathlib import Path

import pytest

from olfactory.resources import (
    PRIVATE_RESOURCE_FILES,
    RESOURCE_MANIFEST_NAME,
    ResourceBundleError,
    validate_resource_bundle,
)
from olfactory.training.registry import ModelRegistry, sha256_file


def _write_bundle(root: Path) -> None:
    checksums = {}
    for name in PRIVATE_RESOURCE_FILES:
        path = root / name
        path.write_bytes(name.encode("utf-8"))
        checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / RESOURCE_MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "files": checksums}),
        encoding="utf-8",
    )


def test_resource_bundle_validates_all_private_files(tmp_path):
    _write_bundle(tmp_path)

    assert validate_resource_bundle(tmp_path) == tmp_path.resolve()


def test_resource_bundle_reports_missing_manifest(tmp_path):
    with pytest.raises(ResourceBundleError, match="manifest is missing") as error:
        validate_resource_bundle(tmp_path)

    assert error.value.code == "RESOURCE_BUNDLE_MISSING"


def test_resource_bundle_reports_checksum_mismatch(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / PRIVATE_RESOURCE_FILES[0]).write_bytes(b"tampered")

    with pytest.raises(ResourceBundleError, match="checksum mismatch") as error:
        validate_resource_bundle(tmp_path)

    assert error.value.code == "RESOURCE_CHECKSUM_MISMATCH"


def test_resource_bundle_rejects_unsupported_manifest_schema(tmp_path):
    _write_bundle(tmp_path)
    manifest_path = tmp_path / RESOURCE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ResourceBundleError, match="unsupported schema") as error:
        validate_resource_bundle(tmp_path)

    assert error.value.code == "RESOURCE_MANIFEST_INVALID"


def test_resource_bundle_rejects_missing_resource_file(tmp_path):
    _write_bundle(tmp_path)
    (tmp_path / PRIVATE_RESOURCE_FILES[-1]).unlink()

    with pytest.raises(ResourceBundleError, match="file is missing") as error:
        validate_resource_bundle(tmp_path)

    assert error.value.code == "RESOURCE_FILE_MISSING"


def test_registry_can_enforce_bundle_boundary(tmp_path):
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"weights")
    entry = {"weights_path": str(outside), "weights_sha256": sha256_file(outside)}

    assert ModelRegistry(tmp_path / "registry.json").verify_entry(
        entry,
        tmp_path / "bundle",
        require_within_root=True,
    ) is False


def test_registry_verifies_declared_runtime_dataset(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    weights = bundle / "weights.pth"
    dataset = bundle / "dataset.pt"
    weights.write_bytes(b"trusted weights")
    dataset.write_bytes(b"trusted dataset")
    entry = {
        "weights_path": weights.name,
        "weights_sha256": sha256_file(weights),
        "dataset_path": dataset.name,
        "dataset_sha256": sha256_file(dataset),
    }
    registry = ModelRegistry(tmp_path / "registry.json")

    assert registry.verify_entry(entry, bundle, require_within_root=True) is True

    # A locally regenerated resource manifest must not be able to bless a
    # different label/fingerprint tensor than the registry-pinned artifact.
    dataset.write_bytes(b"different dataset")

    assert registry.verify_entry(entry, bundle, require_within_root=True) is False


def test_registry_rejects_dataset_outside_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    weights = bundle / "weights.pth"
    dataset = tmp_path / "outside.pt"
    weights.write_bytes(b"weights")
    dataset.write_bytes(b"dataset")
    entry = {
        "weights_path": weights.name,
        "weights_sha256": sha256_file(weights),
        "dataset_path": str(dataset),
        "dataset_sha256": sha256_file(dataset),
    }

    assert ModelRegistry(tmp_path / "registry.json").verify_entry(
        entry,
        bundle,
        require_within_root=True,
    ) is False
