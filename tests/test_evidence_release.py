import numpy as np
import pandas as pd
import pytest

from olfactory.data_foundation import DataFoundationService
from olfactory.data_foundation.evidence import load_evidence_snapshot, evidence_coverage
from olfactory.training.benchmark import build_benchmark_manifest, assert_no_leakage
from olfactory.training.dataset import MolecularTargetTable
from olfactory.target_matching import maturity_from_support


def test_catalog_release_is_immutable_masked_and_has_no_fake_panel(tmp_path):
    service = DataFoundationService(tmp_path, ("floral", "woody"))
    source = {"source_id": "fixture-v1", "kind": "CATALOG", "version": "v1",
              "license_status": "REVIEW_REQUIRED", "evidence_review": "UNREVIEWED"}
    records = [{"record_id": "row-0", "raw_smiles": "CCO", "source_identifier": "not-a-CID",
                "raw_payload": {"odor": "floral"}, "observations": [
                    {"descriptor": "floral", "presence_state": "PRESENT", "intensity": None}]}]
    service.import_evidence_records(source, records)
    service.import_evidence_records(source, records)  # idempotent import
    result = service.create_evidence_snapshot("fixture-release-v1", ["fixture-v1"])
    manifest, molecules, observations = load_evidence_snapshot(result["manifest_path"])
    assert len(molecules) == 1
    assert observations.presence_state.tolist() == ["PRESENT"]
    assert manifest["status"] == "AUDIT_ONLY"
    report = evidence_coverage(manifest, molecules, observations)
    assert report["labels"][0]["present"] == 1
    assert report["labels"][1]["unassessed"] == 1
    assert all(row["maturity"] == "INSUFFICIENT" for row in report["labels"])
    with service.repository.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM assessors").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM stimuli").fetchone()[0] == 0
    assert service.list_snapshots()[0]["dataset_version"] == "fixture-release-v1"
    with pytest.raises(FileExistsError):
        service.create_evidence_snapshot("fixture-release-v1", ["fixture-v1"])
    records[0]["raw_smiles"] = "CCN"
    with pytest.raises(ValueError, match="IMMUTABLE"):
        service.import_evidence_records(source, records)
    path = result["manifest_path"].parent / "molecules.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_evidence_snapshot(result["manifest_path"])


def test_maturity_requires_explicit_negative_support():
    assert maturity_from_support(500, 0).value == "INSUFFICIENT"
    assert maturity_from_support(10, 10).value == "LIMITED_EVIDENCE"
    assert maturity_from_support(50, 50).value == "SUPPORTED"


def test_benchmark_cv_never_uses_calibration_and_rejects_duplicate_rows():
    smiles = ("CCO", "CCN", "CCC", "CCCl", "CCBr", "CCF", "c1ccccc1", "C1CCCCC1",
              "C1CCCC1", "C1CCC1", "C1CC1", "c1ccncc1", "c1ccoc1", "CC(=O)O")
    targets = np.full((len(smiles), 2), np.nan)
    targets[::2, 0] = 1
    targets[1::2, 1] = 0
    table = MolecularTargetTable(smiles, ("floral", "woody"), targets, np.full_like(targets, np.nan),
                                 ("fixture",) * len(smiles), ("ACHIRAL",) * len(smiles))
    manifest = build_benchmark_manifest(table, dataset_version="fixture")
    assert set(manifest["development_indices"]) == set(manifest["train_indices"] + manifest["validation_indices"])
    assert not set(manifest["development_indices"]) & set(manifest["calibration_indices"])
    manifest["train_indices"].append(manifest["train_indices"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        assert_no_leakage(manifest)


def test_fixed_parent_groups_survive_cv_subsetting():
    from olfactory.training.splits import chemical_group_folds
    # A fixed group may unite different scaffolds through earlier curation.
    smiles = ("CCO", "CCN", "c1ccccc1", "C1CCCCC1", "C1CCCC1", "CCBr")
    groups = ("curated-1", "curated-2", "curated-1", "curated-3", "curated-4", "curated-5")
    folds = chemical_group_folds(smiles, np.zeros((6, 2)), fixed_group_ids=groups, fold_count=3)
    membership = {index: number for number, indices in enumerate(folds.folds) for index in indices}
    assert membership[0] == membership[2]


def test_reviewed_ratings_do_not_inflate_support_across_assessors(tmp_path):
    service = DataFoundationService(tmp_path, ("floral",))
    source = {"source_id": "panel-fixture", "kind": "PANEL_INDIVIDUAL", "version": "v1",
              "license_status": "APPROVED", "evidence_review": "APPROVED",
              "reviewer": "fixture-reviewer", "protocol_version": "fixture-v1", "protocol_reference": "fixture://protocol"}
    records = [{"record_id": str(index), "raw_smiles": "CCO", "context": {"dilution": 0.001},
                "observations": [{"descriptor": "floral", "presence_state": "PRESENT", "intensity": None,
                                  "assessor_id": str(index)}]} for index in range(50)]
    service.import_evidence_records(source, records)
    release = service.create_evidence_snapshot("independent-support", ["panel-fixture"])
    manifest, molecules, obs = load_evidence_snapshot(release["manifest_path"])
    report = evidence_coverage(manifest, molecules, obs)
    assert report["labels"][0]["eligible_positive"] == 1
    assert report["labels"][0]["eligible_negative"] == 0


def test_audit_manifest_cannot_be_used_by_training(tmp_path):
    import json
    from olfactory.training.benchmark import load_immutable_manifest
    path = tmp_path / "audit.json"
    path.write_text(json.dumps({"purpose": "AUDIT_ONLY"}))
    with pytest.raises(ValueError, match="AUDIT_ONLY"):
        load_immutable_manifest(path)


def test_failure_before_publication_rolls_back_sqlite_and_staging(tmp_path, monkeypatch):
    service = DataFoundationService(tmp_path, ("floral",))
    service.import_evidence_records(
        {"source_id": "catalog", "kind": "CATALOG", "version": "1", "license_status": "REVIEW_REQUIRED", "evidence_review": "UNREVIEWED"},
        [{"record_id": "one", "raw_smiles": "CCO", "observations": []}],
    )
    def fail(*args, **kwargs):
        raise RuntimeError("simulated interruption")
    monkeypatch.setattr(service.repository, "audit", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        service.create_evidence_snapshot("interrupted", ["catalog"])
    assert service.list_snapshots() == []
    assert not list(service.snapshot_root.iterdir())
