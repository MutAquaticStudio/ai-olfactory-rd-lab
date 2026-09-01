import json

import pandas as pd
import pytest

from olfactory.data_foundation import DataFoundationService, PresenceState
from olfactory.data_foundation.service import standardize_molecule


LABELS = ("fruity", "green", "odorless")


def observation(**overrides):
    row = {
        "study_name": "Pilot study",
        "session_name": "Session 01",
        "assessor_id": "OP-0001",
        "blinded_sample_code": "SMP-0001",
        "smiles": "CCO",
        "descriptor": "fruity",
        "presence_state": PresenceState.PRESENT.value,
        "concentration": 10,
        "concentration_unit": "ppm",
        "solvent": "dipropylene glycol",
        "temperature_c": 22,
        "confidence": 80,
        "replicate_number": 1,
        "intensity": 6,
    }
    row.update(overrides)
    return row


def test_standardization_preserves_identity_levels_and_reports_stereo():
    salt = standardize_molecule("CCO.[Na+]")
    unresolved = standardize_molecule("CC(O)C(=O)O")
    resolved = standardize_molecule("C[C@H](O)C(=O)O")

    assert salt.parent_smiles == "CCO"
    assert salt.connectivity_key == salt.inchikey.split("-")[0]
    assert "LARGEST_FRAGMENT_SELECTED" in salt.standardization_log
    assert unresolved.stereo_state.value == "UNRESOLVED"
    assert resolved.stereo_state.value == "DEFINED"


def test_presence_semantics_do_not_turn_unassessed_into_zero_intensity(tmp_path):
    service = DataFoundationService(tmp_path, LABELS)
    validation = service.validate_assessment(
        observation(presence_state="UNASSESSED", intensity=0)
    )

    assert not validation.is_valid
    assert any(issue.code == "MUST_BE_EMPTY" for issue in validation.issues)


def test_manual_commit_creates_append_only_snapshot_and_duplicate_gate(tmp_path):
    service = DataFoundationService(tmp_path, LABELS)
    result = service.commit_assessment(observation())

    assert result["dataset_snapshot"]["row_count"] == 1
    snapshots = service.list_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["parquet_sha256"] == result["dataset_snapshot"]["parquet_sha256"]
    assert (tmp_path / "snapshots" / snapshots[0]["parquet_path"].split("/")[-1]).exists()

    duplicate = service.validate_assessment(observation())
    assert not duplicate.is_valid
    assert any(issue.code == "DUPLICATE_ASSESSMENT" for issue in duplicate.issues)

    correction = observation(
        intensity=7,
        supersedes_assessment_id=result["assessment_id"],
    )
    corrected = service.commit_assessment(correction)
    assert corrected["dataset_snapshot"]["row_count"] == 2
    rows = service.repository.normalized_assessments()
    assert rows[-1]["supersedes_assessment_id"] == result["assessment_id"]


def test_batch_import_validates_then_commits_exact_bytes(tmp_path):
    service = DataFoundationService(tmp_path, LABELS)
    raw = pd.DataFrame([observation()]).to_csv(index=False).encode("utf-8")

    validation = service.validate_import("panel.csv", raw)
    assert validation.is_valid
    assert validation.validation_token
    committed = service.commit_import(validation.validation_token)

    assert len(committed["assessment_ids"]) == 1
    raw_files = list((tmp_path / "raw").rglob("*.csv"))
    assert len(raw_files) == 1
    assert raw_files[0].read_bytes() == raw

    with pytest.raises(KeyError):
        service.commit_import(validation.validation_token)


def test_source_identifier_cannot_change_molecular_identity(tmp_path):
    service = DataFoundationService(tmp_path, LABELS)
    service.commit_assessment(observation(source_record_id="SRC-1"))

    conflict = service.validate_assessment(
        observation(
            blinded_sample_code="SMP-0002",
            replicate_number=2,
            source_record_id="SRC-1",
            smiles="CCCO",
        )
    )
    assert not conflict.is_valid
    assert any(issue.code == "IDENTITY_CONFLICT" for issue in conflict.issues)
