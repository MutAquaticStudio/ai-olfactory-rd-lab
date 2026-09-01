"""Validation, staging, commit, and immutable snapshot services."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from .contracts import (
    AssessmentInput,
    ImportIssue,
    ImportValidation,
    PresenceState,
    StandardizedMolecule,
    StereoState,
)
from .repository import SCHEMA_VERSION, SQLiteDataRepository


TEMPLATE_COLUMNS = (
    "study_name",
    "session_name",
    "assessor_id",
    "blinded_sample_code",
    "smiles",
    "source_record_id",
    "source_name",
    "source_version",
    "source_license",
    "concentration",
    "concentration_unit",
    "solvent",
    "temperature_c",
    "preparation_time_minutes",
    "descriptor",
    "presence_state",
    "intensity",
    "confidence",
    "replicate_number",
    "notes",
    "supersedes_assessment_id",
)

ALLOWED_CONCENTRATION_UNITS = {
    "M",
    "mM",
    "uM",
    "µM",
    "nM",
    "%v/v",
    "%w/v",
    "ppm",
    "ppb",
}


def default_data_root() -> Path:
    configured = os.environ.get("SCENT_STUDIO_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".scent-molecule-studio").resolve()


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _number(value: Any) -> Optional[float]:
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> Optional[int]:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _column_key(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_")


def standardize_molecule(raw_smiles: str) -> StandardizedMolecule:
    normalized = raw_smiles.strip()
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(normalized, sanitize=True)
    if molecule is None:
        raise ValueError("INVALID_SMILES")

    chooser = rdMolStandardize.LargestFragmentChooser()
    parent = chooser.choose(molecule)
    Chem.SanitizeMol(parent)
    log: List[str] = []
    if parent.GetNumAtoms() != molecule.GetNumAtoms():
        log.append("LARGEST_FRAGMENT_SELECTED")

    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    isomeric = parent_smiles
    connectivity = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    inchikey = Chem.MolToInchiKey(parent)
    if not inchikey:
        raise ValueError("INCHIKEY_UNAVAILABLE")
    connectivity_key = inchikey.split("-")[0]

    potential = Chem.FindPotentialStereo(parent)
    unresolved = any(str(item.specified) == "Unspecified" for item in potential)
    has_defined = any(str(item.specified) == "Specified" for item in potential)
    if unresolved:
        stereo_state = StereoState.UNRESOLVED
    elif has_defined:
        stereo_state = StereoState.DEFINED
    else:
        stereo_state = StereoState.ACHIRAL
    return StandardizedMolecule(
        raw_smiles=normalized,
        parent_smiles=parent_smiles,
        isomeric_smiles=isomeric,
        connectivity_smiles=connectivity,
        inchikey=inchikey,
        connectivity_key=connectivity_key,
        stereo_state=stereo_state,
        standardization_log=tuple(log),
    )


@dataclass
class _StagedImport:
    filename: str
    raw_bytes: bytes
    validation: ImportValidation
    rows: List[Tuple[AssessmentInput, StandardizedMolecule]]


class DataFoundationService:
    """High-level data boundary used by the API and offline tooling."""

    def __init__(
        self,
        root: Optional[Path] = None,
        label_names: Sequence[str] = (),
    ):
        self.root = Path(root or default_data_root())
        self.raw_root = self.root / "raw"
        self.snapshot_root = self.root / "snapshots"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.repository = SQLiteDataRepository(self.root / "studio.sqlite3")
        self.label_names = tuple(str(name) for name in label_names)
        self.label_set = set(self.label_names)
        if self.label_names:
            self.repository.seed_descriptors(self.label_names)
        self._staged: Dict[str, _StagedImport] = {}
        self._stage_lock = threading.Lock()

    @staticmethod
    def template_payload() -> Dict[str, object]:
        return {
            "columns": list(TEMPLATE_COLUMNS),
            "presence_states": [state.value for state in PresenceState],
            "intensity_scale": {"min": 0, "max": 10},
            "confidence_scale": {"min": 0, "max": 100},
            "example": {
                "study_name": "Odorant evaluation — Phase 1",
                "session_name": "Session 01",
                "assessor_id": "OP-0001",
                "blinded_sample_code": "SMP-0001",
                "smiles": "CCO",
                "source_name": "private_panel",
                "source_version": "1",
                "source_license": "PRIVATE",
                "concentration": 10,
                "concentration_unit": "ppm",
                "solvent": "dipropylene glycol",
                "temperature_c": 22,
                "preparation_time_minutes": 30,
                "descriptor": "fruity",
                "presence_state": "PRESENT",
                "intensity": 6,
                "confidence": 80,
                "replicate_number": 1,
            },
        }

    @classmethod
    def template_csv(cls) -> bytes:
        frame = pd.DataFrame([cls.template_payload()["example"]])
        return frame.reindex(columns=TEMPLATE_COLUMNS).to_csv(index=False).encode("utf-8")

    @staticmethod
    def _read_frame(filename: str, raw_bytes: bytes) -> pd.DataFrame:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(io.BytesIO(raw_bytes))
        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        raise ValueError("UNSUPPORTED_FILE_TYPE")

    def _validate_mapping(
        self,
        mapping: Mapping[str, Any],
        row_number: int,
    ) -> Tuple[Optional[AssessmentInput], Optional[StandardizedMolecule], List[ImportIssue]]:
        row = {_column_key(key): value for key, value in mapping.items()}
        issues: List[ImportIssue] = []

        def required(field: str) -> Optional[str]:
            value = _clean(row.get(field))
            if value is None:
                issues.append(ImportIssue(row_number, field, "REQUIRED", f"{field} is required."))
            return value

        study = required("study_name")
        session = required("session_name")
        assessor = required("assessor_id")
        blinded_code = required("blinded_sample_code")
        smiles = required("smiles")
        descriptor = required("descriptor")
        state_text = required("presence_state")
        concentration_unit = required("concentration_unit")
        solvent = required("solvent")

        concentration = _number(row.get("concentration"))
        if concentration is None or concentration <= 0:
            issues.append(ImportIssue(row_number, "concentration", "INVALID_NUMBER", "Concentration must be greater than zero."))
        temperature = _number(row.get("temperature_c"))
        if temperature is None or not (-80 <= temperature <= 150):
            issues.append(ImportIssue(row_number, "temperature_c", "OUT_OF_RANGE", "Temperature must be between -80 and 150 °C."))
        confidence = _number(row.get("confidence"))
        if confidence is None or not (0 <= confidence <= 100):
            issues.append(ImportIssue(row_number, "confidence", "OUT_OF_RANGE", "Confidence must be between 0 and 100."))
        replicate = _integer(row.get("replicate_number"))
        if replicate is None or replicate < 1:
            issues.append(ImportIssue(row_number, "replicate_number", "INVALID_INTEGER", "Replicate number must be a positive integer."))
        prep_time = _number(row.get("preparation_time_minutes"))
        if prep_time is not None and prep_time < 0:
            issues.append(ImportIssue(row_number, "preparation_time_minutes", "OUT_OF_RANGE", "Preparation time cannot be negative."))

        if concentration_unit and concentration_unit not in ALLOWED_CONCENTRATION_UNITS:
            issues.append(
                ImportIssue(
                    row_number,
                    "concentration_unit",
                    "UNKNOWN_UNIT",
                    f"Unsupported concentration unit: {concentration_unit}.",
                )
            )

        presence: Optional[PresenceState] = None
        if state_text:
            try:
                presence = PresenceState(state_text.upper())
            except ValueError:
                issues.append(ImportIssue(row_number, "presence_state", "INVALID_ENUM", "Presence state must be PRESENT, ABSENT, or UNASSESSED."))
        intensity = _number(row.get("intensity"))
        if presence == PresenceState.PRESENT:
            if intensity is None or not (0 <= intensity <= 10):
                issues.append(ImportIssue(row_number, "intensity", "OUT_OF_RANGE", "Present observations require intensity from 0 to 10."))
        elif presence is not None and intensity is not None:
            issues.append(ImportIssue(row_number, "intensity", "MUST_BE_EMPTY", "Intensity must be empty for ABSENT or UNASSESSED."))

        if descriptor and self.label_set and descriptor not in self.label_set:
            issues.append(ImportIssue(row_number, "descriptor", "UNKNOWN_DESCRIPTOR", f"Unknown odor descriptor: {descriptor}."))

        molecule: Optional[StandardizedMolecule] = None
        if smiles:
            try:
                molecule = standardize_molecule(smiles)
            except ValueError as error:
                issues.append(ImportIssue(row_number, "smiles", str(error), "SMILES could not be standardized."))
            else:
                if molecule.stereo_state == StereoState.UNRESOLVED:
                    issues.append(
                        ImportIssue(
                            row_number,
                            "smiles",
                            "UNRESOLVED_STEREO",
                            "Structure has unresolved stereochemistry and will be masked from chiral training targets.",
                            severity="WARNING",
                        )
                    )

        source_name = _clean(row.get("source_name")) or "private_panel"
        source_version = _clean(row.get("source_version")) or "1"
        source_license = _clean(row.get("source_license")) or "PRIVATE"
        source_record_id = _clean(row.get("source_record_id"))
        if molecule and self.repository.source_identity_conflict(
            source_name,
            source_version,
            source_record_id,
            molecule.inchikey,
        ):
            issues.append(ImportIssue(row_number, "source_record_id", "IDENTITY_CONFLICT", "This source identifier is already linked to a different molecular identity."))

        supersedes = _clean(row.get("supersedes_assessment_id"))
        if supersedes and not self.repository.assessment_id_exists(supersedes):
            issues.append(ImportIssue(row_number, "supersedes_assessment_id", "UNKNOWN_ASSESSMENT", "The assessment being corrected does not exist."))
        if molecule and study and session and blinded_code and self.repository.stimulus_identity_conflict(
            study,
            session,
            blinded_code,
            molecule.inchikey,
        ):
            issues.append(ImportIssue(row_number, "blinded_sample_code", "STIMULUS_IDENTITY_CONFLICT", "This blinded sample code is already linked to a different molecule in the session."))

        if any(issue.severity == "ERROR" for issue in issues):
            return None, molecule, issues

        assert all(
            value is not None
            for value in (
                study,
                session,
                assessor,
                blinded_code,
                smiles,
                descriptor,
                presence,
                concentration,
                concentration_unit,
                solvent,
                temperature,
                confidence,
                replicate,
                molecule,
            )
        )
        item = AssessmentInput(
            study_name=study,
            session_name=session,
            assessor_id=assessor,
            blinded_sample_code=blinded_code,
            smiles=smiles,
            descriptor=descriptor,
            presence_state=presence,
            concentration=concentration,
            concentration_unit=concentration_unit,
            solvent=solvent,
            temperature_c=temperature,
            confidence=confidence,
            replicate_number=replicate,
            intensity=intensity,
            source_name=source_name,
            source_version=source_version,
            source_license=source_license,
            source_record_id=source_record_id,
            preparation_time_minutes=prep_time,
            notes=_clean(row.get("notes")),
            supersedes_assessment_id=supersedes,
        )
        if self.repository.assessment_exists(item, molecule):
            issues.append(ImportIssue(row_number, "replicate_number", "DUPLICATE_ASSESSMENT", "This observation already exists; submit a superseding correction instead."))
            return None, molecule, issues
        return item, molecule, issues

    def validate_import(self, filename: str, raw_bytes: bytes) -> ImportValidation:
        if not raw_bytes:
            raise ValueError("EMPTY_UPLOAD")
        frame = self._read_frame(filename, raw_bytes)
        frame.columns = [_column_key(column) for column in frame.columns]
        missing = sorted(set(TEMPLATE_COLUMNS[:5]) - set(frame.columns))
        digest = hashlib.sha256(raw_bytes).hexdigest()
        validation = ImportValidation(filename, digest, len(frame), 0)
        if missing:
            validation.issues.extend(
                ImportIssue(0, field, "MISSING_COLUMN", f"Required column is missing: {field}.")
                for field in missing
            )
            return validation

        parsed: List[Tuple[AssessmentInput, StandardizedMolecule]] = []
        duplicate_keys = set()
        for offset, mapping in enumerate(frame.to_dict(orient="records"), start=2):
            item, molecule, issues = self._validate_mapping(mapping, offset)
            validation.issues.extend(issues)
            if item is None or molecule is None:
                continue
            key = (
                item.study_name,
                item.session_name,
                item.assessor_id,
                item.blinded_sample_code,
                item.descriptor,
                item.replicate_number,
            )
            if key in duplicate_keys:
                validation.issues.append(ImportIssue(offset, "replicate_number", "DUPLICATE_IN_FILE", "Duplicate observation in this upload."))
                continue
            duplicate_keys.add(key)
            parsed.append((item, molecule))
            normalized = item.to_dict()
            normalized.update(
                {
                    "isomeric_smiles": molecule.isomeric_smiles,
                    "connectivity_smiles": molecule.connectivity_smiles,
                    "inchikey": molecule.inchikey,
                    "connectivity_key": molecule.connectivity_key,
                    "stereo_state": molecule.stereo_state.value,
                }
            )
            validation.normalized_rows.append(normalized)

        validation.valid_count = len(parsed)
        if validation.is_valid and validation.valid_count:
            token = str(uuid.uuid4())
            validation.validation_token = token
            with self._stage_lock:
                self._staged[token] = _StagedImport(filename, raw_bytes, validation, parsed)
        return validation

    def validate_assessment(self, mapping: Mapping[str, Any]) -> ImportValidation:
        raw = json.dumps(dict(mapping), ensure_ascii=False, sort_keys=True).encode("utf-8")
        validation = ImportValidation("manual-assessment.json", hashlib.sha256(raw).hexdigest(), 1, 0)
        item, molecule, issues = self._validate_mapping(mapping, 1)
        validation.issues.extend(issues)
        if item is not None and molecule is not None:
            validation.valid_count = 1
            normalized = item.to_dict()
            normalized.update(
                {
                    "isomeric_smiles": molecule.isomeric_smiles,
                    "connectivity_smiles": molecule.connectivity_smiles,
                    "inchikey": molecule.inchikey,
                    "connectivity_key": molecule.connectivity_key,
                    "stereo_state": molecule.stereo_state.value,
                }
            )
            validation.normalized_rows.append(normalized)
        return validation

    def commit_assessment(self, mapping: Mapping[str, Any]) -> Dict[str, object]:
        validation = self.validate_assessment(mapping)
        if not validation.is_valid or validation.valid_count != 1:
            raise ValueError(json.dumps(validation.public_payload(), ensure_ascii=False))
        item, molecule, issues = self._validate_mapping(mapping, 1)
        if item is None or molecule is None or any(issue.severity == "ERROR" for issue in issues):
            raise ValueError("ASSESSMENT_INVALID")
        with self.repository.transaction() as connection:
            assessment_id = self.repository.insert_assessment(connection, item, molecule)
        snapshot = self.create_snapshot()
        return {"assessment_id": assessment_id, "dataset_snapshot": snapshot}

    def commit_import(self, token: str) -> Dict[str, object]:
        with self._stage_lock:
            staged = self._staged.pop(token, None)
        if staged is None:
            raise KeyError("VALIDATION_TOKEN_NOT_FOUND")
        if not staged.validation.is_valid:
            raise ValueError("IMPORT_NOT_VALID")

        raw_dir = self.raw_root / staged.validation.sha256[:2]
        raw_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(staged.filename).suffix.lower()
        raw_path = raw_dir / f"{staged.validation.sha256}{suffix}"
        if raw_path.exists() and hashlib.sha256(raw_path.read_bytes()).hexdigest() != staged.validation.sha256:
            raise RuntimeError("RAW_HASH_COLLISION")
        if not raw_path.exists():
            raw_path.write_bytes(staged.raw_bytes)

        first = staged.rows[0][0]
        assessment_ids: List[str] = []
        with self.repository.transaction() as connection:
            batch_id = self.repository.create_ingestion_batch(
                connection,
                filename=staged.filename,
                raw_sha256=staged.validation.sha256,
                raw_path=str(raw_path),
                source_name=first.source_name,
                source_version=first.source_version,
                source_license=first.source_license,
                row_count=len(staged.rows),
            )
            for item, molecule in staged.rows:
                assessment_ids.append(
                    self.repository.insert_assessment(connection, item, molecule, batch_id)
                )
            self.repository.audit(
                connection,
                "IMPORT_COMMITTED",
                "ingestion_batch",
                batch_id,
                {"sha256": staged.validation.sha256, "rows": len(staged.rows)},
            )
        snapshot = self.create_snapshot()
        return {
            "batch_id": batch_id,
            "assessment_ids": assessment_ids,
            "dataset_snapshot": snapshot,
        }

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def create_snapshot(self) -> Dict[str, object]:
        rows = self.repository.normalized_assessments()
        frame = pd.DataFrame(rows)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        content_hash = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        dataset_version = f"data-{timestamp}-{content_hash[:8]}"
        parquet_path = self.snapshot_root / f"{dataset_version}.parquet"
        manifest_path = self.snapshot_root / f"{dataset_version}.manifest.json"

        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            dir=self.snapshot_root,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            frame.to_parquet(temporary_path, index=False, engine="pyarrow")
            temporary_path.replace(parquet_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        parquet_sha = self._hash_file(parquet_path)
        manifest = {
            "dataset_version": dataset_version,
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": len(frame),
            "parquet_file": parquet_path.name,
            "parquet_sha256": parquet_sha,
            "content_sha256": content_hash,
            "label_semantics": {
                "presence_states": [state.value for state in PresenceState],
                "missing_descriptor": "UNASSESSED",
                "intensity_scale": [0, 10],
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            dir=self.snapshot_root,
            delete=False,
        ) as temporary_manifest:
            json.dump(manifest, temporary_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            manifest_temp_path = Path(temporary_manifest.name)
        manifest_temp_path.replace(manifest_path)
        self.repository.register_snapshot(
            dataset_version,
            parquet_path,
            parquet_sha,
            manifest_path,
            len(frame),
        )
        return manifest

    def list_snapshots(self) -> List[Dict[str, object]]:
        return self.repository.list_snapshots()
