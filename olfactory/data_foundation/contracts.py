"""Stable contracts for sensory observations and import validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PresenceState(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNASSESSED = "UNASSESSED"


class StereoState(str, Enum):
    ACHIRAL = "ACHIRAL"
    DEFINED = "DEFINED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class StandardizedMolecule:
    raw_smiles: str
    parent_smiles: str
    isomeric_smiles: str
    connectivity_smiles: str
    inchikey: str
    connectivity_key: str
    stereo_state: StereoState
    standardization_log: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AssessmentInput:
    study_name: str
    session_name: str
    assessor_id: str
    blinded_sample_code: str
    smiles: str
    descriptor: str
    presence_state: PresenceState
    concentration: float
    concentration_unit: str
    solvent: str
    temperature_c: float
    confidence: float
    replicate_number: int
    intensity: Optional[float] = None
    source_name: str = "private_panel"
    source_version: str = "1"
    source_license: str = "PRIVATE"
    source_record_id: Optional[str] = None
    preparation_time_minutes: Optional[float] = None
    notes: Optional[str] = None
    supersedes_assessment_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["presence_state"] = self.presence_state.value
        return payload


@dataclass(frozen=True)
class ImportIssue:
    row: int
    field: str
    code: str
    message: str
    severity: str = "ERROR"


@dataclass
class ImportValidation:
    filename: str
    sha256: str
    row_count: int
    valid_count: int
    issues: List[ImportIssue] = field(default_factory=list)
    normalized_rows: List[Dict[str, Any]] = field(default_factory=list)
    validation_token: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "ERROR" for issue in self.issues)

    def public_payload(self, preview_limit: int = 20) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "valid_count": self.valid_count,
            "is_valid": self.is_valid,
            "validation_token": self.validation_token,
            "issues": [asdict(issue) for issue in self.issues],
            "preview": self.normalized_rows[:preview_limit],
        }
