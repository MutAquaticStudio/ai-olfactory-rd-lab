"""Append-only SQLite repository for local scientific data."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .contracts import AssessmentInput, StandardizedMolecule


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS molecules (
    molecule_id TEXT PRIMARY KEY,
    raw_smiles TEXT NOT NULL,
    parent_smiles TEXT NOT NULL,
    isomeric_smiles TEXT NOT NULL,
    connectivity_smiles TEXT NOT NULL,
    inchikey TEXT NOT NULL UNIQUE,
    connectivity_key TEXT NOT NULL,
    stereo_state TEXT NOT NULL,
    standardization_log TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_molecules_connectivity ON molecules(connectivity_key);
CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_version TEXT NOT NULL,
    source_license TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_records (
    source_record_pk TEXT PRIMARY KEY,
    batch_id TEXT,
    molecule_id TEXT NOT NULL REFERENCES molecules(molecule_id),
    source_name TEXT NOT NULL,
    source_version TEXT NOT NULL,
    source_license TEXT NOT NULL,
    source_record_id TEXT,
    quality_tier TEXT NOT NULL,
    raw_payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(source_name, source_version, source_record_id)
);
CREATE TABLE IF NOT EXISTS studies (
    study_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    protocol_version TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessors (
    assessor_id TEXT PRIMARY KEY,
    pseudonym TEXT NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS study_sessions (
    session_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES studies(study_id),
    name TEXT NOT NULL,
    started_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(study_id, name)
);
CREATE TABLE IF NOT EXISTS stimuli (
    stimulus_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES study_sessions(session_id),
    molecule_id TEXT NOT NULL REFERENCES molecules(molecule_id),
    blinded_sample_code TEXT NOT NULL,
    concentration REAL NOT NULL,
    concentration_unit TEXT NOT NULL,
    solvent TEXT NOT NULL,
    temperature_c REAL NOT NULL,
    preparation_time_minutes REAL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, blinded_sample_code)
);
CREATE TABLE IF NOT EXISTS descriptor_terms (
    descriptor_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    ontology_version TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ontology_maps (
    map_id TEXT PRIMARY KEY,
    source_term TEXT NOT NULL,
    descriptor_id TEXT NOT NULL REFERENCES descriptor_terms(descriptor_id),
    ontology_version TEXT NOT NULL,
    mapping_note TEXT,
    UNIQUE(source_term, ontology_version)
);
CREATE TABLE IF NOT EXISTS assessments (
    assessment_id TEXT PRIMARY KEY,
    stimulus_id TEXT NOT NULL REFERENCES stimuli(stimulus_id),
    assessor_id TEXT NOT NULL REFERENCES assessors(assessor_id),
    descriptor_id TEXT NOT NULL REFERENCES descriptor_terms(descriptor_id),
    presence_state TEXT NOT NULL,
    intensity REAL,
    confidence REAL NOT NULL,
    replicate_number INTEGER NOT NULL,
    notes TEXT,
    supersedes_assessment_id TEXT REFERENCES assessments(assessment_id),
    ingestion_batch_id TEXT REFERENCES ingestion_batches(batch_id),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dataset_snapshots (
    dataset_version TEXT PRIMARY KEY,
    parquet_path TEXT NOT NULL,
    parquet_sha256 TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_runs (
    run_id TEXT PRIMARY KEY,
    model_family TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    git_commit TEXT,
    split_hash TEXT,
    seed INTEGER,
    metrics TEXT NOT NULL,
    artifact_manifest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class SQLiteDataRepository:
    """Small repository boundary that can later be replaced by PostgreSQL."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            current = connection.execute(
                "SELECT MAX(version) AS version FROM schema_migrations"
            ).fetchone()["version"]
            if current == 1:
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.executescript(
                    """
                    ALTER TABLE assessments RENAME TO assessments_v1;
                    CREATE TABLE assessments (
                        assessment_id TEXT PRIMARY KEY,
                        stimulus_id TEXT NOT NULL REFERENCES stimuli(stimulus_id),
                        assessor_id TEXT NOT NULL REFERENCES assessors(assessor_id),
                        descriptor_id TEXT NOT NULL REFERENCES descriptor_terms(descriptor_id),
                        presence_state TEXT NOT NULL,
                        intensity REAL,
                        confidence REAL NOT NULL,
                        replicate_number INTEGER NOT NULL,
                        notes TEXT,
                        supersedes_assessment_id TEXT REFERENCES assessments(assessment_id),
                        ingestion_batch_id TEXT REFERENCES ingestion_batches(batch_id),
                        created_at TEXT NOT NULL
                    );
                    INSERT INTO assessments SELECT * FROM assessments_v1;
                    DROP TABLE assessments_v1;
                    """
                )
                connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, utc_now()),
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def seed_descriptors(
        self,
        label_names: Sequence[str],
        ontology_version: str = "odor-descriptors-v1",
    ) -> None:
        with self.transaction() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO descriptor_terms(descriptor_id, canonical_name, ontology_version) VALUES (?, ?, ?)",
                [(self._identifier("descriptor", name), name, ontology_version) for name in label_names],
            )

    def create_study(
        self,
        name: str,
        protocol_version: str,
        stage: str,
    ) -> Dict[str, object]:
        study_id = self._identifier("study", name)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT study_id, protocol_version, stage FROM studies WHERE name = ?",
                (name,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO studies(study_id, name, protocol_version, stage, created_at) VALUES (?, ?, ?, ?, ?)",
                    (study_id, name, protocol_version, stage, utc_now()),
                )
                self.audit(
                    connection,
                    "STUDY_CREATED",
                    "study",
                    study_id,
                    {"protocol_version": protocol_version, "stage": stage},
                )
            else:
                study_id = existing["study_id"]
        return {
            "study_id": study_id,
            "name": name,
            "protocol_version": protocol_version if existing is None else existing["protocol_version"],
            "stage": stage if existing is None else existing["stage"],
            "created": existing is None,
        }

    def source_identity_conflict(
        self,
        source_name: str,
        source_version: str,
        source_record_id: Optional[str],
        inchikey: str,
    ) -> bool:
        if not source_record_id:
            return False
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT molecules.inchikey
                FROM source_records
                JOIN molecules USING(molecule_id)
                WHERE source_name = ? AND source_version = ? AND source_record_id = ?
                """,
                (source_name, source_version, source_record_id),
            ).fetchone()
        return row is not None and row["inchikey"] != inchikey

    def stimulus_identity_conflict(
        self,
        study_name: str,
        session_name: str,
        blinded_sample_code: str,
        inchikey: str,
    ) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT molecules.inchikey
                FROM stimuli
                JOIN study_sessions USING(session_id)
                JOIN studies USING(study_id)
                JOIN molecules USING(molecule_id)
                WHERE studies.name = ? AND study_sessions.name = ?
                  AND stimuli.blinded_sample_code = ?
                """,
                (study_name, session_name, blinded_sample_code),
            ).fetchone()
        return row is not None and row["inchikey"] != inchikey

    def assessment_id_exists(self, assessment_id: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        return row is not None

    def assessment_exists(self, item: AssessmentInput, molecule: StandardizedMolecule) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT assessments.assessment_id
                FROM assessments
                JOIN stimuli USING(stimulus_id)
                JOIN study_sessions USING(session_id)
                JOIN studies USING(study_id)
                JOIN assessors USING(assessor_id)
                JOIN descriptor_terms USING(descriptor_id)
                JOIN molecules USING(molecule_id)
                WHERE studies.name = ? AND study_sessions.name = ?
                  AND assessors.pseudonym = ? AND stimuli.blinded_sample_code = ?
                  AND descriptor_terms.canonical_name = ?
                  AND assessments.replicate_number = ? AND molecules.inchikey = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM assessments AS correction
                      WHERE correction.supersedes_assessment_id = assessments.assessment_id
                  )
                """,
                (
                    item.study_name,
                    item.session_name,
                    item.assessor_id,
                    item.blinded_sample_code,
                    item.descriptor,
                    item.replicate_number,
                    molecule.inchikey,
                ),
            ).fetchone()
        if row is None:
            return False
        return item.supersedes_assessment_id != row["assessment_id"]

    @staticmethod
    def _identifier(namespace: str, value: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"scent-studio:{namespace}:{value}"))

    def _molecule_id(self, molecule: StandardizedMolecule) -> str:
        return self._identifier("molecule", molecule.inchikey)

    def create_ingestion_batch(
        self,
        connection: sqlite3.Connection,
        *,
        filename: str,
        raw_sha256: str,
        raw_path: str,
        source_name: str,
        source_version: str,
        source_license: str,
        row_count: int,
    ) -> str:
        batch_id = str(uuid.uuid4())
        connection.execute(
            """INSERT INTO ingestion_batches(
                batch_id, filename, raw_sha256, raw_path, source_name,
                source_version, source_license, row_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                batch_id,
                filename,
                raw_sha256,
                raw_path,
                source_name,
                source_version,
                source_license,
                row_count,
                utc_now(),
            ),
        )
        return batch_id

    def insert_assessment(
        self,
        connection: sqlite3.Connection,
        item: AssessmentInput,
        molecule: StandardizedMolecule,
        batch_id: Optional[str] = None,
    ) -> str:
        now = utc_now()
        molecule_id = self._molecule_id(molecule)
        connection.execute(
            """INSERT OR IGNORE INTO molecules(
                molecule_id, raw_smiles, parent_smiles, isomeric_smiles,
                connectivity_smiles, inchikey, connectivity_key, stereo_state,
                standardization_log, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                molecule_id,
                molecule.raw_smiles,
                molecule.parent_smiles,
                molecule.isomeric_smiles,
                molecule.connectivity_smiles,
                molecule.inchikey,
                molecule.connectivity_key,
                molecule.stereo_state.value,
                json.dumps(molecule.standardization_log),
                now,
            ),
        )

        study_id = self._identifier("study", item.study_name)
        connection.execute(
            "INSERT OR IGNORE INTO studies(study_id, name, protocol_version, stage, created_at) VALUES (?, ?, 'sensory-v1', 'PANEL', ?)",
            (study_id, item.study_name, now),
        )
        assessor_pk = self._identifier("assessor", item.assessor_id)
        connection.execute(
            "INSERT OR IGNORE INTO assessors(assessor_id, pseudonym, created_at) VALUES (?, ?, ?)",
            (assessor_pk, item.assessor_id, now),
        )
        session_id = self._identifier("session", f"{item.study_name}:{item.session_name}")
        connection.execute(
            "INSERT OR IGNORE INTO study_sessions(session_id, study_id, name, created_at) VALUES (?, ?, ?, ?)",
            (session_id, study_id, item.session_name, now),
        )
        stimulus_id = self._identifier("stimulus", f"{session_id}:{item.blinded_sample_code}")
        connection.execute(
            """INSERT OR IGNORE INTO stimuli(
                stimulus_id, session_id, molecule_id, blinded_sample_code,
                concentration, concentration_unit, solvent, temperature_c,
                preparation_time_minutes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stimulus_id,
                session_id,
                molecule_id,
                item.blinded_sample_code,
                item.concentration,
                item.concentration_unit,
                item.solvent,
                item.temperature_c,
                item.preparation_time_minutes,
                now,
            ),
        )
        descriptor_id = self._identifier("descriptor", item.descriptor)
        connection.execute(
            "INSERT OR IGNORE INTO descriptor_terms(descriptor_id, canonical_name, ontology_version) VALUES (?, ?, 'odor-descriptors-v1')",
            (descriptor_id, item.descriptor),
        )
        descriptor_row = connection.execute(
            "SELECT descriptor_id FROM descriptor_terms WHERE canonical_name = ?",
            (item.descriptor,),
        ).fetchone()
        if descriptor_row is None:
            raise RuntimeError("Descriptor term could not be resolved")
        descriptor_id = descriptor_row["descriptor_id"]
        source_pk = str(uuid.uuid4())
        connection.execute(
            """INSERT OR IGNORE INTO source_records(
                source_record_pk, batch_id, molecule_id, source_name, source_version,
                source_license, source_record_id, quality_tier, raw_payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PRIVATE_PANEL', ?, ?)""",
            (
                source_pk,
                batch_id,
                molecule_id,
                item.source_name,
                item.source_version,
                item.source_license,
                item.source_record_id,
                json.dumps(item.to_dict(), ensure_ascii=False),
                now,
            ),
        )
        assessment_id = str(uuid.uuid4())
        connection.execute(
            """INSERT INTO assessments(
                assessment_id, stimulus_id, assessor_id, descriptor_id,
                presence_state, intensity, confidence, replicate_number, notes,
                supersedes_assessment_id, ingestion_batch_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                assessment_id,
                stimulus_id,
                assessor_pk,
                descriptor_id,
                item.presence_state.value,
                item.intensity,
                item.confidence,
                item.replicate_number,
                item.notes,
                item.supersedes_assessment_id,
                batch_id,
                now,
            ),
        )
        self.audit(
            connection,
            "ASSESSMENT_CREATED",
            "assessment",
            assessment_id,
            {"supersedes": item.supersedes_assessment_id},
        )
        return assessment_id

    def audit(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        entity_type: str,
        entity_id: str,
        payload: Dict[str, object],
    ) -> None:
        connection.execute(
            "INSERT INTO audit_events(event_id, event_type, entity_type, entity_id, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                event_type,
                entity_type,
                entity_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )

    def normalized_assessments(self) -> List[Dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT assessments.assessment_id, studies.name AS study_name,
                       study_sessions.name AS session_name,
                       assessors.pseudonym AS assessor_id,
                       stimuli.blinded_sample_code, molecules.raw_smiles,
                       molecules.parent_smiles, molecules.isomeric_smiles,
                       molecules.connectivity_smiles, molecules.inchikey,
                       molecules.connectivity_key, molecules.stereo_state,
                       stimuli.concentration, stimuli.concentration_unit,
                       stimuli.solvent, stimuli.temperature_c,
                       stimuli.preparation_time_minutes,
                       descriptor_terms.canonical_name AS descriptor,
                       assessments.presence_state, assessments.intensity,
                       assessments.confidence, assessments.replicate_number,
                       assessments.notes, assessments.supersedes_assessment_id,
                       assessments.created_at
                FROM assessments
                JOIN stimuli USING(stimulus_id)
                JOIN study_sessions USING(session_id)
                JOIN studies USING(study_id)
                JOIN assessors USING(assessor_id)
                JOIN descriptor_terms USING(descriptor_id)
                JOIN molecules USING(molecule_id)
                ORDER BY assessments.created_at, assessments.assessment_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def register_snapshot(
        self,
        dataset_version: str,
        parquet_path: Path,
        parquet_sha256: str,
        manifest_path: Path,
        row_count: int,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO dataset_snapshots(
                    dataset_version, parquet_path, parquet_sha256,
                    manifest_path, row_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    dataset_version,
                    str(parquet_path),
                    parquet_sha256,
                    str(manifest_path),
                    row_count,
                    utc_now(),
                ),
            )
            self.audit(
                connection,
                "DATASET_SNAPSHOT_CREATED",
                "dataset_snapshot",
                dataset_version,
                {"sha256": parquet_sha256, "row_count": row_count},
            )

    def list_snapshots(self) -> List[Dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM dataset_snapshots ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
