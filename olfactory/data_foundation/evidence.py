"""Versioned source evidence, immutable releases and honest label coverage.

Catalog, individual panel ratings and aggregate ratings share storage, not
measurement semantics. A vocabulary approval never approves sensory evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from .repository import SCHEMA_VERSION, utc_now
from ..target_matching import maturity_from_support


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def checked_file(root: Path, name: str, checksum: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Snapshot path is missing or outside its directory")
    if file_hash(path) != checksum:
        raise ValueError(f"File checksum mismatch: {name}")
    return path


def molecular_identity(smiles: str) -> dict:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, sanitize=True)
        if molecule is None:
            raise ValueError("Invalid source SMILES")
        key = Chem.MolToInchiKey(molecule)
    if not key:
        raise ValueError("InChIKey unavailable")
    potential = list(Chem.FindPotentialStereo(molecule))
    unresolved = any(str(info.specified) != "Specified" for info in potential)
    return {
        "raw_smiles": smiles,
        "parent_smiles": smiles,  # No salt/tautomer/protonation transformation.
        "isomeric_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        "connectivity_smiles": Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
        "inchikey": key, "connectivity_key": key.split("-")[0],
        "stereo_state": "UNRESOLVED" if unresolved else "DEFINED" if potential else "ACHIRAL",
        "standardization_log": ["RDKit parse/sanitize and canonicalization only"],
        "identity_review_required": bool(len(Chem.GetMolFrags(molecule)) > 1
                                         or Chem.GetFormalCharge(molecule)
                                         or any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())),
    }


def validate_evidence_records(source: dict, records: Sequence[dict], labels: Sequence[str]) -> list:
    for field in ("source_id", "version", "license_status", "evidence_review"):
        if not source.get(field):
            raise ValueError(f"Evidence source missing {field}")
    if source.get("kind") not in {"CATALOG", "PANEL_INDIVIDUAL", "PANEL_AGGREGATE"}:
        raise ValueError("Unknown source evidence kind")
    if source["evidence_review"] == "APPROVED" and not all(source.get(key) for key in ("reviewer", "protocol_version", "protocol_reference")):
        raise ValueError("Approved evidence requires an identified reviewer and protocol")
    normalized, ids = [], set()
    for record in records:
        record = dict(record)
        if not record.get("record_id") or record["record_id"] in ids:
            raise ValueError("Duplicate or missing source record ID")
        ids.add(record["record_id"])
        record["identity"] = molecular_identity(record["raw_smiles"])
        record.setdefault("context", {})
        for observation in record.get("observations", []):
            if observation.get("descriptor") is not None and observation["descriptor"] not in labels:
                raise ValueError("Descriptor is outside the fixed label contract")
            state = observation.get("presence_state")
            if state not in {"PRESENT", "ABSENT", "UNASSESSED"}:
                raise ValueError("Invalid presence state")
            if observation.get("intensity") is not None and state != "PRESENT":
                raise ValueError("Conditional intensity requires PRESENT")
            if observation.get("evidence_review") == "APPROVED" and source["evidence_review"] != "APPROVED":
                raise ValueError("Observation approval requires a reviewed source protocol")
            if source["kind"] == "CATALOG" and state == "ABSENT":
                raise ValueError("Catalog omission cannot establish absence")
            if observation.get("intensity") is not None:
                value = observation["intensity"]
                if not np.isfinite(value) or not 0 <= value <= 10 or not observation.get("intensity_transform"):
                    raise ValueError("Normalized intensity requires a documented 0–10 transformation")
        normalized.append(record)
    return normalized


def import_reviewed_catalog(service, artifact_dir: Path, *, csv_path: Path, alias_path: Path,
                            review_path: Path, taxonomy_path: Path) -> str:
    root = Path(artifact_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("review_version"):
        raise ValueError("Vocabulary review policy is required")
    for path, field in ((csv_path, "input_csv_sha256"), (alias_path, "alias_config_sha256"),
                        (review_path, "review_policy_sha256"), (taxonomy_path, "source_taxonomy_sha256")):
        if file_hash(path) != manifest.get(field):
            raise ValueError(f"Catalog checksum mismatch: {field}")
    label_hash = hashlib.sha256(json.dumps(list(service.label_names), separators=(",", ":")).encode()).hexdigest()
    if len(service.label_names) != 113 or label_hash != manifest["production_label_order_sha256"]:
        raise ValueError("Production label order mismatch")
    for name, digest in manifest["files"].items():
        checked_file(root, name, digest)
    mapping = json.loads((root / "mapping.json").read_text())
    if any(item["status"] == "REVIEW_REQUIRED" for item in mapping["mappings"]):
        raise ValueError("Unresolved vocabulary review")
    source_id = "clean-master:" + file_hash(manifest_path)
    source = {
        "source_id": source_id, "kind": "CATALOG", "version": manifest["dataset_version"],
        "license_status": "REVIEW_REQUIRED", "evidence_review": "UNREVIEWED",
        "vocabulary_review": manifest["review_version"], "provenance": manifest,
        "mapping": mapping, "label_names": list(service.label_names),
    }
    frame = pd.read_parquet(root / "molecules.parquet")
    observations = pd.read_parquet(root / "draft_assessments.parquet")
    raw = pd.read_csv(csv_path).astype(object).where(lambda x: x.notna(), None)
    if len(frame) != len(raw) or frame.source_row.duplicated().any() or set(frame.source_row) != set(range(len(raw))):
        raise ValueError("Catalog source row reconciliation failed")
    if len(observations) != len(frame) * len(service.label_names):
        raise ValueError("Catalog target matrix is incomplete")
    mapped = observations.groupby("source_row", sort=False)
    records = []
    for molecule in frame.to_dict("records"):
        index = int(molecule["source_row"])
        rows = mapped.get_group(index)
        if rows.descriptor.duplicated().any() or set(rows.descriptor) != service.label_set:
            raise ValueError("Catalog label rows do not match label contract")
        if not rows.presence_state.isin(["PRESENT", "UNASSESSED"]).all() or rows.intensity.notna().any():
            raise ValueError("Catalog contains inferred negatives or intensity")
        identity = molecular_identity(str(raw.iloc[index]["isomeric_smiles"]))
        if identity["inchikey"] != molecule["inchikey"]:
            raise ValueError("Catalog identity mismatch")
        expected = set(json.loads(molecule["mapped_terms"]))
        if set(rows.loc[rows.presence_state == "PRESENT", "descriptor"]) != expected:
            raise ValueError("Catalog target mapping mismatch")
        records.append({
            "record_id": f"row-{index:08d}", "source_identifier": str(index),
            "source_row": index, "raw_smiles": identity["raw_smiles"],
            "raw_payload": raw.iloc[index].to_dict(), "catalog_provenance": molecule,
            "observations": [{"descriptor": row.descriptor, "presence_state": "PRESENT",
                              "intensity": None, "source_terms": json.loads(row.source_terms),
                              "evidence_review": "UNREVIEWED"}
                             for row in rows.itertuples() if row.presence_state == "PRESENT"],
        })
    service.import_evidence_records(source, records)
    return source_id


def publish_evidence_snapshot(service, version: str, source_ids: Sequence[str]) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", version) or not source_ids:
        raise ValueError("Invalid dataset version or empty sources")
    destination = service.snapshot_root / version
    if destination.exists():
        raise FileExistsError(f"Immutable snapshot already exists: {destination}")
    sources, records = service.repository.evidence_records(source_ids)
    if not records:
        raise ValueError("Cannot publish empty evidence snapshot")
    source_by_id = {source["source_id"]: source for source in sources}
    molecular_rows, observation_rows = [], []
    for record in records:
        key = content_hash([record["source_id"], record["record_id"]])
        identity = record["identity"]
        source = source_by_id[record["source_id"]]
        context = record.get("context", {})
        molecular_rows.append({
            "record_key": key, "source_id": record["source_id"], "record_id": record["record_id"],
            **{k: json.dumps(v) if isinstance(v, list) else v for k, v in identity.items()},
            "raw_payload": json.dumps(record, ensure_ascii=False, allow_nan=False),
            "source_kind": source["kind"], "context": json.dumps(context, sort_keys=True),
        })
        for number, observation in enumerate(record.get("observations", [])):
            observation_rows.append({
                "observation_id": content_hash([key, number]), "record_key": key,
                "source_id": record["source_id"], "source_kind": source["kind"],
                "inchikey": identity["inchikey"], "stereo_state": identity["stereo_state"],
                "context_id": content_hash([record["source_id"], context]),
                "descriptor": observation.get("descriptor"),
                "presence_state": observation["presence_state"], "intensity": observation.get("intensity"),
                "evidence_review": observation.get("evidence_review", source["evidence_review"]),
                "license_status": source["license_status"],
                "identity_review_required": identity["identity_review_required"],
                "raw_observation": json.dumps(observation, ensure_ascii=False, allow_nan=False),
            })
    stage = Path(tempfile.mkdtemp(prefix=".evidence-", dir=service.snapshot_root))
    try:
        pd.DataFrame(molecular_rows).to_parquet(stage / "molecules.parquet", index=False)
        pd.DataFrame(observation_rows, columns=(
            "observation_id", "record_key", "source_id", "source_kind", "inchikey", "stereo_state",
            "context_id", "descriptor", "presence_state", "intensity", "evidence_review",
            "license_status", "identity_review_required", "raw_observation",
        )).to_parquet(stage / "observations.parquet", index=False)
        (stage / "sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False, allow_nan=False))
        manifest = {
            "schema_version": SCHEMA_VERSION, "evidence_schema_version": 1,
            "dataset_version": version, "status": "AUDIT_ONLY", "training_eligible": False,
            "created_at": utc_now(), "label_names": list(service.label_names),
            "label_order_sha256": content_hash(list(service.label_names)),
            "row_count": len(records), "observation_count": len(observation_rows),
            "source_manifest": [{"source_id": s["source_id"], "sha256": content_hash(s)} for s in sources],
            "missing_descriptor": "UNASSESSED", "publication_gate": "EVIDENCE_REVIEW_REQUIRED",
            "files": {name: file_hash(stage / name) for name in ("molecules.parquet", "observations.parquet", "sources.json")},
        }
        manifest["manifest_sha256"] = content_hash(manifest)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        # Holding the SQLite write transaction serializes publication of a version.
        with service.repository.transaction() as connection:
            if destination.exists():
                raise FileExistsError(f"Immutable snapshot already exists: {destination}")
            connection.execute("INSERT INTO dataset_snapshots VALUES (?, ?, ?, ?, ?, ?)",
                               (version, str(destination / "observations.parquet"), manifest["files"]["observations.parquet"],
                                str(destination / "manifest.json"), len(records), utc_now()))
            service.repository.audit(connection, "EVIDENCE_SNAPSHOT_CREATED", "dataset_snapshot", version,
                                     {"manifest_sha256": manifest["manifest_sha256"], "status": "AUDIT_ONLY"})
            stage.rename(destination)
        return {**manifest, "manifest_path": destination / "manifest.json"}
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # Only this function's generated staging directory.


def load_evidence_snapshot(path: Path) -> tuple:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    declared = manifest.get("manifest_sha256")
    if declared != content_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Snapshot manifest checksum mismatch")
    if manifest.get("evidence_schema_version") != 1:
        raise ValueError("Unsupported evidence snapshot schema")
    if content_hash(manifest["label_names"]) != manifest["label_order_sha256"]:
        raise ValueError("Snapshot label order mismatch")
    paths = {name: checked_file(path.parent, name, sha) for name, sha in manifest["files"].items()}
    molecules = pd.read_parquet(paths["molecules.parquet"])
    observations = pd.read_parquet(paths["observations.parquet"])
    sources = json.loads(paths["sources.json"].read_text())
    if manifest["source_manifest"] != [{"source_id": s["source_id"], "sha256": content_hash(s)} for s in sources]:
        raise ValueError("Snapshot source manifest mismatch")
    if len(molecules) != manifest["row_count"] or len(observations) != manifest["observation_count"]:
        raise ValueError("Snapshot row count mismatch")
    if molecules.record_key.duplicated().any() or observations.observation_id.duplicated().any():
        raise ValueError("Duplicate evidence identity")
    if not set(observations.record_key) <= set(molecules.record_key):
        raise ValueError("Orphan observations")
    return manifest, molecules, observations


def evidence_coverage(manifest: dict, molecules: pd.DataFrame, observations: pd.DataFrame) -> dict:
    """Count independent molecular identities, never number of panelists."""
    universe = set(molecules.inchikey)
    rows = []
    for label in manifest["label_names"]:
        selected = observations[observations.descriptor == label]
        states = {state: set(selected.loc[selected.presence_state == state, "inchikey"])
                  for state in ("PRESENT", "ABSENT")}
        conflicts = set()
        for (key, _), group in selected.groupby(["inchikey", "context_id"]):
            if {"PRESENT", "ABSENT"} <= set(group.presence_state):
                conflicts.add(key)
        eligible = selected[
            selected.evidence_review.eq("APPROVED") & selected.license_status.eq("APPROVED")
            & selected.source_kind.eq("PANEL_INDIVIDUAL") & ~selected.stereo_state.eq("UNRESOLVED")
            & ~selected.identity_review_required.astype(bool) & ~selected.inchikey.isin(conflicts)
        ]
        pos = set(eligible.loc[eligible.presence_state == "PRESENT", "inchikey"])
        neg = set(eligible.loc[eligible.presence_state == "ABSENT", "inchikey"])
        # Context-dependent identities need a chosen measurement domain before calibration.
        variable = pos & neg
        pos, neg = pos - variable, neg - variable
        catalog = selected[selected.source_kind == "CATALOG"]
        rows.append({
            "descriptor": label, "present": len(states["PRESENT"]), "absent": len(states["ABSENT"]),
            "unassessed": len(universe - states["PRESENT"] - states["ABSENT"]),
            "catalog_positive": catalog.loc[catalog.presence_state == "PRESENT", "inchikey"].nunique(),
            "eligible_positive": len(pos), "eligible_negative": len(neg),
            "maturity": maturity_from_support(len(pos), len(neg)).value,
            "conflict_molecules": len(conflicts), "context_dependent": len(variable),
            "unresolved_stereo": selected.loc[selected.stereo_state == "UNRESOLVED", "inchikey"].nunique(),
            "connectivity_groups": molecules.loc[molecules.inchikey.isin(states["PRESENT"] | states["ABSENT"]), "connectivity_key"].nunique(),
            "sources": sorted(set(selected.source_id)),
            "valid_intensity_molecules": eligible.loc[eligible.intensity.notna(), "inchikey"].nunique(),
            "calibration_eligible": len(pos) >= 10 and len(neg) >= 10,
        })
    priority = {label: index for index, label in enumerate(("musk", "jasmine", "floral", "woody", "citrus"))}
    queue = sorted(rows, key=lambda row: (priority.get(row["descriptor"], 5),
                   min(row["eligible_positive"], row["eligible_negative"]), row["descriptor"]))
    return {"dataset_version": manifest["dataset_version"], "status": "AUDIT_ONLY",
            "source_rows": len(molecules), "unique_molecules": len(universe),
            "connectivity_groups": molecules.connectivity_key.nunique(),
            "unresolved_stereo_molecules": molecules.loc[molecules.stereo_state == "UNRESOLVED", "inchikey"].nunique(),
            "labels": rows, "collection_queue": [row["descriptor"] for row in queue],
            "training_eligible": False,
            "blocked_reasons": ["No released, protocol-reviewed training target table"],
            "production_blocked_reasons": ["Prospective blinded panel not completed"],
            "note": "Present/absent may overlap across conditions; support is never assessor count."}


def audit_target_table(manifest: dict, molecules: pd.DataFrame, observations: pd.DataFrame):
    """Audit-only matrix. Missing labels stay NaN; never handed to training."""
    from ..training.dataset import MolecularTargetTable
    unique = molecules.sort_values(["inchikey", "record_key"]).drop_duplicates("inchikey")
    labels = tuple(manifest["label_names"])
    targets = np.full((len(unique), len(labels)), np.nan, dtype=np.float32)
    by_key = {key: index for index, key in enumerate(unique.inchikey)}
    by_label = {label: index for index, label in enumerate(labels)}
    for (key, label), group in observations.groupby(["inchikey", "descriptor"]):
        if label not in by_label:
            continue
        states = set(group.presence_state) - {"UNASSESSED"}
        if len(states) == 1:
            targets[by_key[key], by_label[label]] = float("PRESENT" in states)
    return MolecularTargetTable(tuple(unique.isomeric_smiles), labels, targets,
                                 np.full_like(targets, np.nan), tuple(unique.source_id), tuple(unique.stereo_state))
