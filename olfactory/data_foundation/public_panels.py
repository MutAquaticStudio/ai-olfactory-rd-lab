"""License-gated, streaming imports of source-specific public panel ratings.

The source rating and its scale are evidence. Conversion into training labels
requires a separate protocol/mapping review; missing slider values stay unknown.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .evidence import checked_file, content_hash, file_hash, molecular_identity
from .repository import SCHEMA_VERSION, utc_now
from .sources import REGISTRY_PATH, load_source_registry
from .keller_trials import DESCRIPTORS, recover_keller_trials


def _validate_cas_correction(service, correction, workbook_config):
    """A CAS-only correction is not approval of identity, ratings or training."""
    import re
    if correction.get("authorization") != "EXPLICIT_USER_CAS_CORRECTION":
        raise PermissionError("CAS correction requires explicit user authorization")
    for field in ("correction_id", "authorized_at", "reason", "source_identifier", "source_names", "evidence_urls"):
        if not correction.get(field):
            raise ValueError(f"CAS correction missing {field}")
    version = correction.get("parent_dataset_version", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", version):
        raise ValueError("Invalid parent dataset version")
    new = correction.get("new_cas", "")
    if not re.fullmatch(r"\d{2,7}-\d{2}-\d", new):
        raise ValueError("Invalid corrected CAS format")
    digits = new.replace("-", "")
    if sum((i + 1) * int(c) for i, c in enumerate(digits[-2::-1])) % 10 != int(digits[-1]):
        raise ValueError("Invalid corrected CAS check digit")
    if not correction.get("old_cas") or correction["old_cas"] == new:
        raise ValueError("CAS correction must change a specific source value")
    if not isinstance(correction["source_names"], list) or not all(isinstance(x, str) and x for x in correction["source_names"]):
        raise ValueError("CAS correction requires exact source names")
    for field in ("expected_measurement_rows", "expected_trials"):
        if type(correction.get(field)) is not int or correction[field] <= 0:
            raise ValueError(f"Invalid {field}")
    root = service.snapshot_root / version
    path = checked_file(root, "manifest.json", correction.get("parent_manifest_sha256"))
    parent = json.loads(path.read_text())
    if (parent.get("dataset_version") != version or parent.get("status") != "AUDIT_ONLY"
            or parent.get("cas_correction") or parent.get("label_names") != list(service.label_names)
            or parent.get("original_workbook", {}).get("sha256") != workbook_config["sha256"]
            or correction.get("original_workbook_sha256") != workbook_config["sha256"]):
        raise ValueError("CAS correction parent/config mismatch")
    for filename, checksum in parent["files"].items():
        checked_file(root, filename, checksum)
    return parent


def _keller_chunks(root: Path, files: dict, labels: tuple, chunk_size: int):
    stimuli = pd.read_csv(files["stimuli.csv"], dtype={"Stimulus": str, "CIDs": str})
    molecules = pd.read_csv(files["molecules.csv"], dtype={"CID": str})
    if stimuli.Stimulus.duplicated().any():
        raise ValueError("Duplicate Keller stimulus identity")
    identities, invalid = {}, set()
    for cid, group in molecules.groupby("CID"):
        values = group.CanonicalSMILES.dropna().unique()
        if len(values) != 1:
            invalid.add(cid)
            continue
        try:
            identities[cid] = molecular_identity(str(values[0]))
        except ValueError:
            invalid.add(cid)
    stimulus_map = stimuli.set_index("Stimulus").to_dict("index")
    offset = 0
    for frame in pd.read_csv(files["behavior.csv"], dtype=str, chunksize=chunk_size):
        required = {"Stimulus", "Subject", "MeasurementValue", "Value"}
        if not required <= set(frame):
            raise ValueError("Unsupported Keller behavior schema")
        rows = []
        for row in frame.to_dict("records"):
            stimulus = stimulus_map.get(row["Stimulus"], {})
            cid = stimulus.get("CIDs")
            identity = identities.get(cid, {})
            source_term = str(row["MeasurementValue"]).strip()
            term = source_term.lower()
            value = row["Value"] if pd.notna(row["Value"]) else None
            try:
                numeric = float(value) if value is not None else None
            except (ValueError, TypeError):
                numeric = None
            warnings = []
            if numeric is not None and not 0 <= numeric <= 100:
                warnings.append("RATING_OUT_OF_RANGE")
                numeric = None
            if cid in invalid or not identity:
                warnings.append("STRUCTURE_REVIEW_REQUIRED")
            if term == "musky":
                warnings.append("MUSKY_SEMANTIC_CONFLICT_WITH_PERFUMERY_MUSK")
            rows.append({
                "source_row": offset, "stimulus_id": row["Stimulus"],
                "assessor_id": row["Subject"] if pd.notna(row["Subject"]) else None,
                "replicate_number": None, "session_id": None,
                "source_identifier": cid, "identifier_type": "SOURCE_REPORTED_CID",
                "isomeric_smiles": identity.get("isomeric_smiles"), "inchikey": identity.get("inchikey"),
                "stereo_state": identity.get("stereo_state", "UNKNOWN"),
                "context": json.dumps({"study": "keller_2016", "concentration": stimulus.get("Concentration"),
                                       "ratio": stimulus.get("Ratio"), "solvent": stimulus.get("Solvent")}, sort_keys=True),
                "source_term": source_term, "descriptor": term if term in labels else None,
                "mapping_state": "EXACT_TERM_PENDING_PROTOCOL" if term in labels else "REVIEW_REQUIRED",
                "raw_value": value, "raw_numeric_value": numeric,
                "scale": "0-100 source slider" if numeric is not None else None,
                "presence_state": "UNASSESSED", "intensity": None,
                "evidence_review": "UNREVIEWED", "warning": ";".join(warnings),
                "source_kind": "PANEL_INDIVIDUAL", "measurement_table": "behavior.csv",
            })
            offset += 1
        yield pd.DataFrame(rows)


def _dravnieks_chunks(root: Path, files: dict, labels: tuple, chunk_size: int):
    """Keep applicability/use aggregates separate; never invent panelists."""
    stimuli = pd.read_csv(files["stimuli.csv"], dtype=str).set_index("Stimulus")
    if stimuli.index.duplicated().any():
        raise ValueError("Duplicate Dravnieks stimulus identity")
    molecular = pd.read_csv(files["molecules.csv"], dtype=str)
    smiles_column = next((name for name in ("IsomericSMILES", "CanonicalSMILES", "SMILES") if name in molecular), None)
    identities = {}
    if "CID" in molecular and smiles_column:
        for cid, group in molecular.groupby("CID"):
            values = group[smiles_column].dropna().unique()
            if len(values) == 1:
                try:
                    identities[cid] = molecular_identity(str(values[0]))
                except ValueError:
                    pass
    for filename in ("behavior_1.csv", "behavior_2.csv"):
        for wide in pd.read_csv(files[filename], dtype=str, chunksize=max(1, chunk_size // 150)):
            long = wide.melt(id_vars="Stimulus", var_name="source_term", value_name="raw_value")
            rows = []
            for row in long.itertuples():
                stimulus = stimuli.loc[row.Stimulus] if row.Stimulus in stimuli.index else None
                context = stimulus.dropna().to_dict() if stimulus is not None else {}
                identity = identities.get(context.get("CID"), {})
                rows.append({
                    "stimulus_id": row.Stimulus, "source_term": row.source_term,
                    "source_identifier": context.get("CID"),
                    "isomeric_smiles": identity.get("isomeric_smiles"), "inchikey": identity.get("inchikey"),
                    "stereo_state": identity.get("stereo_state", "UNKNOWN"),
                    "descriptor": row.source_term.lower() if row.source_term.lower() in labels else None,
                    "raw_value": row.raw_value if pd.notna(row.raw_value) else None,
                    "raw_numeric_value": pd.to_numeric(row.raw_value, errors="coerce"),
                    "measurement_table": filename, "source_kind": "PANEL_AGGREGATE",
                    "assessor_id": None, "replicate_number": None, "presence_state": "UNASSESSED",
                    "intensity": None, "evidence_review": "UNREVIEWED",
                    "context": json.dumps(context),
                    "scale": "source applicability aggregate" if filename == "behavior_1.csv" else "source use aggregate",
                    "mapping_state": "REVIEW_REQUIRED", "warning": "AGGREGATE_NOT_INDIVIDUAL_ASSESSMENT",
                })
            yield pd.DataFrame(rows)


def import_public_panel(service, archive_dir: Path, *, registry_path: Path = REGISTRY_PATH,
                        dataset_version: str, chunk_size: int = 10000,
                        original_workbook: str | None = None,
                        cas_correction: dict | None = None) -> dict:
    """Store normalized ratings outside Git and register in the same SQLite DB."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", dataset_version):
        raise ValueError("Invalid dataset version")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    root = Path(archive_dir).resolve()
    source_manifest = json.loads((root / "source-manifest.json").read_text())
    archive = source_manifest["archive"]
    registry = load_source_registry(registry_path)
    config = registry["archives"][archive]
    if config["license_status"] != "APPROVED":
        raise PermissionError("LICENSE_REVIEW_REQUIRED")
    if source_manifest["pyrfume_data_commit"] != registry["pyrfume_data_commit"]:
        raise ValueError("Source version mismatch")
    files = {item["filename"]: checked_file(root, item["filename"], item["sha256"])
             for item in source_manifest["files"]}
    if set(files) != set(config["files"]):
        raise ValueError("Archive files do not match pinned registry")
    adapters = {"keller_2016": _keller_chunks, "dravnieks_1985": _dravnieks_chunks}
    if archive not in adapters:
        raise ValueError("No panel adapter for this archive")
    workbook_path, workbook_config = None, None
    if original_workbook is not None:
        workbook_config = config.get("original_workbook")
        if archive != "keller_2016" or not workbook_config or original_workbook != workbook_config["filename"]:
            raise ValueError("Original workbook must match the pinned Keller registry filename")
        workbook_path = checked_file(root, original_workbook, workbook_config["sha256"])
    target = service.snapshot_root / dataset_version
    if target.exists():
        raise FileExistsError(f"Immutable panel release already exists: {target}")
    parent = None
    if cas_correction is not None:
        if workbook_path is None:
            raise ValueError("CAS correction requires original Keller workbook recovery")
        parent = _validate_cas_correction(service, cas_correction, workbook_config)
        if parent["source_manifest"] != source_manifest:
            raise ValueError("CAS correction source archive mismatch")
    stage = Path(tempfile.mkdtemp(prefix=".panel-", dir=service.snapshot_root))
    writer = None
    counts, warnings, terms = Counter(), Counter(), Counter()
    trial_ids, structure_review = set(), Counter()
    corrected_trials = set()
    schema = None
    try:
        frames = adapters[archive](root, files, service.label_names, chunk_size)
        if workbook_path is not None:
            frames = recover_keller_trials(frames, workbook_path, workbook_config["sha256"])
        for frame in frames:
            if cas_correction is not None:
                mask = (frame.source_cas.eq(cas_correction["old_cas"])
                        & frame.raw_source_identifier.eq(cas_correction["source_identifier"])
                        & frame.source_odor_name.isin(cas_correction["source_names"]))
                # Preserve literal source values; only the effective CAS changes.
                frame["raw_source_cas"] = frame.source_cas.copy()
                frame["cas_correction_id"] = None
                frame.loc[mask, "source_cas"] = cas_correction["new_cas"]
                frame.loc[mask, "cas_correction_id"] = cas_correction["correction_id"]
                counts["cas_corrected_rows"] += int(mask.sum())
                corrected_trials.update(frame.loc[mask, "trial_id"])
            counts["measurement_rows"] += len(frame)
            counts["missing_values"] += int(frame.raw_value.isna().sum())
            counts["numeric_ratings"] += int(frame.raw_numeric_value.notna().sum())
            counts["explicit_zero_ratings"] += int((frame.raw_numeric_value == 0).sum())
            if archive == "keller_2016":
                descriptor_mask = frame.source_term.isin(DESCRIPTORS)
                counts["explicit_zero_descriptor_ratings"] += int((descriptor_mask & frame.raw_numeric_value.eq(0)).sum())
                counts["explicit_zero_other_ratings"] += int((~descriptor_mask & frame.raw_numeric_value.eq(0)).sum())
                counts["positive_descriptor_ratings"] += int((descriptor_mask & frame.raw_numeric_value.gt(0)).sum())
                counts["blank_descriptor_values"] += int((descriptor_mask & frame.raw_value.isna()).sum())
            if workbook_path is not None:
                for trial in frame.drop_duplicates("trial_id").itertuples():
                    if trial.trial_id not in trial_ids:
                        trial_ids.add(trial.trial_id)
                        counts["recovered_trials"] += 1
                        counts["source_marked_replicate_trials"] += int(trial.source_replicate_marker == "true")
                        counts["detected_trials"] += int(trial.detection_response == "I smell something")
                        counts["undetected_trials"] += int(trial.detection_response == "I can't smell anything")
                counts["unrecorded_descriptor_values_when_detected"] += int((
                    frame.source_term.isin(DESCRIPTORS) & frame.raw_value.isna()
                    & frame.detection_response.eq("I smell something")).sum())
                for record in frame[["source_cas", "raw_source_identifier", "stereo_state", "isomeric_smiles"]].itertuples(index=False, name=None):
                    if record[2] in ("UNRESOLVED", "UNKNOWN"):
                        structure_review[tuple("" if pd.isna(value) else value for value in record)] += 1
            counts["mapped_term_rows"] += int(frame.descriptor.notna().sum())
            counts["unresolved_stereo_rows"] += int(frame.get("stereo_state", pd.Series(dtype=str)).eq("UNRESOLVED").sum())
            warnings.update(warning for value in frame.warning for warning in value.split(";") if warning)
            terms.update(frame.source_term)
            # Fix types before the first batch, including all-null columns.
            numeric_fields = {"raw_numeric_value", "intensity"}
            if schema is None:
                schema = pa.schema([(name, pa.float64() if name in numeric_fields else pa.string()) for name in frame.columns])
                writer = pq.ParquetWriter(stage / "ratings.parquet", schema, compression="zstd")
            for field in schema:
                frame[field.name] = pd.to_numeric(frame[field.name], errors="coerce") if field.name in numeric_fields else frame[field.name].astype("string")
            writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
        if writer is None:
            raise ValueError("Empty panel source")
        writer.close()
        writer = None
        if cas_correction is not None:
            counts["cas_corrected_trials"] = len(corrected_trials)
            if (counts["cas_corrected_rows"] != cas_correction["expected_measurement_rows"]
                    or len(corrected_trials) != cas_correction["expected_trials"]
                    or counts["measurement_rows"] != parent["row_count"]):
                raise ValueError("CAS correction affected-row/trial count mismatch")
            (stage / "cas_correction.json").write_text(json.dumps(
                {**cas_correction, "applied_at": utc_now(), "status": "APPLIED_CAS_ONLY",
                 "sample_identity_approved": False, "training_eligible": False}, indent=2, sort_keys=True))
        report = {
            "archive": archive, "dataset_version": dataset_version, "counts": dict(counts),
            "warnings": dict(warnings), "terms": dict(terms), "status": "REVIEW_REQUIRED",
            "training_eligible": False, "license_status": config["license_status"],
            "blocked_reasons": ["Protocol and descriptor mapping require review",
                                "Missing ratings are not assessed negatives",
                                ("Source trial identity recovered; session/order and repeat agreement remain unverified"
                                 if workbook_path else "Replicate/session identity must be recovered from original study records"),
                                "No reviewed 113-label target table"],
            "production_blocked_reasons": ["Prospective blinded panel not completed"],
        }
        (stage / "review_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        pd.DataFrame([{"source_term": term, "count": count, "proposed_target": "", "reviewer": "",
                       "protocol_reference": "", "decision": "PENDING"} for term, count in sorted(terms.items())]).to_csv(stage / "review_queue.csv", index=False)
        if workbook_path is not None:
            columns = ["source_cas", "raw_source_identifier", "stereo_state", "isomeric_smiles",
                       "measurement_rows", "decision"]
            pd.DataFrame([(*key, count, "REVIEW_REQUIRED") for key, count in sorted(structure_review.items())],
                         columns=columns).to_csv(stage / "structure_review_queue.csv", index=False)
        manifest = {"schema_version": SCHEMA_VERSION, "public_panel_schema_version": 2 if workbook_path else 1,
                    "dataset_version": dataset_version, "status": "AUDIT_ONLY", "training_eligible": False,
                    "created_at": utc_now(), "source_manifest": source_manifest, "source_config": config,
                    "label_names": list(service.label_names), "row_count": counts["measurement_rows"],
                    "files": {p.name: file_hash(p) for p in stage.iterdir()}}
        if workbook_path is not None:
            manifest["original_workbook"] = workbook_config
        if cas_correction is not None:
            manifest["public_panel_schema_version"] = 3
            manifest["cas_correction"] = cas_correction
        manifest["manifest_sha256"] = content_hash(manifest)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        with service.repository.transaction() as connection:
            if target.exists():
                raise FileExistsError("Immutable panel release exists")
            connection.execute("INSERT INTO dataset_snapshots VALUES (?, ?, ?, ?, ?, ?)",
                               (dataset_version, str(target / "ratings.parquet"), manifest["files"]["ratings.parquet"],
                                str(target / "manifest.json"), counts["measurement_rows"], utc_now()))
            source_id = archive + ":" + registry["pyrfume_data_commit"] + ":" + dataset_version
            payload = json.dumps({"source_id": source_id, "kind": "PANEL_INDIVIDUAL" if archive == "keller_2016" else "PANEL_AGGREGATE",
                                  "version": registry["pyrfume_data_commit"], "license_status": "APPROVED",
                                  "evidence_review": "UNREVIEWED", "manifest_path": str(target / "manifest.json")}, sort_keys=True)
            connection.execute("INSERT INTO evidence_sources VALUES (?, ?, ?)", (source_id, payload, utc_now()))
            service.repository.audit(connection, "PUBLIC_PANEL_IMPORTED", "dataset_snapshot", dataset_version,
                                     {"manifest_sha256": manifest["manifest_sha256"], "training_eligible": False})
            if cas_correction is not None:
                service.repository.audit(connection, "PUBLIC_PANEL_CAS_CORRECTED", "dataset_snapshot", dataset_version,
                                         {"correction_id": cas_correction["correction_id"],
                                          "parent_dataset_version": cas_correction["parent_dataset_version"],
                                          "corrected_rows": counts["cas_corrected_rows"], "training_eligible": False})
            stage.rename(target)
        return {"manifest_path": str(target / "manifest.json"), "report": report}
    finally:
        if writer is not None:
            writer.close()
        if stage.exists():
            shutil.rmtree(stage)
