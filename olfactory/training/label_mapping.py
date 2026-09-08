"""Draft clean-master taxonomy mapping without inventing negative labels."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem, rdBase

from .registry import sha256_file


MAPPING_SCHEMA_VERSION = 1


def normalize_source_term(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().lower().split())


def parse_term_list(value: object) -> Tuple[str, ...]:
    if value is None or bool(pd.isna(value)):
        return ()
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part for part in text.split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Odor terms must be a JSON list or comma-separated string")
    return tuple(
        normalized
        for item in parsed
        if (normalized := normalize_source_term(item))
    )


def load_alias_mapping(path: Path, production_labels: Sequence[str]) -> Tuple[str, Dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != MAPPING_SCHEMA_VERSION:
        raise ValueError("Unsupported odor-label alias schema")
    labels = set(map(str, production_labels))
    aliases = {
        normalize_source_term(source): normalize_source_term(target)
        for source, target in payload.get("aliases", {}).items()
    }
    invalid = sorted(set(aliases.values()) - labels)
    if invalid:
        raise ValueError(f"Alias mapping contains labels outside the 113-label contract: {invalid}")
    collisions = sorted(set(aliases) & labels)
    if collisions:
        raise ValueError(f"Aliases must not replace exact production labels: {collisions}")
    return str(payload["mapping_version"]), aliases


def _label_order_sha256(labels: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(labels), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _term_set_sha256(terms: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(terms), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_source_taxonomy_review(
    policy_path: Path,
    taxonomy_path: Path,
    unresolved_terms: Sequence[str],
) -> Tuple[str, Dict[str, str]]:
    """Validate an owner-approved, taxonomy-only decision against a pinned snapshot."""
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    if int(policy.get("schema_version", 0)) != MAPPING_SCHEMA_VERSION:
        raise ValueError("Unsupported odor-label review schema")
    if policy.get("decision") != "SOURCE_TAXONOMY_ONLY":
        raise ValueError("Only the conservative SOURCE_TAXONOMY_ONLY review is supported")
    normalized_terms = tuple(sorted(map(normalize_source_term, unresolved_terms)))
    if int(policy.get("expected_term_count", -1)) != len(normalized_terms):
        raise ValueError("Review policy term count does not match the unresolved vocabulary")
    if policy.get("expected_terms_sha256") != _term_set_sha256(normalized_terms):
        raise ValueError("Review policy checksum does not match the unresolved vocabulary")

    taxonomy = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
    if str(taxonomy.get("version")) != str(policy.get("source_taxonomy_version")):
        raise ValueError("Review policy and source-taxonomy versions do not match")
    tiers: Dict[str, str] = {}
    for key, tier in (
        ("GRAND_FAMILIES", "GRAND_FAMILY"),
        ("SUBFAMILIES", "SUBFAMILY"),
        ("DESCRIPTORS", "DESCRIPTOR"),
        ("TEXTURES", "TEXTURE"),
        ("SENSATIONS", "SENSATION"),
    ):
        for value in taxonomy.get(key, ()):
            normalized = normalize_source_term(value)
            if normalized in tiers:
                raise ValueError(f"Source-taxonomy term appears in multiple tiers: {normalized}")
            tiers[normalized] = tier
    unknown = sorted(set(normalized_terms) - set(tiers))
    if unknown:
        raise ValueError(f"Approved terms are absent from the pinned source taxonomy: {unknown}")
    return str(policy["review_version"]), {term: tiers[term] for term in normalized_terms}


def build_master_label_review(
    input_csv: Path,
    production_labels: Sequence[str],
    alias_path: Path,
    output_dir: Path,
    *,
    dataset_version: str = "clean-master-113-draft-v1",
    review_policy_path: Optional[Path] = None,
    source_taxonomy_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Create a provenance-preserving draft; it is deliberately not trainable."""
    source_path = Path(input_csv).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Draft mapping output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    labels = tuple(normalize_source_term(value) for value in production_labels)
    if len(labels) != 113 or len(set(labels)) != 113:
        raise ValueError("Draft mapping requires exactly 113 unique production labels")
    mapping_version, aliases = load_alias_mapping(alias_path, labels)
    frame = pd.read_csv(source_path)
    required = {"isomeric_smiles", "odor_types", "odor_descriptors", "sources"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Clean-master dataset is missing columns: {sorted(missing)}")

    occurrences: Counter[str] = Counter()
    mapping_kind: Dict[str, str] = {}
    mapped_target: Dict[str, str] = {}
    mapped_by_row: List[Dict[str, List[str]]] = []
    invalid_structures: List[Dict[str, object]] = []
    molecule_records: List[Dict[str, object]] = []
    for row_index, row in frame.iterrows():
        terms = tuple(dict.fromkeys(
            (*parse_term_list(row["odor_types"]), *parse_term_list(row["odor_descriptors"]))
        ))
        targets: Dict[str, List[str]] = defaultdict(list)
        for term in terms:
            occurrences[term] += 1
            if term in labels:
                mapping_kind[term] = "EXACT"
                mapped_target[term] = term
                targets[term].append(term)
            elif term in aliases:
                mapping_kind[term] = "EXPLICIT_ALIAS"
                mapped_target[term] = aliases[term]
                targets[aliases[term]].append(term)
            else:
                mapping_kind.setdefault(term, "REVIEW_REQUIRED")
        raw_smiles = str(row["isomeric_smiles"]).strip()
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(raw_smiles, sanitize=True)
        if molecule is None:
            invalid_structures.append({"source_row": int(row_index), "raw_smiles": raw_smiles})
            mapped_by_row.append({})
            continue
        canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        connectivity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        inchikey = Chem.MolToInchiKey(molecule)
        molecule_records.append(
            {
                "source_row": int(row_index),
                "raw_smiles": raw_smiles,
                "canonical_isomeric_smiles": canonical,
                "connectivity_smiles": connectivity,
                "inchikey": inchikey,
                "connectivity_key": inchikey.split("-")[0] if inchikey else "",
                "sources": str(row["sources"]),
                "raw_terms": json.dumps(terms, ensure_ascii=False),
                "mapped_terms": json.dumps(targets, ensure_ascii=False, sort_keys=True),
                "standardization_log": json.dumps(["RDKit canonicalization only"]),
            }
        )
        mapped_by_row.append(dict(targets))

    review_version = None
    taxonomy_tiers: Dict[str, str] = {}
    unresolved_terms = sorted(
        term for term, status in mapping_kind.items() if status == "REVIEW_REQUIRED"
    )
    if review_policy_path is not None:
        if source_taxonomy_path is None:
            raise ValueError("A pinned source taxonomy is required with a review policy")
        review_version, taxonomy_tiers = load_source_taxonomy_review(
            review_policy_path,
            source_taxonomy_path,
            unresolved_terms,
        )
        for term in unresolved_terms:
            mapping_kind[term] = "SOURCE_TAXONOMY_ONLY"

    molecule_frame = pd.DataFrame(molecule_records)
    molecule_path = output / "molecules.parquet"
    molecule_frame.to_parquet(molecule_path, index=False)
    assessment_records: List[Dict[str, object]] = []
    valid_rows = {int(record["source_row"]): record for record in molecule_records}
    for source_row, targets in enumerate(mapped_by_row):
        molecule = valid_rows.get(source_row)
        if molecule is None:
            continue
        for label in labels:
            source_terms = targets.get(label, [])
            assessment_records.append(
                {
                    "source_row": source_row,
                    "inchikey": molecule["inchikey"],
                    "isomeric_smiles": molecule["canonical_isomeric_smiles"],
                    "descriptor": label,
                    "presence_state": "PRESENT" if source_terms else "UNASSESSED",
                    "intensity": None,
                    "source_terms": json.dumps(source_terms, ensure_ascii=False),
                    "review_state": "DRAFT_UNREVIEWED",
                    "mapping_version": mapping_version,
                }
            )
    assessment_path = output / "draft_assessments.parquet"
    pd.DataFrame(assessment_records).to_parquet(assessment_path, index=False)

    review_rows = [
        {
            "source_term": term,
            "occurrences": int(occurrences[term]),
            "status": mapping_kind[term],
            "source_taxonomy_tier": taxonomy_tiers.get(term, ""),
            "proposed_target": mapped_target.get(term, ""),
            "review_decision": "",
            "reviewer": "",
            "notes": "",
        }
        for term in sorted(occurrences)
        if mapping_kind[term] == "REVIEW_REQUIRED"
    ]
    review_path = output / "review_queue.csv"
    pd.DataFrame(
        review_rows,
        columns=(
            "source_term",
            "occurrences",
            "status",
            "proposed_target",
            "review_decision",
            "reviewer",
            "notes",
        ),
    ).to_csv(review_path, index=False)
    mapping_records = [
        {
            "source_term": term,
            "status": mapping_kind[term],
            "target_label": mapped_target.get(term),
            "source_taxonomy_tier": taxonomy_tiers.get(term),
            "occurrences": int(occurrences[term]),
        }
        for term in sorted(occurrences)
    ]
    mapping_path = output / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "mapping_version": mapping_version,
                "review_version": review_version,
                "production_label_count": len(labels),
                "production_label_order_sha256": _label_order_sha256(labels),
                "mappings": mapping_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = {
        "dataset_version": dataset_version,
        "mapping_version": mapping_version,
        "input_rows": int(len(frame)),
        "valid_molecules": int(len(molecule_records)),
        "invalid_molecules": int(len(invalid_structures)),
        "source_term_count": int(len(occurrences)),
        "exact_term_count": sum(value == "EXACT" for value in mapping_kind.values()),
        "alias_term_count": sum(value == "EXPLICIT_ALIAS" for value in mapping_kind.values()),
        "review_term_count": sum(value == "REVIEW_REQUIRED" for value in mapping_kind.values()),
        "source_taxonomy_only_term_count": sum(
            value == "SOURCE_TAXONOMY_ONLY" for value in mapping_kind.values()
        ),
        "source_taxonomy_only_occurrences": sum(
            occurrences[term]
            for term, value in mapping_kind.items()
            if value == "SOURCE_TAXONOMY_ONLY"
        ),
        "present_records": sum(bool(values) for targets in mapped_by_row for values in targets.values()),
        "absent_records": 0,
        "training_eligible": False,
        "blocked_reasons": (
            (["mapping_requires_human_review"] if any(
                value == "REVIEW_REQUIRED" for value in mapping_kind.values()
            ) else [])
            + [
                "source_taxonomy_only_terms_are_not_113_label_targets",
                "catalog_omissions_are_unassessed_not_negative",
                "no_reviewed_panel_intensity",
            ]
        ),
        "invalid_structures": invalid_structures,
    }
    report_path = output / "coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "mapping_version": mapping_version,
        "review_version": review_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(source_path),
        "input_csv_sha256": sha256_file(source_path),
        "alias_config_sha256": sha256_file(Path(alias_path)),
        "review_policy_sha256": (
            sha256_file(Path(review_policy_path)) if review_policy_path is not None else None
        ),
        "source_taxonomy_sha256": (
            sha256_file(Path(source_taxonomy_path)) if source_taxonomy_path is not None else None
        ),
        "production_label_order_sha256": _label_order_sha256(labels),
        "files": {
            path.name: sha256_file(path)
            for path in (molecule_path, assessment_path, review_path, mapping_path, report_path)
        },
        "training_eligible": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "manifest_path": str(manifest_path), "report": report}


__all__ = [
    "build_master_label_review",
    "load_alias_mapping",
    "load_source_taxonomy_review",
    "normalize_source_term",
    "parse_term_list",
]
