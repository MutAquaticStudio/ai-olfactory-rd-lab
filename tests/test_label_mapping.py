import json
import hashlib

import pandas as pd

from olfactory.training.label_mapping import (
    build_master_label_review,
    normalize_source_term,
    parse_term_list,
)


def test_term_parser_handles_json_and_plain_catalog_fields():
    assert parse_term_list('["Floral", "Jasminy"]') == ("floral", "jasminy")
    assert parse_term_list("Woody, Dry") == ("woody", "dry")
    assert parse_term_list(float("nan")) == ()
    assert normalize_source_term("  Black   Currant ") == "black currant"


def test_master_mapping_keeps_omissions_unassessed_and_unknowns_in_review(tmp_path):
    labels = tuple(["floral", "jasmine", "woody"] + [f"label-{index}" for index in range(110)])
    source = tmp_path / "clean_master.csv"
    pd.DataFrame(
        [
            {
                "isomeric_smiles": "CCO",
                "odor_types": '["Floral"]',
                "odor_descriptors": '["Jasminy", "Unmapped nuance"]',
                "sources": '["fixture"]',
            },
            {
                "isomeric_smiles": "CCN",
                "odor_types": '["Woody"]',
                "odor_descriptors": "[]",
                "sources": '["fixture"]',
            },
        ]
    ).to_csv(source, index=False)
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mapping_version": "fixture-v1",
                "aliases": {"jasminy": "jasmine"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "draft"

    result = build_master_label_review(source, labels, aliases, output)

    assessments = pd.read_parquet(output / "draft_assessments.parquet")
    first = assessments[assessments["source_row"] == 0].set_index("descriptor")
    assert first.loc["floral", "presence_state"] == "PRESENT"
    assert first.loc["jasmine", "presence_state"] == "PRESENT"
    assert first.loc["woody", "presence_state"] == "UNASSESSED"
    assert "ABSENT" not in set(assessments["presence_state"])
    review = pd.read_csv(output / "review_queue.csv")
    assert review["source_term"].tolist() == ["unmapped nuance"]
    assert result["training_eligible"] is False
    assert result["report"]["absent_records"] == 0


def test_owner_review_preserves_official_taxonomy_terms_without_forcing_targets(tmp_path):
    labels = tuple(["floral", "woody"] + [f"label-{index}" for index in range(111)])
    source = tmp_path / "clean_master.csv"
    pd.DataFrame(
        [{
            "isomeric_smiles": "CCO",
            "odor_types": '["Woody", "White Floral"]',
            "odor_descriptors": '["Zesty"]',
            "sources": '["fixture"]',
        }]
    ).to_csv(source, index=False)
    aliases = tmp_path / "aliases.json"
    aliases.write_text(
        json.dumps({"schema_version": 1, "mapping_version": "fixture-v1", "aliases": {}}),
        encoding="utf-8",
    )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text(
        json.dumps({
            "version": "1.2",
            "GRAND_FAMILIES": ["Woody"],
            "SUBFAMILIES": ["White Floral"],
            "DESCRIPTORS": [],
            "TEXTURES": [],
            "SENSATIONS": ["Zesty"],
        }),
        encoding="utf-8",
    )
    unresolved = ["white floral", "zesty"]
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({
            "schema_version": 1,
            "review_version": "fixture-review-v1",
            "decision": "SOURCE_TAXONOMY_ONLY",
            "expected_term_count": len(unresolved),
            "expected_terms_sha256": hashlib.sha256(
                json.dumps(sorted(unresolved), separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "source_taxonomy_version": "1.2",
        }),
        encoding="utf-8",
    )
    output = tmp_path / "reviewed"

    result = build_master_label_review(
        source,
        labels,
        aliases,
        output,
        review_policy_path=review,
        source_taxonomy_path=taxonomy,
    )

    mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
    records = {item["source_term"]: item for item in mapping["mappings"]}
    assert records["woody"]["status"] == "EXACT"
    assert records["white floral"] == {
        "occurrences": 1,
        "source_taxonomy_tier": "SUBFAMILY",
        "source_term": "white floral",
        "status": "SOURCE_TAXONOMY_ONLY",
        "target_label": None,
    }
    assert records["zesty"]["source_taxonomy_tier"] == "SENSATION"
    assert pd.read_csv(output / "review_queue.csv").empty
    assessments = pd.read_parquet(output / "draft_assessments.parquet")
    assert assessments.set_index("descriptor").loc["woody", "presence_state"] == "PRESENT"
    assert assessments.set_index("descriptor").loc["floral", "presence_state"] == "UNASSESSED"
    assert result["report"]["review_term_count"] == 0
    assert result["report"]["source_taxonomy_only_term_count"] == 2
    assert result["training_eligible"] is False
