from dataclasses import replace
from pathlib import Path

import json
from fastapi.testclient import TestClient

from olfactory.academic_evidence import (
    AcademicDocument,
    AcademicEvidenceService,
    AcademicEvidenceStore,
    ReviewState,
    annotate_chunk_provenance,
    evidence_records_from_document,
    extract_structure_mentions,
    normalize_structure,
)
from olfactory.api import AppResources, create_app
from olfactory.models import OdorPredictor, SMILES_LSTM


def _document(content_type: str = "full_text") -> AcademicDocument:
    return AcademicDocument(
        paper_id="paper-test",
        title="A controlled odorant study",
        link="https://doi.org/10.1234/example",
        source="journal",
        doi="10.1234/example",
        published_date="2024",
        content_type=content_type,
        text_sha256="a" * 64,
        source_type="PRIMARY_STUDY",
        license_status="OA_CONFIRMED",
        open_access=content_type == "full_text",
    )


def test_normalize_structure_is_stereo_aware_and_round_trips():
    normalized = normalize_structure("C[C@H](O)C(=O)O")
    assert normalized.rdkit_valid
    assert normalized.stereo_state == "DEFINED"
    assert normalized.isomeric_smiles == "C[C@H](O)C(=O)O"
    assert normalized.connectivity_smiles == "CC(O)C(=O)O"
    assert normalized.exact_identity_ready


def test_invalid_salt_and_name_fail_closed():
    invalid = normalize_structure("C1(CC")
    assert not invalid.rdkit_valid
    assert invalid.review_required

    salt = normalize_structure("CC.O")
    assert salt.rdkit_valid
    assert salt.raw_value == "CC.O"
    assert "MULTIPLE_FRAGMENTS_PRESERVED" in salt.conflict_flags
    assert salt.review_required

    cas = normalize_structure("64-17-5")
    assert not cas.rdkit_valid
    assert cas.error_code == "CAS_REQUIRES_STRUCTURE"
    assert cas.review_required

    name = normalize_structure("hedione")
    assert name.error_code == "NAME_ONLY"
    assert name.review_required


def test_extractor_keeps_identifier_provenance_and_content_type():
    text = "SMILES: CCO. InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3. CAS 64-17-5. Compound name: ethanol."
    mentions = extract_structure_mentions(
        text,
        paper_id="paper-1",
        title="Study",
        link="https://example.org/paper",
        doi="10.1000/test",
        content_type="abstract",
    )
    assert {item.kind for item in mentions} >= {"SMILES", "INCHI", "CAS", "NAME"}
    assert all(item.content_type == "abstract" for item in mentions)
    assert all(item.evidence_excerpt and item.span_start is not None for item in mentions)


def test_chunk_annotation_adds_retrievable_provenance():
    records = evidence_records_from_document(_document(), "SMILES: CCO")
    annotated = annotate_chunk_provenance(records, [{"page_content": "intro"}, {"page_content": "SMILES: CCO"}])
    assert annotated[0].mention.chunk_index == 1


def test_store_requires_review_and_excludes_abstract_by_default(tmp_path):
    store = AcademicEvidenceStore(tmp_path / "evidence.jsonl")
    full_text = evidence_records_from_document(_document(), "SMILES: CCO")
    abstract = evidence_records_from_document(_document("abstract"), "SMILES: CCO")
    store.upsert((*full_text, *abstract), dataset_sha256="dataset-hash")

    pending = store.verify("CCO")
    assert pending.status == "REVIEW_REQUIRED"
    assert len(pending.matches) == 1
    assert pending.matches[0].document.content_type == "full_text"

    accepted = replace(
        full_text[0],
        review_state=ReviewState.ACCEPTED.value,
        mention=replace(full_text[0].mention, chunk_index=0),
    )
    store.upsert((accepted,), dataset_sha256="dataset-hash")
    verified = store.verify("CCO")
    assert verified.status == "EXACT_MATCH"
    assert verified.matches[0].match_level == "EXACT_STEREO"

    with_abstracts = store.verify("CCO", include_abstracts=True)
    assert len(with_abstracts.matches) == 2
    assert any(item.document.content_type == "abstract" for item in with_abstracts.matches)
    assert store.manifest_path.is_file()
    assert store.sources()[0]["evidence_count"] == 2

    rejected = store.set_review_state(accepted.evidence_id, ReviewState.REJECTED)
    assert rejected.review_state == ReviewState.REJECTED.value
    assert '"dataset_sha256": "dataset-hash"' in store.manifest_path.read_text()
    assert store.verify("CCO").status == "NO_EXACT_EVIDENCE"


def test_connectivity_match_with_different_stereo_requires_review(tmp_path):
    store = AcademicEvidenceStore(tmp_path / "evidence.jsonl")
    source = evidence_records_from_document(
        _document(),
        "SMILES: C[C@H](O)C(=O)O",
    )[0]
    store.upsert((replace(
        source,
        review_state=ReviewState.ACCEPTED.value,
        mention=replace(source.mention, chunk_index=0),
    ),))

    opposite = "C[C@@H](O)C(=O)O"
    summary = store.verify(opposite)
    assert summary.status == "REVIEW_REQUIRED"
    assert summary.matches[0].match_level == "EXACT_CONNECTIVITY"
    assert "CONNECTIVITY_MATCH_STEREO_REVIEW" in summary.conflicts


def test_unverified_license_cannot_be_promoted_to_exact(tmp_path):
    store = AcademicEvidenceStore(tmp_path / "evidence.jsonl")
    document = AcademicDocument(
        **{**_document().__dict__, "license_status": "OA_ROUTE_UNVERIFIED", "open_access": False}
    )
    source = evidence_records_from_document(document, "SMILES: CCO")[0]
    source = replace(
        source,
        review_state=ReviewState.ACCEPTED.value,
        mention=replace(source.mention, chunk_index=0),
    )
    store.upsert((source,))
    result = store.verify("CCO")
    assert result.status == "REVIEW_REQUIRED"
    assert "LICENSE_NOT_VERIFIED" in result.conflicts


def test_store_rejects_dataset_hash_mismatch(tmp_path):
    store = AcademicEvidenceStore(tmp_path / "evidence.jsonl")
    record = evidence_records_from_document(_document(), "SMILES: CCO")[0]
    store.upsert((record,), dataset_sha256="dataset-a")
    try:
        store.assert_compatible("dataset-b")
    except ValueError as error:
        assert "dataset hash mismatch" in str(error)
    else:  # pragma: no cover - assertion gives a clearer failure
        raise AssertionError("expected dataset hash mismatch")


def _api_resources(service: AcademicEvidenceService) -> AppResources:
    mapping = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "odor_taxonomy_mapping_v1_2.json")
        .read_text(encoding="utf-8")
    )
    labels = tuple(str(label) for label in mapping["labels"])
    model = OdorPredictor().eval()
    vocab = {"<PAD>": 0, "<END>": 1, "C": 2}
    creator = SMILES_LSTM(len(vocab), vocab["<PAD>"]).eval()
    return AppResources(
        odor_model=model,
        label_names=labels,
        creator_model=creator,
        char_to_idx=vocab,
        idx_to_char=tuple(vocab),
        existing_isomeric_smiles_set=set(),
        pubchem_client=object(),
        academic_evidence_service=service,
    )


def test_academic_evidence_api_has_stable_contract(tmp_path):
    store = AcademicEvidenceStore(tmp_path / "evidence.jsonl")
    source = evidence_records_from_document(_document(), "SMILES: CCO")[0]
    store.upsert((replace(
        source,
        review_state=ReviewState.ACCEPTED.value,
        mention=replace(source.mention, chunk_index=0),
    ),))
    service = AcademicEvidenceService(store)

    with TestClient(create_app(resources=_api_resources(service))) as client:
        found = client.post(
            "/api/v1/academic/evidence/query",
            json={"isomeric_smiles": "CCO"},
        )
        invalid = client.post(
            "/api/v1/academic/evidence/query",
            json={"isomeric_smiles": "not-a-structure"},
        )
        sources = client.get("/api/v1/academic/sources")
        detail = client.get(f"/api/v1/academic/evidence/{source.evidence_id}")

    assert found.status_code == 200
    assert found.json()["status"] == "EXACT_MATCH"
    assert found.json()["matches"][0]["document"]["doi"] == "10.1234/example"
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "INVALID_STRUCTURE_IDENTIFIER"
    assert sources.status_code == 200
    assert sources.json()["sources"][0]["content_type"] == "full_text"
    assert detail.status_code == 200
    assert detail.json()["evidence_id"] == source.evidence_id
