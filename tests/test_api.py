import json
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from rdkit import Chem

import olfactory.api as api_module
from olfactory.api import AppResources, create_app
from olfactory.chemistry import ConformerEnsembleResult, screen_molecule
from olfactory.data_foundation import DataFoundationService
from olfactory.generation import (
    GenerationEvent,
    GenerationPhase,
    GenerationResult,
    ReviewCandidate,
)
from olfactory.models import OdorPredictor, SMILES_LSTM
from olfactory.pubchem import NoveltyResult, NoveltyStatus
from olfactory.references import (
    PubChemProvider,
    ReferenceEvidence,
    ReferenceGate,
    ReferenceGateStatus,
    ReferenceProviderMetadata,
    ReferenceStatus,
    ReferenceVerifier,
)


ROOT = Path(__file__).resolve().parent.parent


class RecordingPubChem:
    def __init__(self):
        self.calls = []

    def verify(self, smiles, *, consent):
        self.calls.append((smiles, consent))
        return NoveltyResult(NoveltyStatus.NOT_FOUND, smiles)


class RecordingExternalCatalog:
    def __init__(self):
        self.calls = []
        self._metadata = ReferenceProviderMetadata(
            provider="LICENSED_CATALOG",
            display_name="Licensed catalog",
            source_type="FRAGRANCE_CATALOG",
            enabled=True,
            external=True,
            query_types=("FULL_INCHIKEY",),
            dataset_version="licensed-test",
            license_status="APPROVED",
        )

    @property
    def metadata(self):
        return self._metadata

    def check(self, query, *, consent):
        self.calls.append((query, consent))
        return ReferenceEvidence(
            "LICENSED_CATALOG",
            ReferenceStatus.NO_MATCH,
            None,
            query.inchi_key,
        )


def make_resources():
    mapping = json.loads(
        (ROOT / "data" / "odor_taxonomy_mapping_v1_2.json").read_text(encoding="utf-8")
    )
    labels = tuple(str(label) for label in mapping["labels"])
    odor_model = OdorPredictor().eval()
    for parameter in odor_model.parameters():
        parameter.data.zero_()
    vocabulary = {"<PAD>": 0, "<END>": 1, "C": 2}
    creator = SMILES_LSTM(len(vocabulary), vocabulary["<PAD>"]).eval()
    return AppResources(
        odor_model=odor_model,
        label_names=labels,
        creator_model=creator,
        char_to_idx=vocabulary,
        idx_to_char=("<PAD>", "<END>", "C"),
        existing_isomeric_smiles_set=set(),
        pubchem_client=RecordingPubChem(),
    )


def test_health_and_meta_expose_stable_contract():
    resources = make_resources()
    with TestClient(create_app(resources=resources)) as client:
        health = client.get("/api/v1/health")
        meta = client.get("/api/v1/meta")

    assert health.status_code == 200
    assert health.json() == {"status": "ready", "ready": True, "resource_error": None}
    assert meta.status_code == 200
    body = meta.json()
    assert len(body["label_names"]) == 113
    assert body["generation_limits"] == {
        "required_candidates": 5,
        "shortlist_count": 3,
        "max_attempts": 200,
        "max_seconds": 120.0,
        "max_event_lines": 30,
        "candidate_stereo_limit": 4,
    }
    assert body["stereo"] == {
        "analysis_option_limit": 16,
        "candidate_variant_limit": 4,
    }
    assert body["conformer_ensemble"]["normal_sampling_count"] == 50
    assert body["conformer_ensemble"]["macrocycle_sampling_count"] == 100
    reference = body["reference_verification"]
    assert reference["required_external_consents"] == ["PUBCHEM"]
    assert [item["provider"] for item in reference["providers"]] == [
        "PUBCHEM",
        "TGSC",
        "SCENTREE",
    ]
    assert reference["providers"][0]["enabled"] is True
    assert reference["providers"][1]["license_status"] == "NOT_CONFIGURED"
    assert reference["providers"][2]["license_status"] == "NOT_CONFIGURED"


def test_analysis_returns_structure_screen_predictions_and_taxonomy():
    with TestClient(create_app(resources=make_resources())) as client:
        response = client.post("/api/v1/analysis", json={"smiles": "CCO"})

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_state"] == "COMPLETE"
    assert body["identifiers"]["canonical_smiles"] == "CCO"
    assert body["structure_2d_svg"].lstrip().startswith("<?xml")
    assert body["chemistry_screen"]["decision"] in {"PASS", "REVIEW", "REJECT"}
    assert len(body["predicted_odor_profile"]["model_output"]) == 113
    assert len(body["predicted_odor_profile"]["taxonomy"]["facets"]) == 11
    assert body["conformer_ensemble"]["available"]
    assert body["prediction_v2"]["model_version"] == "judge-v1-legacy"
    assert body["prediction_v2"]["calibrated"] is False
    assert body["prediction_v2"]["reliability_state"] == "OUT_OF_DOMAIN"
    assert len(body["prediction_v2"]["presence_predictions"]) == 113
    assert body["reference_checks"] == []
    assert body["reference_gate"]["status"] == "NOT_RUN"
    assert body["academic_evidence"]["status"] == "NO_EXACT_EVIDENCE"


def test_invalid_smiles_uses_stable_error_shape():
    with TestClient(create_app(resources=make_resources())) as client:
        response = client.post("/api/v1/analysis", json={"smiles": "not-smiles"})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_SMILES"


def test_analysis_keeps_result_when_3d_embedding_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "build_conformer_ensemble",
        lambda _: ConformerEnsembleResult(
            conformers=(),
            method=None,
            requested_count=50,
            embedded_count=0,
            converged_count=0,
            is_macrocycle=False,
            error="EMBED_FAILED",
        ),
    )
    with TestClient(create_app(resources=make_resources())) as client:
        response = client.post("/api/v1/analysis", json={"smiles": "CCO"})

    assert response.status_code == 200
    assert response.json()["conformer_ensemble"] == {
        "available": False,
        "method": None,
        "requested_count": 50,
        "embedded_count": 0,
        "converged_count": 0,
        "is_macrocycle": False,
        "error": "EMBED_FAILED",
        "conformers": [],
    }


def test_hedione_requires_stereo_selection_before_prediction_or_3d(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Judge and conformer builder must stay locked")

    monkeypatch.setattr(api_module, "predict_probabilities", forbidden)
    monkeypatch.setattr(api_module, "build_conformer_ensemble", forbidden)
    with TestClient(create_app(resources=make_resources())) as client:
        response = client.post(
            "/api/v1/analysis",
            json={"smiles": "CCCCC1C(CC(=O)C1)CC(=O)OC"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_state"] == "STEREO_REQUIRED"
    assert body["unresolved_stereo_elements"] == 2
    assert len(body["stereo_options"]) == 4
    assert body["predicted_odor_profile"] is None
    assert body["conformer_ensemble"] is None


def test_analysis_with_more_than_sixteen_options_requires_manual_input():
    with TestClient(create_app(resources=make_resources())) as client:
        response = client.post(
            "/api/v1/analysis",
            json={"smiles": "CC(F)C(Cl)C(Br)C(I)C(O)C(N)C(S)C"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["analysis_state"] == "STEREO_INPUT_REQUIRED"
    assert body["stereo_options"] == []
    assert body["predicted_odor_profile"] is None
    assert body["conformer_ensemble"] is None


def test_resource_failure_is_degraded_without_exposing_paths(monkeypatch):
    def fail_resources():
        raise FileNotFoundError("/private/model/path")

    monkeypatch.setattr(api_module, "load_app_resources", fail_resources)
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health")
        meta = client.get("/api/v1/meta")

    assert health.json() == {
        "status": "degraded",
        "ready": False,
        "resource_error": "FileNotFoundError",
    }
    assert meta.status_code == 503
    assert "/private/model/path" not in meta.text


def test_unhandled_error_returns_stable_details_without_stack_trace(monkeypatch):
    def fail_analysis(*_):
        raise RuntimeError("sensitive implementation message")

    monkeypatch.setattr(api_module, "analyze_smiles", fail_analysis)
    with TestClient(
        create_app(resources=make_resources()),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/api/v1/analysis", json={"smiles": "CCO"})

    assert response.status_code == 500
    assert response.json() == {
        "detail": {
            "code": "INTERNAL_ERROR",
            "message": "The request could not be completed.",
            "technical_details": "RuntimeError",
        }
    }
    assert "sensitive implementation message" not in response.text


def test_generation_requires_consent_without_network_call():
    resources = make_resources()
    with TestClient(create_app(resources=resources)) as client:
        response = client.post(
            "/api/v1/candidates/stream",
            json={
                "target_descriptors": [resources.label_names[0]],
                "sampling_diversity": 0.8,
                "pubchem_consent": False,
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "PUBCHEM_CONSENT_REQUIRED"
    assert resources.pubchem_client.calls == []


def test_generation_requires_consent_for_every_enabled_external_provider():
    resources = make_resources()
    catalog = RecordingExternalCatalog()
    resources.reference_verifier = ReferenceVerifier(
        [PubChemProvider(resources.pubchem_client), catalog]
    )
    with TestClient(create_app(resources=resources)) as client:
        response = client.post(
            "/api/v1/candidates/stream",
            json={
                "target_descriptors": [resources.label_names[0]],
                "sampling_diversity": 0.8,
                "reference_consents": ["PUBCHEM"],
            },
        )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "REFERENCE_CONSENT_REQUIRED"
    assert resources.pubchem_client.calls == []
    assert catalog.calls == []


def test_generation_stream_orders_progress_before_completion(monkeypatch):
    resources = make_resources()

    def fake_pool(**_):
        yield GenerationEvent(
            phase=GenerationPhase.SAMPLING,
            attempt=1,
            accepted=0,
            invalid=0,
            duplicates=0,
            rejected=0,
            reviews=0,
            found=0,
            unverified=0,
        )
        return GenerationResult(
            accepted_candidates=(),
            review_queue=(),
            attempts=1,
            elapsed_seconds=0.1,
            invalid=0,
            duplicates=0,
            rejected=0,
            found=0,
            unverified=0,
            reached_attempt_limit=False,
            reached_time_limit=False,
        )

    monkeypatch.setattr(api_module, "generate_candidate_pool", fake_pool)
    with TestClient(create_app(resources=resources)) as client:
        response = client.post(
            "/api/v1/candidates/stream",
            json={
                "target_descriptors": [resources.label_names[0]],
                "sampling_diversity": 0.8,
                "pubchem_consent": False,
                "reference_consents": ["PUBCHEM"],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.index("event: progress") < response.text.index("event: complete")
    assert '"phase":"SAMPLING"' in response.text
    assert '"attempts":1' in response.text


def test_generation_completion_exposes_reference_review_evidence(monkeypatch):
    resources = make_resources()
    evidence = ReferenceEvidence(
        provider="TGSC",
        status=ReferenceStatus.UNVERIFIED,
        match_level=None,
        queried_identifier="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        source_version="licensed-test",
        error_code="SNAPSHOT_EXPIRED",
    )
    gate = ReferenceGate(
        ReferenceGateStatus.REVIEW_REQUIRED,
        ("TGSC",),
        "REFERENCE_UNVERIFIED",
    )

    def fake_pool(**_):
        if False:
            yield None
        return GenerationResult(
            accepted_candidates=(),
            review_queue=(
                ReviewCandidate(
                    "CCO",
                    screen_molecule(Chem.MolFromSmiles("CCO")),
                    reference_checks=(evidence,),
                    reference_gate=gate,
                    review_category="REFERENCE",
                ),
            ),
            attempts=1,
            elapsed_seconds=0.1,
            invalid=0,
            duplicates=0,
            rejected=0,
            found=0,
            unverified=0,
            reached_attempt_limit=False,
            reached_time_limit=False,
            reference_unverified=1,
        )

    monkeypatch.setattr(api_module, "generate_candidate_pool", fake_pool)
    with TestClient(create_app(resources=resources)) as client:
        response = client.post(
            "/api/v1/candidates/stream",
            json={
                "target_descriptors": [resources.label_names[0]],
                "sampling_diversity": 0.8,
                "reference_consents": ["PUBCHEM"],
            },
        )

    data_line = next(
        line for line in response.text.splitlines() if line.startswith("data:")
    )
    completion = json.loads(data_line.removeprefix("data:"))
    item = completion["review_queue"][0]
    assert item["review_category"] == "REFERENCE"
    assert item["reference_gate"]["status"] == "REVIEW_REQUIRED"
    assert item["reference_checks"][0]["provider"] == "TGSC"
    assert completion["summary"]["reference_unverified"] == 1


def test_data_intake_contract_validates_and_commits_manual_assessment(tmp_path):
    resources = make_resources()
    resources.data_service = DataFoundationService(tmp_path, resources.label_names)
    descriptor = resources.label_names[0]
    payload = {
        "study_name": "Pilot study",
        "session_name": "Session 01",
        "assessor_id": "OP-0001",
        "blinded_sample_code": "SMP-0001",
        "smiles": "CCO",
        "descriptor": descriptor,
        "presence_state": "PRESENT",
        "concentration": 10,
        "concentration_unit": "ppm",
        "solvent": "dipropylene glycol",
        "temperature_c": 22,
        "confidence": 80,
        "replicate_number": 1,
        "intensity": 6,
    }
    with TestClient(create_app(resources=resources)) as client:
        template = client.get("/api/v1/data/templates")
        validation = client.post("/api/v1/assessments/validate", json=payload)
        response = client.post("/api/v1/assessments", json=payload)
        versions = client.get("/api/v1/datasets/versions")

    assert template.status_code == 200
    assert "presence_state" in template.json()["columns"]
    assert validation.status_code == 200
    assert validation.json()["is_valid"] is True
    assert response.status_code == 200
    assert response.json()["dataset_snapshot"]["row_count"] == 1
    assert len(versions.json()["versions"]) == 1


def test_data_intake_batch_import_uses_validation_token(tmp_path):
    resources = make_resources()
    resources.data_service = DataFoundationService(tmp_path, resources.label_names)
    descriptor = resources.label_names[0]
    csv_text = (
        "study_name,session_name,assessor_id,blinded_sample_code,smiles,descriptor,"
        "presence_state,concentration,concentration_unit,solvent,temperature_c,confidence,replicate_number,intensity\n"
        f"Pilot,Session 01,OP-1,SMP-1,CCO,{descriptor},PRESENT,10,ppm,DPG,22,80,1,5\n"
    )
    with TestClient(create_app(resources=resources)) as client:
        validated = client.post(
            "/api/v1/data/imports/validate",
            files={"file": ("panel.csv", csv_text.encode("utf-8"), "text/csv")},
        )
        committed = client.post(
            "/api/v1/data/imports/commit",
            json={"validation_token": validated.json()["validation_token"]},
        )

    assert validated.status_code == 200
    assert validated.json()["is_valid"]
    assert committed.status_code == 200
    assert len(committed.json()["assessment_ids"]) == 1
