from __future__ import annotations

from typing import Any, Dict, List

import pytest

from olfactory.receptor_binding import (
    BOLTZ_MODEL,
    BoltzConfig,
    BoltzReceptorBindingProvider,
    ReceptorBindingError,
    build_boltz_request,
    build_boltz_provider_from_env,
    normalize_boltz_result,
    normalize_isomeric_smiles,
    normalize_receptor_sequence,
)


class FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self, responses: List[FakeResponse]):
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def request(self, method, url, *, headers, json, timeout):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected network request")
        return self.responses.pop(0)


def test_normalization_rejects_ambiguous_protein_and_invalid_smiles():
    assert normalize_receptor_sequence(" ACDE\nFG ") == "ACDEFG"
    assert normalize_isomeric_smiles(" CCO ") == "CCO"

    with pytest.raises(ReceptorBindingError) as sequence_error:
        normalize_receptor_sequence("ACDX")
    assert sequence_error.value.code == "INVALID_RECEPTOR_SEQUENCE"

    with pytest.raises(ReceptorBindingError) as smiles_error:
        normalize_isomeric_smiles("not-smiles")
    assert smiles_error.value.code == "INVALID_SMILES"


def test_request_uses_explicit_protein_ligand_binding_and_stable_idempotency():
    first = build_boltz_request("ACDEFG", "CCO")
    second = build_boltz_request(" ACD EFG ", " CCO ")

    assert first == second
    assert first["model"] == BOLTZ_MODEL
    assert first["input"] == {
        "entities": [
            {"chain_ids": ["A"], "type": "protein", "value": "ACDEFG"},
            {"chain_ids": ["B"], "type": "ligand_smiles", "value": "CCO"},
        ],
        "binding": {"type": "ligand_protein_binding", "binder_chain_id": "B"},
    }
    assert str(first["idempotency_key"]).startswith("nox-or-binding-v1-")


def test_config_identifies_workspace_test_and_live_keys(monkeypatch):
    test_config = BoltzConfig("sk_bc_ws_test_example")
    assert test_config.mode == "test"
    assert test_config.key_scope == "workspace"

    live_config = BoltzConfig("sk_bc_admin_live_example")
    assert live_config.mode == "live"
    assert live_config.key_scope == "admin"

    monkeypatch.delenv("BOLTZ_API_KEY", raising=False)
    assert build_boltz_provider_from_env() is None


def test_cost_estimate_uses_api_key_header_and_never_exposes_key_in_result():
    session = RecordingSession(
        [
            FakeResponse(
                200,
                {
                    "estimated_cost_usd": "0.0123",
                    "breakdown": {
                        "application": "structure_and_binding",
                        "cost_per_unit_usd": "0.0123",
                        "num_units": 1,
                    },
                    "disclaimer": "estimate",
                },
            )
        ]
    )
    provider = BoltzReceptorBindingProvider(
        BoltzConfig("sk_bc_ws_test_secret", timeout_seconds=7.0),
        session=session,
    )

    result = provider.estimate_cost("ACDEFG", "CCO")

    assert result["estimated_cost_usd"] == "0.0123"
    assert result["mode"] == "test"
    assert "secret" not in repr(result)
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/compute/v1/predictions/structure-and-binding/estimate-cost")
    assert call["headers"]["x-api-key"] == "sk_bc_ws_test_secret"
    assert call["timeout"] == 7.0
    assert call["json"]["input"]["binding"]["binder_chain_id"] == "B"


def test_result_is_normalized_as_research_evidence_not_activation():
    payload = {
        "id": "pred_test_1",
        "status": "succeeded",
        "livemode": False,
        "model": "boltz-2.1",
        "version": "2.1-test",
        "idempotency_key": "nox-or-binding-v1-test",
        "created_at": "2026-09-17T00:00:00Z",
        "completed_at": "2026-09-17T00:00:01Z",
        "input": {
            "entities": [
                {"chain_ids": ["A"], "type": "protein", "value": "ACDEFG"},
                {"chain_ids": ["B"], "type": "ligand_smiles", "value": "CCO"},
            ]
        },
        "output": {
            "binding_metrics": {
                "type": "ligand_protein_binding_metrics",
                "binding_confidence": 0.81,
                "optimization_score": 0.62,
            },
            "best_sample": {
                "metrics": {
                    "structure_confidence": 0.77,
                    "ptm": 0.72,
                    "iptm": 0.69,
                    "ligand_iptm": 0.66,
                    "complex_plddt": 0.79,
                    "complex_iplddt": 0.74,
                    "complex_pde": 2.1,
                    "complex_ipde": 3.2,
                },
                "structure": {
                    "url": "https://example.invalid/complex.cif",
                    "url_expires_at": "2026-09-18T00:00:00Z",
                },
            },
        },
    }

    evidence = normalize_boltz_result(payload)

    assert evidence["evidence_type"] == "PREDICTED_RECEPTOR_BINDING"
    assert evidence["provider"] == "BOLTZ"
    assert evidence["binding_metrics"] == {
        "binding_confidence": pytest.approx(0.81),
        "optimization_score": pytest.approx(0.62),
        "type": "ligand_protein_binding_metrics",
    }
    assert evidence["structure_metrics"]["ligand_iptm"] == pytest.approx(0.66)
    assert evidence["input_fingerprints"]["receptor_sequence_sha256"]
    assert evidence["input_fingerprints"]["ligand_smiles_sha256"]
    serialized = repr(evidence)
    assert "ACDEFG" not in serialized
    assert "Predicted binding is not evidence of receptor activation" in serialized


def test_provider_start_and_retrieve_keep_vendor_output_behind_normalizer():
    start_payload = {
        "id": "pred_123",
        "status": "pending",
        "livemode": False,
        "model": "boltz-2.1",
        "version": "2.1-test",
        "input": {
            "entities": [
                {"chain_ids": ["A"], "type": "protein", "value": "ACDEFG"},
                {"chain_ids": ["B"], "type": "ligand_smiles", "value": "CCO"},
            ]
        },
        "output": None,
    }
    retrieve_payload = {
        **start_payload,
        "status": "succeeded",
        "output": {
            "binding_metrics": {
                "type": "ligand_protein_binding_metrics",
                "binding_confidence": 0.7,
                "optimization_score": 0.5,
            },
            "best_sample": {
                "metrics": {"structure_confidence": 0.8},
                "structure": {"url": "https://example.invalid/complex.cif"},
            },
        },
    }
    session = RecordingSession([FakeResponse(200, start_payload), FakeResponse(200, retrieve_payload)])
    provider = BoltzReceptorBindingProvider(BoltzConfig("sk_bc_ws_test_secret"), session=session)

    started = provider.start("ACDEFG", "CCO")
    completed = provider.retrieve("pred_123")

    assert started["provider_resource_id"] == "pred_123"
    assert started["status"] == "pending"
    assert completed["status"] == "succeeded"
    assert completed["binding_metrics"]["binding_confidence"] == pytest.approx(0.7)
    assert len(session.calls) == 2


def test_invalid_input_or_resource_id_never_calls_external_service():
    session = RecordingSession([])
    provider = BoltzReceptorBindingProvider(BoltzConfig("sk_bc_ws_test_secret"), session=session)

    with pytest.raises(ReceptorBindingError):
        provider.estimate_cost("ACDX", "CCO")
    with pytest.raises(ReceptorBindingError):
        provider.retrieve("../not-a-resource")

    assert session.calls == []
