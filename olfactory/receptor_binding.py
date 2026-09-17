"""Research-only receptor-ligand binding evidence providers.

This module deliberately sits outside the production odor-prediction contract.
A Boltz binding score is computational evidence about a proposed molecular
complex; it is not receptor activation, sensory validation, or an odor score.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Mapping, Optional

import requests
from rdkit import Chem, rdBase


BOLTZ_MODEL = "boltz-2.1"
DEFAULT_BOLTZ_BASE_URL = "https://api.boltz.bio"
DEFAULT_TIMEOUT_SECONDS = 30.0
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
_ALLOWED_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_RESOURCE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,255}$")

RESEARCH_LIMITATIONS = (
    "Predicted binding is not evidence of receptor activation or agonism.",
    "Predicted binding is not experimental sensory evidence or an odor prediction.",
    "This evidence is research-only and must not affect candidate ranking without a separate validation gate.",
)


class ReceptorBindingError(RuntimeError):
    """Stable error raised by receptor-binding providers."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class BoltzConfig:
    api_key: str
    base_url: str = DEFAULT_BOLTZ_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def mode(self) -> str:
        if self.api_key.startswith(("sk_bc_ws_test_", "sk_bc_admin_test_")):
            return "test"
        if self.api_key.startswith(("sk_bc_ws_live_", "sk_bc_admin_live_")):
            return "live"
        return "unknown"

    @property
    def key_scope(self) -> str:
        if self.api_key.startswith(("sk_bc_ws_test_", "sk_bc_ws_live_")):
            return "workspace"
        if self.api_key.startswith(("sk_bc_admin_test_", "sk_bc_admin_live_")):
            return "admin"
        return "unknown"

    @classmethod
    def from_env(cls) -> Optional["BoltzConfig"]:
        api_key = os.environ.get("BOLTZ_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.environ.get("BOLTZ_BASE_URL", DEFAULT_BOLTZ_BASE_URL).strip()
        raw_timeout = os.environ.get(
            "BOLTZ_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_TIMEOUT_SECONDS),
        )
        try:
            timeout_seconds = float(raw_timeout)
        except ValueError as error:
            raise ReceptorBindingError(
                "BOLTZ_CONFIG_INVALID",
                "BOLTZ_REQUEST_TIMEOUT_SECONDS must be numeric.",
            ) from error
        if timeout_seconds <= 0:
            raise ReceptorBindingError(
                "BOLTZ_CONFIG_INVALID",
                "BOLTZ_REQUEST_TIMEOUT_SECONDS must be greater than zero.",
            )
        if not base_url.startswith("https://"):
            raise ReceptorBindingError(
                "BOLTZ_CONFIG_INVALID",
                "BOLTZ_BASE_URL must use HTTPS.",
            )
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout_seconds=timeout_seconds,
        )


def normalize_receptor_sequence(value: str) -> str:
    """Normalize an amino-acid sequence without guessing ambiguous residues."""
    sequence = "".join(value.split()).upper()
    if not sequence:
        raise ReceptorBindingError(
            "INVALID_RECEPTOR_SEQUENCE",
            "Receptor sequence is required.",
        )
    invalid = sorted(set(sequence) - _ALLOWED_AMINO_ACIDS)
    if invalid:
        raise ReceptorBindingError(
            "INVALID_RECEPTOR_SEQUENCE",
            "Receptor sequence contains unsupported one-letter residue codes: "
            + ", ".join(invalid),
        )
    return sequence


def normalize_isomeric_smiles(value: str) -> str:
    """Parse and canonicalize a ligand while preserving specified stereochemistry."""
    raw = value.strip()
    if not raw:
        raise ReceptorBindingError("INVALID_SMILES", "Ligand SMILES is required.")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(raw, sanitize=True)
    if molecule is None:
        raise ReceptorBindingError(
            "INVALID_SMILES",
            "Ligand SMILES could not be parsed by RDKit.",
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_boltz_request(
    receptor_sequence: str,
    isomeric_smiles: str,
    *,
    model: str = BOLTZ_MODEL,
) -> Dict[str, object]:
    """Build the minimal Boltz-2.1 ligand-protein binding request."""
    sequence = normalize_receptor_sequence(receptor_sequence)
    smiles = normalize_isomeric_smiles(isomeric_smiles)
    request: Dict[str, object] = {
        "input": {
            "entities": [
                {
                    "chain_ids": ["A"],
                    "type": "protein",
                    "value": sequence,
                },
                {
                    "chain_ids": ["B"],
                    "type": "ligand_smiles",
                    "value": smiles,
                },
            ],
            "binding": {
                "type": "ligand_protein_binding",
                "binder_chain_id": "B",
            },
        },
        "model": model,
    }
    request["idempotency_key"] = deterministic_idempotency_key(request)
    return request


def deterministic_idempotency_key(request: Mapping[str, object]) -> str:
    """Prevent accidental duplicate compute for an identical normalized request."""
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"nox-or-binding-v1-{digest[:48]}"


def _safe_number(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalize_boltz_result(payload: Mapping[str, Any]) -> Dict[str, object]:
    """Map vendor output into an additive, vendor-labelled research evidence record."""
    output = _mapping(payload.get("output"))
    best_sample = _mapping(output.get("best_sample"))
    metrics = _mapping(best_sample.get("metrics"))
    binding = _mapping(output.get("binding_metrics"))
    structure = _mapping(best_sample.get("structure"))
    ligand_structure = _mapping(best_sample.get("ligand_structure"))
    input_payload = _mapping(payload.get("input"))
    entities = input_payload.get("entities")

    receptor_sequence = ""
    ligand_smiles = ""
    if isinstance(entities, list):
        for entity in entities:
            item = _mapping(entity)
            if item.get("type") == "protein" and not receptor_sequence:
                receptor_sequence = str(item.get("value") or "")
            if item.get("type") == "ligand_smiles" and not ligand_smiles:
                ligand_smiles = str(item.get("value") or "")

    error = _mapping(payload.get("error"))
    return {
        "schema_version": 1,
        "evidence_type": "PREDICTED_RECEPTOR_BINDING",
        "provider": "BOLTZ",
        "provider_resource_id": payload.get("id"),
        "status": payload.get("status"),
        "livemode": bool(payload.get("livemode", False)),
        "model": payload.get("model") or BOLTZ_MODEL,
        "model_version": payload.get("version"),
        "idempotency_key": payload.get("idempotency_key"),
        "created_at": payload.get("created_at"),
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "expires_at": payload.get("expires_at"),
        "input_fingerprints": {
            "receptor_sequence_sha256": _sha256_text(receptor_sequence) if receptor_sequence else None,
            "ligand_smiles_sha256": _sha256_text(ligand_smiles) if ligand_smiles else None,
        },
        "binding_metrics": {
            "binding_confidence": _safe_number(binding.get("binding_confidence")),
            "optimization_score": _safe_number(binding.get("optimization_score")),
            "type": binding.get("type"),
        },
        "structure_metrics": {
            key: _safe_number(metrics.get(key))
            for key in (
                "structure_confidence",
                "ptm",
                "iptm",
                "ligand_iptm",
                "protein_iptm",
                "complex_plddt",
                "complex_iplddt",
                "complex_pde",
                "complex_ipde",
            )
        },
        "artifacts": {
            "complex_structure_url": structure.get("url"),
            "complex_structure_url_expires_at": structure.get("url_expires_at"),
            "ligand_structure_url": ligand_structure.get("url"),
            "ligand_structure_url_expires_at": ligand_structure.get("url_expires_at"),
        },
        "error": (
            {
                "code": error.get("code"),
                "message": error.get("message"),
            }
            if error
            else None
        ),
        "limitations": list(RESEARCH_LIMITATIONS),
    }


class BoltzReceptorBindingProvider:
    """Thin synchronous client for research benchmarking against Boltz-2.1."""

    endpoint = "/compute/v1/predictions/structure-and-binding"

    def __init__(self, config: BoltzConfig, *, session: Optional[Any] = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    @property
    def metadata(self) -> Dict[str, object]:
        return {
            "provider": "BOLTZ",
            "model": BOLTZ_MODEL,
            "mode": self.config.mode,
            "key_scope": self.config.key_scope,
            "external": True,
            "research_only": True,
            "production_ranking": False,
        }

    def estimate_cost(self, receptor_sequence: str, isomeric_smiles: str) -> Dict[str, object]:
        request = build_boltz_request(receptor_sequence, isomeric_smiles)
        payload = self._request(
            "POST",
            f"{self.endpoint}/estimate-cost",
            json_payload=request,
        )
        return {
            "estimated_cost_usd": str(payload.get("estimated_cost_usd", "")),
            "breakdown": payload.get("breakdown"),
            "disclaimer": payload.get("disclaimer"),
            "idempotency_key": request["idempotency_key"],
            "mode": self.config.mode,
        }

    def start(self, receptor_sequence: str, isomeric_smiles: str) -> Dict[str, object]:
        request = build_boltz_request(receptor_sequence, isomeric_smiles)
        payload = self._request("POST", self.endpoint, json_payload=request)
        return normalize_boltz_result(payload)

    def retrieve(self, resource_id: str) -> Dict[str, object]:
        if not _RESOURCE_ID.fullmatch(resource_id):
            raise ReceptorBindingError(
                "INVALID_PROVIDER_RESOURCE_ID",
                "Boltz resource ID has an invalid format.",
            )
        payload = self._request("GET", f"{self.endpoint}/{resource_id}")
        return normalize_boltz_result(payload)

    def wait(
        self,
        resource_id: str,
        *,
        poll_seconds: float = 5.0,
        max_wait_seconds: float = 900.0,
    ) -> Dict[str, object]:
        if poll_seconds <= 0 or max_wait_seconds <= 0:
            raise ValueError("poll_seconds and max_wait_seconds must be greater than zero")
        deadline = time.monotonic() + max_wait_seconds
        while True:
            result = self.retrieve(resource_id)
            status = str(result.get("status") or "").lower()
            if status in TERMINAL_STATUSES:
                return result
            if time.monotonic() >= deadline:
                raise ReceptorBindingError(
                    "BOLTZ_WAIT_TIMEOUT",
                    "Boltz prediction did not reach a terminal state before the local wait limit.",
                )
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Mapping[str, object]] = None,
    ) -> Mapping[str, Any]:
        url = f"{self.config.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                headers={
                    "x-api-key": self.config.api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=dict(json_payload) if json_payload is not None else None,
                timeout=self.config.timeout_seconds,
            )
        except requests.Timeout as error:
            raise ReceptorBindingError(
                "BOLTZ_TIMEOUT",
                "Boltz API request timed out.",
            ) from error
        except requests.RequestException as error:
            raise ReceptorBindingError(
                "BOLTZ_NETWORK_ERROR",
                "Boltz API request failed before a response was received.",
            ) from error

        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = {}
            detail = _mapping(body.get("error") if isinstance(body, Mapping) else None)
            if not detail and isinstance(body, Mapping):
                detail = _mapping(body.get("detail"))
            message = str(detail.get("message") or f"Boltz API returned HTTP {response.status_code}.")
            code = str(detail.get("code") or f"BOLTZ_HTTP_{response.status_code}")
            raise ReceptorBindingError(code, message, status_code=response.status_code)

        try:
            payload = response.json()
        except ValueError as error:
            raise ReceptorBindingError(
                "BOLTZ_INVALID_RESPONSE",
                "Boltz API returned a non-JSON response.",
                status_code=response.status_code,
            ) from error
        if not isinstance(payload, Mapping):
            raise ReceptorBindingError(
                "BOLTZ_INVALID_RESPONSE",
                "Boltz API returned an unexpected response shape.",
                status_code=response.status_code,
            )
        return payload


def build_boltz_provider_from_env(*, session: Optional[Any] = None) -> Optional[BoltzReceptorBindingProvider]:
    config = BoltzConfig.from_env()
    if config is None:
        return None
    return BoltzReceptorBindingProvider(config, session=session)
