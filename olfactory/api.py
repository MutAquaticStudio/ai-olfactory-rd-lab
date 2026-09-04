"""FastAPI boundary for the Scent Molecule Studio web interface."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Set, Tuple

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rdkit import Chem, rdBase
import torch

from .academic_evidence import (
    ACADEMIC_EVIDENCE_SCHEMA_VERSION,
    AcademicEvidenceService,
    AcademicEvidenceStore,
)
from .chemistry import (
    CONFORMER_CACHE_SIZE,
    MACROCYCLE_CLUSTER_RMSD,
    MACROCYCLE_CONFORMER_COUNT,
    MAX_ENSEMBLE_SIZE,
    NORMAL_CLUSTER_RMSD,
    NORMAL_CONFORMER_COUNT,
    ChemicalScreenResult,
    ConformerEnsembleResult,
    build_conformer_ensemble,
    screen_molecule,
)
from .copy import COPY, REASON_COPY
from .depictions import display_descriptors, molecule_svg
from .data_foundation import DataFoundationService
from .features import (
    create_morgan_tensor,
    geometric_mean,
    predict_probabilities,  # kept as a compatibility seam for v0.5 tests/extensions
    smiles_representations,
    top_descriptors,
)
from .prediction_integrity import (
    PredictionIdentity,
    legacy_prediction_payload,
    nearest_training_similarity,
)
from .prediction import LegacyMorganPredictor, MoleculePredictor
from .generation import (
    GenerationEvent,
    GenerationResult,
    RankedCandidate,
    ReviewCandidate,
    generate_candidate_pool,
    rank_candidates,
    sample_smiles_string,
)
from .models import OdorPredictor, SMILES_LSTM
from .pubchem import PubChemClient
from .references import (
    PUBCHEM,
    ReferenceEvidence,
    ReferenceGate,
    ReferenceGateStatus,
    ReferenceVerifier,
    build_reference_verifier,
)
from .resources import (
    ResourceBundleError,
    default_resource_dir,
    load_existing_isomeric_smiles_set,
    load_odor_model,
    load_smiles_model,
    load_training_fingerprints,
    validate_resource_bundle,
)
from .taxonomy import TaxonomyProfile, load_mapping, project_probabilities
from .stereo import enumerate_stereo_options
from .training.registry import ModelRegistry


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRONTEND_DIST = APP_ROOT / "frontend" / "dist"
RESOURCE_DIR = default_resource_dir()
DATASET_PATH = RESOURCE_DIR / "odor_morgan_tensor_dataset.pt"
ODOR_WEIGHTS_PATH = RESOURCE_DIR / "odor_predictor_weights.pth"
CREATOR_WEIGHTS_PATH = RESOURCE_DIR / "smiles_creator_weights.pth"
VOCAB_PATH = APP_ROOT / "smiles_vocab.json"
REFERENCE_PATH = APP_ROOT / "clean_dataset.csv"
MODEL_REGISTRY_PATH = APP_ROOT / "model_registry.json"
ACADEMIC_EVIDENCE_PATH = Path(
    os.environ.get(
        "SCENT_STUDIO_ACADEMIC_EVIDENCE_PATH",
        str(APP_ROOT / "faiss_academic_index" / "academic_evidence.jsonl"),
    )
).expanduser()

REQUIRED_CANDIDATES = 5
SHORTLIST_COUNT = 3
MAX_ATTEMPTS = 200
MAX_SECONDS = 120.0
MAX_EVENT_LINES = 30
ANALYSIS_STEREO_LIMIT = 16
CANDIDATE_STEREO_LIMIT = 4


class AnalysisRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=4096)


class CandidateRequest(BaseModel):
    target_descriptors: List[str] = Field(min_length=1)
    sampling_diversity: float = Field(ge=0.2, le=1.2)
    pubchem_consent: bool = False
    reference_consents: List[str] = Field(default_factory=list)


class ImportCommitRequest(BaseModel):
    validation_token: str = Field(min_length=1, max_length=128)


class StudyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    protocol_version: str = Field(default="sensory-v1", min_length=1, max_length=80)
    stage: str = Field(default="PANEL", pattern="^(EXPERT|PANEL)$")


class AssessmentRequest(BaseModel):
    study_name: str = Field(min_length=1, max_length=200)
    session_name: str = Field(min_length=1, max_length=200)
    assessor_id: str = Field(min_length=1, max_length=120)
    blinded_sample_code: str = Field(min_length=1, max_length=120)
    smiles: str = Field(min_length=1, max_length=4096)
    descriptor: str = Field(min_length=1, max_length=120)
    presence_state: str = Field(pattern="^(PRESENT|ABSENT|UNASSESSED)$")
    concentration: float
    concentration_unit: str = Field(min_length=1, max_length=24)
    solvent: str = Field(min_length=1, max_length=120)
    temperature_c: float
    confidence: float
    replicate_number: int
    intensity: Optional[float] = None
    source_name: str = Field(default="private_panel", min_length=1, max_length=120)
    source_version: str = Field(default="1", min_length=1, max_length=120)
    source_license: str = Field(default="PRIVATE", min_length=1, max_length=120)
    source_record_id: Optional[str] = Field(default=None, max_length=200)
    preparation_time_minutes: Optional[float] = None
    notes: Optional[str] = Field(default=None, max_length=2000)
    supersedes_assessment_id: Optional[str] = Field(default=None, max_length=120)


class AcademicEvidenceQuery(BaseModel):
    isomeric_smiles: str = Field(min_length=1, max_length=4096)
    include_abstracts: bool = False


@dataclass
class AppResources:
    odor_model: OdorPredictor
    label_names: Tuple[str, ...]
    creator_model: SMILES_LSTM
    char_to_idx: Dict[str, int]
    idx_to_char: Tuple[str, ...]
    existing_isomeric_smiles_set: Set[str]
    pubchem_client: PubChemClient
    reference_verifier: Optional[ReferenceVerifier] = None
    training_fingerprints: Optional[torch.Tensor] = None
    judge_identity: PredictionIdentity = field(
        default_factory=lambda: PredictionIdentity(
            model_version="judge-v1-legacy",
            dataset_version="legacy-clean-3522",
            calibration_version="uncalibrated",
            model_status="LEGACY_BASELINE",
        )
    )
    creator_registry_entry: Dict[str, object] = field(default_factory=dict)
    odor_lock: threading.Lock = field(default_factory=threading.Lock)
    creator_lock: threading.Lock = field(default_factory=threading.Lock)
    data_service: Optional[DataFoundationService] = None
    predictor: Optional[MoleculePredictor] = None
    academic_evidence_service: Optional[AcademicEvidenceService] = None

    def __post_init__(self) -> None:
        if self.reference_verifier is None:
            self.reference_verifier = build_reference_verifier(self.pubchem_client)
        if self.predictor is None:
            self.predictor = LegacyMorganPredictor(
                self.odor_model,
                self.label_names,
                identity=self.judge_identity,
                training_fingerprints=self.training_fingerprints,
            )
        if self.academic_evidence_service is None:
            # Ingest writes derived evidence next to the local FAISS batches.
            # Keep the API pointed at that same path while allowing an explicit
            # environment override for a reviewed evidence store.
            self.academic_evidence_service = AcademicEvidenceService(
                AcademicEvidenceStore(ACADEMIC_EVIDENCE_PATH)
            )


def load_app_resources() -> AppResources:
    resource_dir = validate_resource_bundle()
    registry = ModelRegistry(MODEL_REGISTRY_PATH)
    judge_entry = registry.production("judge") or {}
    creator_entry = registry.production("creator") or {}
    if judge_entry and not registry.verify_entry(
        judge_entry,
        resource_dir,
        require_within_root=True,
    ):
        raise ResourceBundleError(
            "MODEL_CHECKSUM_MISMATCH",
            "Judge resource checksum verification failed.",
        )
    if creator_entry and not registry.verify_entry(
        creator_entry,
        resource_dir,
        require_within_root=True,
    ):
        raise ResourceBundleError(
            "MODEL_CHECKSUM_MISMATCH",
            "Creator resource checksum verification failed.",
        )
    dataset_path = resource_dir / DATASET_PATH.name
    judge_weights = resource_dir / str(judge_entry.get("weights_path", ODOR_WEIGHTS_PATH.name))
    creator_weights = resource_dir / str(creator_entry.get("weights_path", CREATOR_WEIGHTS_PATH.name))
    odor_model, label_names = load_odor_model(dataset_path, judge_weights)
    creator_model, char_to_idx, idx_to_char = load_smiles_model(
        VOCAB_PATH,
        creator_weights,
    )
    pubchem_client = PubChemClient()
    tgsc_manifest = os.environ.get("TGSC_REFERENCE_MANIFEST")
    scentree_manifest = os.environ.get("SCENTREE_REFERENCE_MANIFEST")
    resources = AppResources(
        odor_model=odor_model,
        label_names=label_names,
        creator_model=creator_model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        existing_isomeric_smiles_set=load_existing_isomeric_smiles_set(REFERENCE_PATH),
        pubchem_client=pubchem_client,
        reference_verifier=build_reference_verifier(
            pubchem_client,
            tgsc_manifest=Path(tgsc_manifest).expanduser() if tgsc_manifest else None,
            scentree_manifest=(
                Path(scentree_manifest).expanduser() if scentree_manifest else None
            ),
        ),
        training_fingerprints=load_training_fingerprints(dataset_path),
        judge_identity=PredictionIdentity(
            model_version=str(judge_entry.get("model_version", "judge-v1-legacy")),
            dataset_version=str(judge_entry.get("dataset_version", "legacy-clean-3522")),
            calibration_version=str(judge_entry.get("calibration_version", "uncalibrated")),
            model_status=str(judge_entry.get("status", "LEGACY_BASELINE")),
        ),
        creator_registry_entry=creator_entry,
    )
    resources.data_service = DataFoundationService(label_names=label_names)
    return resources


def _screen_payload(screen: ChemicalScreenResult) -> Dict[str, object]:
    return {
        "decision": screen.decision.value,
        "reason_codes": list(screen.reason_codes),
        "reasons": [REASON_COPY.get(code, code) for code in screen.reason_codes],
        "descriptors": dict(screen.descriptors),
        "is_macrocycle": screen.is_macrocycle,
        "macrocycle_ring_size": screen.macrocycle_ring_size,
        "macrocycle_carbon_fraction": screen.macrocycle_carbon_fraction,
        "macrocycle_heteroatoms": screen.macrocycle_heteroatoms,
        "alerts": list(screen.alerts),
    }


def _ensemble_payload(result: ConformerEnsembleResult) -> Dict[str, object]:
    return {
        "available": result.available,
        "method": result.method,
        "requested_count": result.requested_count,
        "embedded_count": result.embedded_count,
        "converged_count": result.converged_count,
        "is_macrocycle": result.is_macrocycle,
        "error": result.error,
        "conformers": [
            {
                "molblock": conformer.molblock,
                "relative_energy": conformer.relative_energy,
            }
            for conformer in result.conformers
        ],
    }


def _taxonomy_payload(profile: TaxonomyProfile) -> Dict[str, object]:
    return {
        "facets": [
            {"name": name, "probability": probability}
            for name, probability in profile.facets
        ],
        "textures": [
            {"name": name, "probability": probability}
            for name, probability in profile.textures
        ],
        "sensations": [
            {"name": name, "probability": probability}
            for name, probability in profile.sensations
        ],
        "projection_name": profile.projection_name,
        "taxonomy_version": profile.taxonomy_version,
    }


def _reference_evidence_payload(item: ReferenceEvidence) -> Dict[str, object]:
    return {
        "provider": item.provider,
        "status": item.status.value,
        "match_level": item.match_level.value if item.match_level else None,
        "queried_identifier": item.queried_identifier,
        "record_ids": list(item.record_ids),
        "record_urls": list(item.record_urls),
        "checked_at": item.checked_at,
        "source_version": item.source_version,
        "error_code": item.error_code,
    }


def _reference_gate_payload(gate: Optional[ReferenceGate]) -> Dict[str, object]:
    if gate is None:
        return {
            "status": ReferenceGateStatus.NOT_RUN.value,
            "blocking_providers": [],
            "reason_code": "REFERENCE_CHECK_NOT_RUN",
        }
    return {
        "status": gate.status.value,
        "blocking_providers": list(gate.blocking_providers),
        "reason_code": gate.reason_code,
    }


def _academic_evidence_payload(summary: object) -> Dict[str, object]:
    """Serialize local evidence without exposing implementation details."""
    to_dict = getattr(summary, "to_dict", None)
    if to_dict is None:
        return {
            "status": "NO_EXACT_EVIDENCE",
            "query_isomeric_smiles": None,
            "normalized_structure": None,
            "matches": [],
            "conflicts": [],
            "provenance": [],
        }
    return to_dict()


def analyze_smiles(resources: AppResources, raw_smiles: str) -> Dict[str, object]:
    normalized = raw_smiles.strip()
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(normalized, sanitize=True)
    if molecule is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_SMILES",
                "message": COPY["invalid_smiles"],
            },
        )

    isomeric_smiles, canonical_smiles = smiles_representations(molecule)
    screen = screen_molecule(molecule)
    response: Dict[str, object] = {
        "input_smiles": normalized,
        "identifiers": {
            "isomeric_smiles": isomeric_smiles,
            "canonical_smiles": canonical_smiles,
        },
        "structure_2d_svg": molecule_svg(molecule),
        "chemistry_screen": _screen_payload(screen),
        "display_descriptors": display_descriptors(molecule, screen.descriptors),
        "predicted_odor_profile": None,
        "prediction_v2": None,
        "conformer_ensemble": None,
        "stereo_options": [],
        # Analysis never transmits identifiers without a dedicated consent flow.
        "reference_checks": [],
        "reference_gate": _reference_gate_payload(None),
        "academic_evidence": None,
    }

    resolution = enumerate_stereo_options(molecule, limit=ANALYSIS_STEREO_LIMIT)
    if resolution.exceeds_limit:
        response.update(
            {
                "analysis_state": "STEREO_INPUT_REQUIRED",
                "unresolved_stereo_elements": resolution.unresolved_elements,
            }
        )
        return response
    if resolution.options:
        response.update(
            {
                "analysis_state": "STEREO_REQUIRED",
                "unresolved_stereo_elements": resolution.unresolved_elements,
                "stereo_options": [
                    {
                        "isomeric_smiles": option.isomeric_smiles,
                        "cip_summary": option.cip_summary,
                        "structure_2d_svg": option.structure_2d_svg,
                    }
                    for option in resolution.options
                ],
            }
        )
        return response

    with resources.odor_lock:
        if resources.predictor is None:
            raise RuntimeError("Molecule predictor is unavailable")
        prediction_batch = resources.predictor.predict([isomeric_smiles])
        probabilities = torch.from_numpy(prediction_batch.presence_probability[0])
    probability_values = [float(value) for value in probabilities.tolist()]
    similarity = nearest_training_similarity(
        create_morgan_tensor(molecule),
        resources.training_fingerprints,
    )
    profile = project_probabilities(probability_values, resources.label_names)
    ensemble = build_conformer_ensemble(isomeric_smiles)
    prediction_identity = resources.judge_identity
    prediction_similarity = similarity
    if prediction_batch is not None:
        prediction_identity = PredictionIdentity(
            model_version=prediction_batch.model_version,
            dataset_version=prediction_batch.dataset_version,
            calibration_version=prediction_batch.calibration_version,
            model_status=resources.judge_identity.model_status,
        )
        if prediction_batch.training_similarity.size and np.isfinite(
            prediction_batch.training_similarity[0]
        ):
            prediction_similarity = float(prediction_batch.training_similarity[0])
    prediction_payload = legacy_prediction_payload(
        probability_values,
        resources.label_names,
        prediction_identity,
        prediction_similarity,
    )
    # Keep the v1 list unchanged while exposing the v2 contract additively.
    prediction_payload.update(
        {
            "presence_probability": probability_values,
            "expected_intensity": [
                None if not np.isfinite(value) else float(value)
                for value in prediction_batch.expected_intensity[0]
            ],
            "ensemble_uncertainty": [
                None if not np.isfinite(value) else float(value)
                for value in prediction_batch.ensemble_uncertainty[0]
            ],
            "training_similarity": prediction_payload["nearest_training_similarity"],
            "reliability": prediction_payload["reliability_state"],
        }
    )
    response.update(
        {
            "analysis_state": "COMPLETE",
            "unresolved_stereo_elements": 0,
            "conformer_ensemble": _ensemble_payload(ensemble),
            "predicted_odor_profile": {
                "top_descriptors": [
                    {"name": name, "probability": probability}
                    for name, probability in top_descriptors(
                        probabilities,
                        resources.label_names,
                        count=5,
                    )
                ],
                "taxonomy": _taxonomy_payload(profile),
                "model_output": [
                    {"name": label, "probability": probability}
                    for label, probability in zip(
                        resources.label_names,
                        probability_values,
                    )
                ],
            },
            "prediction_v2": prediction_payload,
        }
    )
    if resources.academic_evidence_service is not None:
        academic_summary = resources.academic_evidence_service.verify(
            isomeric_smiles,
            include_abstracts=False,
        )
        response["academic_evidence"] = _academic_evidence_payload(academic_summary)
    return response


def _event_payload(event: GenerationEvent) -> Dict[str, object]:
    return {
        "phase": event.phase.value,
        "attempt": event.attempt,
        "accepted": event.accepted,
        "invalid": event.invalid,
        "duplicates": event.duplicates,
        "rejected": event.rejected,
        "reviews": event.reviews,
        "found": event.found,
        "unverified": event.unverified,
        "reference_matches": event.reference_matches,
        "reference_unverified": event.reference_unverified,
        "detail": event.detail,
    }


def _review_payload(item: ReviewCandidate) -> Dict[str, object]:
    molecule = Chem.MolFromSmiles(item.isomeric_smiles)
    return {
        "isomeric_smiles": item.isomeric_smiles,
        "structure_2d_svg": molecule_svg(molecule) if molecule is not None else None,
        "chemistry_screen": _screen_payload(item.chemical_screen),
        "review_category": item.review_category,
        "reference_checks": [
            _reference_evidence_payload(evidence)
            for evidence in item.reference_checks
        ],
        "reference_gate": _reference_gate_payload(item.reference_gate),
    }


def _candidate_payload(candidate: RankedCandidate) -> Dict[str, object]:
    molecule = Chem.MolFromSmiles(candidate.isomeric_smiles)
    if molecule is None:
        raise ValueError("Ranked candidate could not be parsed")
    ensemble = candidate.conformer_ensemble
    return {
        "isomeric_smiles": candidate.isomeric_smiles,
        "canonical_smiles": candidate.canonical_smiles,
        "target_fit": candidate.target_fit,
        "target_probabilities": [
            {"name": name, "probability": probability}
            for name, probability in candidate.target_probabilities
        ],
        "supporting_descriptors": [
            {"name": name, "probability": probability}
            for name, probability in candidate.supporting_descriptors
        ],
        "structure_2d_svg": molecule_svg(molecule),
        "conformer_ensemble": _ensemble_payload(ensemble),
        "chemistry_screen": _screen_payload(candidate.chemical_screen),
        "display_descriptors": display_descriptors(
            molecule,
            candidate.chemical_screen.descriptors,
        ),
        "novelty": {
            "status": candidate.novelty.status.value,
            "cids": list(candidate.novelty.cids),
            "error_code": candidate.novelty.error_code,
        },
        "reference_checks": [
            _reference_evidence_payload(evidence)
            for evidence in candidate.reference_checks
        ],
        "reference_gate": _reference_gate_payload(candidate.reference_gate),
    }


def _completion_payload(
    resources: AppResources,
    result: GenerationResult,
    targets: Sequence[str],
) -> Dict[str, object]:
    with resources.odor_lock:
        ranked = rank_candidates(
            resources.predictor,
            resources.label_names,
            targets,
            result.accepted_candidates,
        )
    return {
        "shortlist": [
            _candidate_payload(candidate) for candidate in ranked[:SHORTLIST_COUNT]
        ],
        "review_queue": [_review_payload(item) for item in result.review_queue],
        "summary": {
            "attempts": result.attempts,
            "accepted": len(result.accepted_candidates),
            "reviews": len(result.review_queue),
            "invalid": result.invalid,
            "duplicates": result.duplicates,
            "rejected": result.rejected,
            "found": result.found,
            "unverified": result.unverified,
            "reference_matches": result.reference_matches,
            "reference_unverified": result.reference_unverified,
            "elapsed_seconds": result.elapsed_seconds,
            "reached_attempt_limit": result.reached_attempt_limit,
            "reached_time_limit": result.reached_time_limit,
        },
    }


def _advance_generator(
    stream: Generator[GenerationEvent, None, GenerationResult],
) -> Tuple[bool, Optional[GenerationEvent], Optional[GenerationResult]]:
    try:
        return True, next(stream), None
    except StopIteration as stopped:
        return False, None, stopped.value


def _sse(event_name: str, payload: Dict[str, object]) -> str:
    return "event: {0}\ndata: {1}\n\n".format(
        event_name,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def _resources(request: Request) -> AppResources:
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "RESOURCES_UNAVAILABLE",
                "message": COPY["resource_error"],
            },
        )
    return resources


def _data_service(request: Request) -> DataFoundationService:
    resources = _resources(request)
    if resources.data_service is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DATA_FOUNDATION_UNAVAILABLE",
                "message": "The local scientific data store is unavailable.",
            },
        )
    return resources.data_service


def _model_payload(model: BaseModel) -> Dict[str, object]:
    dump = getattr(model, "model_dump", None)
    return dump() if dump is not None else model.dict()


def create_app(
    resources: Optional[AppResources] = None,
    frontend_dist: Optional[Path] = None,
) -> FastAPI:
    dist_path = (frontend_dist or DEFAULT_FRONTEND_DIST).resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if resources is not None:
            application.state.resources = resources
            application.state.resource_error = None
        else:
            try:
                application.state.resources = await asyncio.to_thread(load_app_resources)
                application.state.resource_error = None
            except Exception as error:
                application.state.resources = None
                application.state.resource_error = getattr(error, "code", type(error).__name__)
        yield

    application = FastAPI(
        title=COPY["app_title"],
        version="0.5.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.exception_handler(Exception)
    async def unhandled_error(_: Request, error: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "INTERNAL_ERROR",
                    "message": "The request could not be completed.",
                    "technical_details": type(error).__name__,
                }
            },
        )

    @application.get("/api/v1/health")
    async def health(request: Request):
        ready = getattr(request.app.state, "resources", None) is not None
        return {
            "status": "ready" if ready else "degraded",
            "ready": ready,
            "resource_error": getattr(request.app.state, "resource_error", None),
        }

    @application.get("/api/v1/meta")
    async def meta(request: Request):
        current = _resources(request)
        mapping = load_mapping(current.label_names)
        verifier = current.reference_verifier
        if verifier is None:
            raise RuntimeError("Reference verifier is unavailable")
        return {
            "label_names": list(current.label_names),
            "taxonomy_version": mapping["taxonomy_version"],
            "projection_name": mapping["projection_name"],
            "generation_limits": {
                "required_candidates": REQUIRED_CANDIDATES,
                "shortlist_count": SHORTLIST_COUNT,
                "max_attempts": MAX_ATTEMPTS,
                "max_seconds": MAX_SECONDS,
                "max_event_lines": MAX_EVENT_LINES,
                "candidate_stereo_limit": CANDIDATE_STEREO_LIMIT,
            },
            "conformer_ensemble": {
                "normal_sampling_count": NORMAL_CONFORMER_COUNT,
                "macrocycle_sampling_count": MACROCYCLE_CONFORMER_COUNT,
                "max_displayed": MAX_ENSEMBLE_SIZE,
                "normal_cluster_rmsd": NORMAL_CLUSTER_RMSD,
                "macrocycle_cluster_rmsd": MACROCYCLE_CLUSTER_RMSD,
                "cache_size": CONFORMER_CACHE_SIZE,
            },
            "stereo": {
                "analysis_option_limit": ANALYSIS_STEREO_LIMIT,
                "candidate_variant_limit": CANDIDATE_STEREO_LIMIT,
            },
            "capabilities": {"structure_2d": True, "conformer_3d": True},
            "data_foundation": {
                "available": current.data_service is not None,
                "label_semantics": ["PRESENT", "ABSENT", "UNASSESSED"],
                "intensity_scale": [0, 10],
            },
            "academic_evidence": {
                "available": current.academic_evidence_service is not None,
                "schema_version": ACADEMIC_EVIDENCE_SCHEMA_VERSION,
                "default_content_type": "full_text",
                "abstracts_opt_in": True,
                "training_auto_import": False,
                "identity_policy": "EXACT_STEREO_WITH_PROVENANCE",
                "safety_policy": "COMPUTATIONAL_TRIAGE_ONLY",
            },
            "prediction_contract": {
                "name": "PredictionBatch",
                "version": "2",
                "fields": [
                    "model_version",
                    "dataset_version",
                    "calibration_version",
                    "presence_probability",
                    "expected_intensity",
                    "ensemble_uncertainty",
                    "training_similarity",
                    "reliability_state",
                ],
                "production_adapter": type(current.predictor).__name__ if current.predictor else None,
            },
            "training": {
                "primary_python": "3.11",
                "compatibility_python": ["3.10", "3.12"],
                "deepchem_training_only": True,
                "promotion_requires_quality_gate": True,
            },
            "reference_verification": {
                "providers": [
                    {
                        "provider": item.provider,
                        "display_name": item.display_name,
                        "source_type": item.source_type,
                        "enabled": item.enabled,
                        "external": item.external,
                        "query_types": list(item.query_types),
                        "dataset_version": item.dataset_version,
                        "license_status": item.license_status,
                        "configuration_error": item.configuration_error,
                    }
                    for item in verifier.metadata
                ],
                "required_external_consents": list(
                    verifier.required_external_providers
                ),
                "shortlist_policy": (
                    "CHEMISTRY_PASS_AND_ALL_ENABLED_REFERENCES_NO_MATCH"
                ),
            },
            "models": {
                "judge": {
                    "model_version": current.judge_identity.model_version,
                    "dataset_version": current.judge_identity.dataset_version,
                    "calibration_version": current.judge_identity.calibration_version,
                    "status": current.judge_identity.model_status,
                },
                "creator": {
                    "model_version": str(current.creator_registry_entry.get("model_version", "creator-v1-legacy")),
                    "dataset_version": str(current.creator_registry_entry.get("dataset_version", "legacy-clean-3522")),
                    "status": str(current.creator_registry_entry.get("status", "LEGACY_BASELINE")),
                },
            },
        }

    @application.get("/api/v1/data/templates")
    async def data_templates(request: Request, format: str = "json"):
        service = _data_service(request)
        if format.lower() == "csv":
            return Response(
                content=service.template_csv(),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="sensory_assessment_template.csv"'},
            )
        return service.template_payload()

    @application.post("/api/v1/data/imports/validate")
    async def validate_data_import(request: Request, file: UploadFile = File(...)):
        service = _data_service(request)
        try:
            raw_bytes = await file.read()
            validation = await asyncio.to_thread(
                service.validate_import,
                file.filename or "upload.csv",
                raw_bytes,
            )
            return validation.public_payload()
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": str(error),
                    "message": "The uploaded file could not be validated.",
                },
            ) from error

    @application.post("/api/v1/data/imports/commit")
    async def commit_data_import(payload: ImportCommitRequest, request: Request):
        service = _data_service(request)
        try:
            return await asyncio.to_thread(service.commit_import, payload.validation_token)
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "VALIDATION_TOKEN_NOT_FOUND",
                    "message": "The validated import is no longer available. Validate the file again.",
                },
            ) from error

    @application.post("/api/v1/studies")
    async def create_study(payload: StudyRequest, request: Request):
        service = _data_service(request)
        body = _model_payload(payload)
        return await asyncio.to_thread(
            service.repository.create_study,
            str(body["name"]),
            str(body["protocol_version"]),
            str(body["stage"]),
        )

    @application.post("/api/v1/assessments")
    async def create_assessment(payload: AssessmentRequest, request: Request):
        service = _data_service(request)
        body = _model_payload(payload)
        validation = await asyncio.to_thread(service.validate_assessment, body)
        if not validation.is_valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "ASSESSMENT_VALIDATION_FAILED",
                    "message": "The assessment contains validation errors.",
                    "validation": validation.public_payload(),
                },
            )
        return await asyncio.to_thread(service.commit_assessment, body)

    @application.post("/api/v1/assessments/validate")
    async def validate_assessment(payload: AssessmentRequest, request: Request):
        service = _data_service(request)
        validation = await asyncio.to_thread(
            service.validate_assessment,
            _model_payload(payload),
        )
        return validation.public_payload()

    @application.get("/api/v1/datasets/versions")
    async def dataset_versions(request: Request):
        return {"versions": await asyncio.to_thread(_data_service(request).list_snapshots)}

    @application.post("/api/v1/academic/evidence/query")
    async def academic_evidence_query(
        payload: AcademicEvidenceQuery,
        request: Request,
    ):
        current = _resources(request)
        service = current.academic_evidence_service
        if service is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "ACADEMIC_EVIDENCE_UNAVAILABLE",
                    "message": "Academic evidence is currently unavailable.",
                },
            )
        try:
            summary = await asyncio.to_thread(
                service.verify,
                payload.isomeric_smiles,
                include_abstracts=payload.include_abstracts,
            )
            return _academic_evidence_payload(summary)
        except ValueError as error:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_STRUCTURE_IDENTIFIER",
                    "message": "The structure could not be verified.",
                    "technical_details": str(error),
                },
            ) from error

    @application.get("/api/v1/academic/evidence/{evidence_id}")
    async def academic_evidence_detail(evidence_id: str, request: Request):
        current = _resources(request)
        service = current.academic_evidence_service
        if service is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "ACADEMIC_EVIDENCE_UNAVAILABLE",
                    "message": "Academic evidence is currently unavailable.",
                },
            )
        evidence = await asyncio.to_thread(service.get, evidence_id)
        if evidence is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "ACADEMIC_EVIDENCE_NOT_FOUND",
                    "message": "The academic evidence record was not found.",
                },
            )
        return evidence.to_dict()

    @application.get("/api/v1/academic/sources")
    async def academic_sources(request: Request):
        current = _resources(request)
        service = current.academic_evidence_service
        if service is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "ACADEMIC_EVIDENCE_UNAVAILABLE",
                    "message": "Academic evidence is currently unavailable.",
                },
            )
        return {"sources": await asyncio.to_thread(service.sources)}

    @application.post("/api/v1/analysis")
    async def analysis(payload: AnalysisRequest, request: Request):
        current = _resources(request)
        return await asyncio.to_thread(analyze_smiles, current, payload.smiles)

    @application.post("/api/v1/candidates/stream")
    async def candidate_stream(payload: CandidateRequest, request: Request):
        current = _resources(request)
        verifier = current.reference_verifier
        if verifier is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "REFERENCE_RESOURCES_UNAVAILABLE",
                    "message": COPY["resource_error"],
                },
            )
        effective_consents = {
            str(provider).upper() for provider in payload.reference_consents
        }
        # Deprecated compatibility field retained for the v0.5 client.
        if payload.pubchem_consent:
            effective_consents.add(PUBCHEM)
        missing_consents = sorted(
            set(verifier.required_external_providers) - effective_consents
        )
        if missing_consents:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": (
                        "PUBCHEM_CONSENT_REQUIRED"
                        if missing_consents == [PUBCHEM]
                        else "REFERENCE_CONSENT_REQUIRED"
                    ),
                    "message": COPY["reference_consent_required"].format(
                        providers=", ".join(missing_consents)
                    ),
                },
            )
        unknown = sorted(set(payload.target_descriptors) - set(current.label_names))
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "UNKNOWN_TARGET_DESCRIPTOR",
                    "message": "Unknown target descriptor: {0}".format(", ".join(unknown)),
                },
            )

        def locked_sampler(**kwargs):
            with current.creator_lock:
                return sample_smiles_string(**kwargs)

        target_indices = [
            current.label_names.index(label) for label in payload.target_descriptors
        ]

        def locked_variant_scorer(molecules):
            with current.odor_lock:
                if current.predictor is None:
                    raise RuntimeError("Molecule predictor is unavailable")
                variant_smiles = [
                    Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                    for molecule in molecules
                ]
                prediction = current.predictor.predict(variant_smiles)
                matrix = torch.from_numpy(prediction.presence_probability)
            return [
                geometric_mean(probabilities[target_indices])
                for probabilities in matrix
            ]

        stream = generate_candidate_pool(
            creator_model=current.creator_model,
            char_to_idx=current.char_to_idx,
            idx_to_char=current.idx_to_char,
            temperature=payload.sampling_diversity,
            existing_isomeric_smiles_set=current.existing_isomeric_smiles_set,
            reference_verifier=verifier,
            reference_consents=effective_consents,
            required_count=REQUIRED_CANDIDATES,
            max_attempts=MAX_ATTEMPTS,
            max_seconds=MAX_SECONDS,
            stereo_limit=CANDIDATE_STEREO_LIMIT,
            sampler=locked_sampler,
            variant_scorer=locked_variant_scorer,
        )

        async def event_source():
            try:
                while True:
                    if await request.is_disconnected():
                        stream.close()
                        return
                    active, event, result = await asyncio.to_thread(
                        _advance_generator,
                        stream,
                    )
                    if active and event is not None:
                        yield _sse("progress", _event_payload(event))
                        continue
                    if result is None:
                        raise RuntimeError("Generation stream ended without a result")
                    completion = await asyncio.to_thread(
                        _completion_payload,
                        current,
                        result,
                        payload.target_descriptors,
                    )
                    yield _sse("complete", completion)
                    return
            except asyncio.CancelledError:
                stream.close()
                raise
            except Exception as error:
                stream.close()
                yield _sse(
                    "error",
                    {
                        "code": "GENERATION_FAILED",
                        "message": COPY["generation_error"],
                        "technical_details": type(error).__name__,
                    },
                )

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    assets_path = dist_path / "assets"
    if assets_path.is_dir():
        application.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @application.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        requested = (dist_path / full_path).resolve()
        if dist_path in requested.parents and requested.is_file():
            return FileResponse(requested)
        index = dist_path / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Frontend build not found")

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("olfactory.api:app", host="127.0.0.1", port=8000, workers=1)


if __name__ == "__main__":
    main()
