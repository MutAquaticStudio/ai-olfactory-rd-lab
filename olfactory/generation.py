"""Candidate sampling, screening, identity checks and target-fit ranking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from rdkit import Chem, rdBase

from .chemistry import (
    ChemicalDecision,
    ChemicalScreenResult,
    ConformerEnsembleResult,
    build_conformer_ensemble,
    screen_molecule,
)
from .features import (
    canonical_isomeric_smiles,
    geometric_mean,
    predict_probabilities,
    smiles_representations,
    top_descriptors,
)
from .models import OdorPredictor, SMILES_LSTM
from .pubchem import NoveltyResult, NoveltyStatus, PubChemClient
from .references import (
    PUBCHEM,
    ReferenceEvidence,
    ReferenceGate,
    ReferenceGateStatus,
    ReferenceStatus,
    ReferenceVerifier,
    build_reference_verifier,
    reference_query_from_smiles,
)
from .stereo import resolved_molecules


PAD_TOKEN = "<PAD>"
END_TOKEN = "<END>"


class GenerationPhase(str, Enum):
    SAMPLING = "SAMPLING"
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    REJECTED = "REJECTED"
    REVIEW = "REVIEW"
    PUBCHEM_CHECK = "PUBCHEM_CHECK"
    PUBCHEM_FOUND = "PUBCHEM_FOUND"
    PUBCHEM_UNVERIFIED = "PUBCHEM_UNVERIFIED"
    CHECKING_REFERENCES = "CHECKING_REFERENCES"
    CATALOG_MATCH = "CATALOG_MATCH"
    REFERENCE_UNVERIFIED = "REFERENCE_UNVERIFIED"
    REFERENCE_ACCEPTED = "REFERENCE_ACCEPTED"
    ACCEPTED = "ACCEPTED"
    STEREO_ENUMERATION = "STEREO_ENUMERATION"
    STEREO_REVIEW = "STEREO_REVIEW"
    RANKING = "RANKING"
    PREPARING_3D = "PREPARING_3D"


@dataclass(frozen=True)
class GenerationEvent:
    phase: GenerationPhase
    attempt: int
    accepted: int
    invalid: int
    duplicates: int
    rejected: int
    reviews: int
    found: int
    unverified: int
    detail: Optional[str] = None
    reference_matches: int = 0
    reference_unverified: int = 0


@dataclass(frozen=True)
class ScreenedCandidate:
    raw_smiles: str
    isomeric_smiles: str
    canonical_smiles: str
    molecule: Chem.Mol
    chemical_screen: ChemicalScreenResult
    novelty: NoveltyResult
    conformer_ensemble: ConformerEnsembleResult
    reference_checks: Tuple[ReferenceEvidence, ...] = ()
    reference_gate: Optional[ReferenceGate] = None


@dataclass(frozen=True)
class ReviewCandidate:
    isomeric_smiles: str
    chemical_screen: ChemicalScreenResult
    reference_checks: Tuple[ReferenceEvidence, ...] = ()
    reference_gate: Optional[ReferenceGate] = None
    review_category: str = "CHEMISTRY"


@dataclass(frozen=True)
class GenerationResult:
    accepted_candidates: Tuple[ScreenedCandidate, ...]
    review_queue: Tuple[ReviewCandidate, ...]
    attempts: int
    elapsed_seconds: float
    invalid: int
    duplicates: int
    rejected: int
    found: int
    unverified: int
    reached_attempt_limit: bool
    reached_time_limit: bool
    reference_matches: int = 0
    reference_unverified: int = 0


@dataclass(frozen=True)
class RankedCandidate:
    isomeric_smiles: str
    canonical_smiles: str
    target_fit: float
    target_probabilities: Tuple[Tuple[str, float], ...]
    supporting_descriptors: Tuple[Tuple[str, float], ...]
    probability_vector: Tuple[float, ...]
    chemical_screen: ChemicalScreenResult
    novelty: NoveltyResult
    conformer_ensemble: ConformerEnsembleResult
    reference_checks: Tuple[ReferenceEvidence, ...] = ()
    reference_gate: Optional[ReferenceGate] = None


def sample_smiles_string(
    model: SMILES_LSTM,
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
    temperature: float,
    start_str: str = "C",
    max_len: int = 60,
) -> str:
    """Sample one raw SMILES character by character."""
    if temperature <= 0:
        raise ValueError("Sampling diversity must be greater than zero")
    unknown = set(start_str) - set(char_to_idx)
    if unknown:
        raise ValueError("Start string contains tokens outside the vocabulary")

    device = next(model.parameters()).device
    token_ids = torch.tensor(
        [[char_to_idx[character] for character in start_str]],
        dtype=torch.long,
        device=device,
    )
    generated = start_str
    pad_idx = char_to_idx[PAD_TOKEN]
    end_idx = char_to_idx[END_TOKEN]

    with torch.inference_mode():
        logits, hidden = model(token_ids)
        for _ in range(max_len - len(start_str)):
            sampling_logits = logits[0, -1] / temperature
            sampling_logits[pad_idx] = float("-inf")
            probabilities = torch.softmax(sampling_logits, dim=-1)
            next_idx = int(torch.multinomial(probabilities, 1).item())
            if next_idx == end_idx:
                break
            generated += idx_to_char[next_idx]
            next_token = torch.tensor([[next_idx]], dtype=torch.long, device=device)
            logits, hidden = model(next_token, hidden)
    return generated


def _event(
    phase: GenerationPhase,
    attempt: int,
    accepted: int,
    counters: Dict[str, int],
    detail: Optional[str] = None,
) -> GenerationEvent:
    return GenerationEvent(
        phase=phase,
        attempt=attempt,
        accepted=accepted,
        invalid=counters["invalid"],
        duplicates=counters["duplicates"],
        rejected=counters["rejected"],
        reviews=counters["reviews"],
        found=counters["found"],
        unverified=counters["unverified"],
        detail=detail,
        reference_matches=counters["reference_matches"],
        reference_unverified=counters["reference_unverified"],
    )


def _review_result(
    screen: ChemicalScreenResult,
    reason_code: str,
) -> ChemicalScreenResult:
    return ChemicalScreenResult(
        decision=ChemicalDecision.REVIEW,
        reason_codes=(reason_code,),
        descriptors=screen.descriptors,
        is_macrocycle=screen.is_macrocycle,
        macrocycle_ring_size=screen.macrocycle_ring_size,
        macrocycle_carbon_fraction=screen.macrocycle_carbon_fraction,
        macrocycle_heteroatoms=screen.macrocycle_heteroatoms,
        alerts=screen.alerts,
    )


def generate_candidate_pool(
    *,
    creator_model: SMILES_LSTM,
    char_to_idx: Dict[str, int],
    idx_to_char: Sequence[str],
    temperature: float,
    existing_isomeric_smiles_set: Set[str],
    pubchem_client: Optional[PubChemClient] = None,
    consent: bool = False,
    reference_verifier: Optional[ReferenceVerifier] = None,
    reference_consents: Iterable[str] = (),
    required_count: int = 5,
    max_attempts: int = 200,
    max_seconds: float = 120.0,
    stereo_limit: int = 4,
    sampler: Callable[..., str] = sample_smiles_string,
    variant_scorer: Optional[Callable[[Sequence[Chem.Mol]], Sequence[float]]] = None,
    conformer_builder: Callable[[str], ConformerEnsembleResult] = build_conformer_ensemble,
    clock: Callable[[], float] = time.monotonic,
) -> Generator[GenerationEvent, None, GenerationResult]:
    """Yield live events and return stereo-resolved, fail-closed candidates."""
    if reference_verifier is None:
        if pubchem_client is None:
            raise ValueError("A reference verifier or PubChem client is required")
        reference_verifier = build_reference_verifier(pubchem_client)
    effective_consents = {str(provider).upper() for provider in reference_consents}
    if consent:
        effective_consents.add(PUBCHEM)
    accepted: List[ScreenedCandidate] = []
    review_queue: List[ReviewCandidate] = []
    generated_connectivity_keys: Set[str] = set()
    counters = {
        "invalid": 0,
        "duplicates": 0,
        "rejected": 0,
        "reviews": 0,
        "found": 0,
        "unverified": 0,
        "reference_matches": 0,
        "reference_unverified": 0,
    }
    attempts = 0
    start_time = clock()
    stopped_for_time = False

    def timed_out() -> bool:
        return clock() - start_time >= max_seconds

    while len(accepted) < required_count and attempts < max_attempts:
        if timed_out():
            break
        attempts += 1
        yield _event(GenerationPhase.SAMPLING, attempts, len(accepted), counters)
        raw_smiles = sampler(
            model=creator_model,
            char_to_idx=char_to_idx,
            idx_to_char=idx_to_char,
            temperature=temperature,
        )
        try:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(raw_smiles, sanitize=True)
            if molecule is None:
                raise ValueError("invalid SMILES")
        except Exception:
            counters["invalid"] += 1
            yield _event(GenerationPhase.INVALID, attempts, len(accepted), counters)
            continue

        _, connectivity_key = smiles_representations(molecule)
        if connectivity_key in generated_connectivity_keys:
            counters["duplicates"] += 1
            yield _event(GenerationPhase.DUPLICATE, attempts, len(accepted), counters)
            continue
        generated_connectivity_keys.add(connectivity_key)

        chemical_screen = screen_molecule(molecule)
        if chemical_screen.decision is ChemicalDecision.REJECT:
            counters["rejected"] += 1
            yield _event(
                GenerationPhase.REJECTED,
                attempts,
                len(accepted),
                counters,
                ", ".join(chemical_screen.reason_codes),
            )
            continue
        if chemical_screen.decision is ChemicalDecision.REVIEW:
            counters["reviews"] += 1
            isomeric_smiles = canonical_isomeric_smiles(molecule)
            review_queue.append(ReviewCandidate(isomeric_smiles, chemical_screen))
            yield _event(
                GenerationPhase.REVIEW,
                attempts,
                len(accepted),
                counters,
                ", ".join(chemical_screen.reason_codes),
            )
            continue

        variants = resolved_molecules(molecule, limit=stereo_limit)
        if variants is None:
            counters["reviews"] += 1
            unresolved_smiles = canonical_isomeric_smiles(molecule)
            review_queue.append(
                ReviewCandidate(
                    unresolved_smiles,
                    _review_result(chemical_screen, "STEREO_VARIANT_LIMIT"),
                )
            )
            yield _event(
                GenerationPhase.STEREO_REVIEW,
                attempts,
                len(accepted),
                counters,
                "STEREO_VARIANT_LIMIT",
            )
            continue
        if len(variants) > 1:
            yield _event(
                GenerationPhase.STEREO_ENUMERATION,
                attempts,
                len(accepted),
                counters,
                f"{len(variants)} variants",
            )

        if timed_out():
            stopped_for_time = True
            break
        scores = (
            list(variant_scorer(variants))
            if variant_scorer is not None
            else [0.0] * len(variants)
        )
        if len(scores) != len(variants):
            raise ValueError("Variant scorer returned an unexpected number of scores")
        yield _event(GenerationPhase.RANKING, attempts, len(accepted), counters)
        ranked_variants = sorted(
            zip(scores, variants),
            key=lambda item: (-float(item[0]), canonical_isomeric_smiles(item[1])),
        )

        selected = False
        for _, variant in ranked_variants:
            if timed_out():
                stopped_for_time = True
                break
            isomeric_smiles = canonical_isomeric_smiles(variant)
            if isomeric_smiles in existing_isomeric_smiles_set:
                counters["duplicates"] += 1
                yield _event(
                    GenerationPhase.DUPLICATE,
                    attempts,
                    len(accepted),
                    counters,
                )
                continue

            yield _event(
                GenerationPhase.CHECKING_REFERENCES,
                attempts,
                len(accepted),
                counters,
            )
            # Kept for one release so existing SSE consumers retain PubChem progress.
            yield _event(
                GenerationPhase.PUBCHEM_CHECK,
                attempts,
                len(accepted),
                counters,
            )
            reference_bundle = reference_verifier.verify(
                reference_query_from_smiles(isomeric_smiles),
                consents=effective_consents,
            )
            novelty = reference_bundle.pubchem_novelty(isomeric_smiles)
            pubchem_evidence = reference_bundle.evidence_for(PUBCHEM)
            matching_evidence = tuple(
                item
                for item in reference_bundle.evidences
                if item.status is ReferenceStatus.MATCH
            )
            uncertain_evidence = tuple(
                item
                for item in reference_bundle.evidences
                if item.status
                in {ReferenceStatus.AMBIGUOUS, ReferenceStatus.UNVERIFIED}
            )
            if pubchem_evidence and pubchem_evidence.status is ReferenceStatus.MATCH:
                counters["found"] += 1
                yield _event(
                    GenerationPhase.PUBCHEM_FOUND,
                    attempts,
                    len(accepted),
                    counters,
                )
            if (
                pubchem_evidence
                and pubchem_evidence.status is ReferenceStatus.UNVERIFIED
            ):
                counters["unverified"] += 1
                yield _event(
                    GenerationPhase.PUBCHEM_UNVERIFIED,
                    attempts,
                    len(accepted),
                    counters,
                    novelty.error_code,
                )
            catalog_matches = tuple(
                item for item in matching_evidence if item.provider != PUBCHEM
            )
            if catalog_matches:
                yield _event(
                    GenerationPhase.CATALOG_MATCH,
                    attempts,
                    len(accepted),
                    counters,
                    ", ".join(item.provider for item in catalog_matches),
                )
            if matching_evidence:
                counters["reference_matches"] += 1
            if uncertain_evidence:
                counters["reference_unverified"] += 1
                yield _event(
                    GenerationPhase.REFERENCE_UNVERIFIED,
                    attempts,
                    len(accepted),
                    counters,
                    ", ".join(item.provider for item in uncertain_evidence),
                )
            if reference_bundle.gate.status is not ReferenceGateStatus.PASS:
                counters["reviews"] += 1
                reason_code = (
                    "KNOWN_IN_INDUSTRY_CATALOG"
                    if catalog_matches
                    else "KNOWN_IN_REFERENCE_SOURCE"
                    if matching_evidence
                    else "REFERENCE_UNVERIFIED"
                )
                review_queue.append(
                    ReviewCandidate(
                        isomeric_smiles,
                        _review_result(chemical_screen, reason_code),
                        reference_checks=reference_bundle.evidences,
                        reference_gate=reference_bundle.gate,
                        review_category="REFERENCE",
                    )
                )
                continue
            yield _event(
                GenerationPhase.REFERENCE_ACCEPTED,
                attempts,
                len(accepted),
                counters,
                ", ".join(
                    item.provider
                    for item in reference_bundle.evidences
                    if item.status is ReferenceStatus.NO_MATCH
                ),
            )

            if timed_out():
                stopped_for_time = True
                break
            yield _event(
                GenerationPhase.PREPARING_3D,
                attempts,
                len(accepted),
                counters,
            )
            ensemble = conformer_builder(isomeric_smiles)
            if not ensemble.available:
                counters["reviews"] += 1
                review_queue.append(
                    ReviewCandidate(
                        isomeric_smiles,
                        _review_result(chemical_screen, "CONFORMER_UNAVAILABLE"),
                    )
                )
                yield _event(
                    GenerationPhase.REVIEW,
                    attempts,
                    len(accepted),
                    counters,
                    ensemble.error,
                )
                continue

            _, canonical_smiles = smiles_representations(variant)
            accepted.append(
                ScreenedCandidate(
                    raw_smiles=raw_smiles,
                    isomeric_smiles=isomeric_smiles,
                    canonical_smiles=canonical_smiles,
                    molecule=variant,
                    chemical_screen=chemical_screen,
                    novelty=novelty,
                    conformer_ensemble=ensemble,
                    reference_checks=reference_bundle.evidences,
                    reference_gate=reference_bundle.gate,
                )
            )
            yield _event(GenerationPhase.ACCEPTED, attempts, len(accepted), counters)
            selected = True
            break
        if stopped_for_time:
            break
        if selected:
            continue

    elapsed = clock() - start_time
    return GenerationResult(
        accepted_candidates=tuple(accepted),
        review_queue=tuple(review_queue),
        attempts=attempts,
        elapsed_seconds=elapsed,
        invalid=counters["invalid"],
        duplicates=counters["duplicates"],
        rejected=counters["rejected"],
        found=counters["found"],
        unverified=counters["unverified"],
        reached_attempt_limit=attempts >= max_attempts and len(accepted) < required_count,
        reached_time_limit=(stopped_for_time or elapsed >= max_seconds)
        and len(accepted) < required_count,
        reference_matches=counters["reference_matches"],
        reference_unverified=counters["reference_unverified"],
    )


def rank_candidates(
    judge_model: OdorPredictor,
    label_names: Sequence[str],
    target_descriptors: Sequence[str],
    candidates: Sequence[ScreenedCandidate],
) -> List[RankedCandidate]:
    """Rank only chemistry/reference PASS structures using geometric mean."""
    if not target_descriptors:
        return []
    eligible = [
        candidate
        for candidate in candidates
        if candidate.chemical_screen.decision is ChemicalDecision.PASS
        and candidate.novelty.status is NoveltyStatus.NOT_FOUND
        and (
            candidate.reference_gate is None
            or candidate.reference_gate.status is ReferenceGateStatus.PASS
        )
    ]
    if not eligible:
        return []

    label_to_index = {str(label): index for index, label in enumerate(label_names)}
    missing = [label for label in target_descriptors if label not in label_to_index]
    if missing:
        raise ValueError(f"Unknown target descriptors: {', '.join(missing)}")
    target_indices = [label_to_index[label] for label in target_descriptors]
    excluded = set(target_indices)
    probability_matrix = predict_probabilities(
        judge_model,
        [candidate.molecule for candidate in eligible],
    )
    ranked: List[RankedCandidate] = []
    for candidate, probabilities in zip(eligible, probability_matrix):
        ranked.append(
            RankedCandidate(
                isomeric_smiles=candidate.isomeric_smiles,
                canonical_smiles=candidate.canonical_smiles,
                target_fit=geometric_mean(probabilities[target_indices]),
                target_probabilities=tuple(
                    (label, float(probabilities[index].item()))
                    for label, index in zip(target_descriptors, target_indices)
                ),
                supporting_descriptors=top_descriptors(
                    probabilities,
                    label_names,
                    count=3,
                    excluded_indices=excluded,
                ),
                probability_vector=tuple(float(value) for value in probabilities.tolist()),
                chemical_screen=candidate.chemical_screen,
                novelty=candidate.novelty,
                conformer_ensemble=candidate.conformer_ensemble,
                reference_checks=candidate.reference_checks,
                reference_gate=candidate.reference_gate,
            )
        )
    return sorted(ranked, key=lambda item: item.target_fit, reverse=True)
