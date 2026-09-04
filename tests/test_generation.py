from itertools import cycle

import pytest
import torch
from rdkit import Chem

import olfactory.generation as generation_module
from olfactory.chemistry import (
    ChemicalDecision,
    ConformerEnsembleResult,
    ConformerRecord,
    screen_molecule,
)
from olfactory.generation import (
    GenerationPhase,
    ScreenedCandidate,
    generate_candidate_pool,
    generate_target_aligned_pool,
    rank_candidates,
)
from olfactory.models import OdorPredictor
from olfactory.pubchem import NoveltyResult, NoveltyStatus
from olfactory.references import (
    MatchLevel,
    ReferenceCheckBundle,
    ReferenceEvidence,
    ReferenceGate,
    ReferenceGateStatus,
    ReferenceStatus,
)


class StaticPubChem:
    def __init__(self, statuses=None):
        self.statuses = iter(statuses or cycle([NoveltyStatus.NOT_FOUND]))
        self.calls = []

    def verify(self, smiles, *, consent):
        self.calls.append((smiles, consent))
        status = next(self.statuses)
        return NoveltyResult(status, smiles)


class StaticReferenceVerifier:
    def __init__(self, bundle):
        self.bundle = bundle
        self.calls = []

    def verify(self, query, *, consents):
        self.calls.append((query, set(consents)))
        return self.bundle


def available_ensemble(_smiles=""):
    return ConformerEnsembleResult(
        conformers=(ConformerRecord("test molblock", 0.0),),
        method="MMFF94s",
        requested_count=50,
        embedded_count=10,
        converged_count=8,
        is_macrocycle=False,
    )


def sampler_from(values):
    values = iter(values)

    def sample(**_kwargs):
        return next(values)

    return sample


def exhaust(stream):
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as completed:
            return events, completed.value


def test_generation_stops_exactly_at_five_accepted_candidates():
    values = ["CCO", "CCCO", "CCCCO", "CCOC", "CC(=O)OC", "CCCCCO"]
    client = StaticPubChem()
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=client,
        consent=True,
        sampler=sampler_from(values),
        conformer_builder=available_ensemble,
    )
    events, result = exhaust(stream)
    assert len(result.accepted_candidates) == 5
    assert result.attempts == 5
    assert sum(event.phase is GenerationPhase.ACCEPTED for event in events) == 5


def test_generation_attempt_limit_and_counters_are_exact():
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=StaticPubChem(),
        consent=True,
        sampler=lambda **_kwargs: "not-a-smiles",
        max_attempts=3,
        conformer_builder=available_ensemble,
    )
    _, result = exhaust(stream)
    assert result.attempts == 3
    assert result.invalid == 3
    assert result.reached_attempt_limit


def test_generation_time_limit_can_stop_before_first_attempt():
    times = iter([0.0, 121.0, 121.0, 121.0])
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=StaticPubChem(),
        consent=True,
        sampler=lambda **_kwargs: "CCO",
        clock=lambda: next(times),
        max_seconds=120.0,
        conformer_builder=available_ensemble,
    )
    _, result = exhaust(stream)
    assert result.attempts == 0
    assert result.reached_time_limit


def candidate(smiles, decision, novelty_status):
    molecule = Chem.MolFromSmiles(smiles)
    screen = screen_molecule(molecule)
    if decision is not screen.decision:
        screen = type(screen)(
            decision=decision,
            reason_codes=screen.reason_codes,
            descriptors=screen.descriptors,
            is_macrocycle=screen.is_macrocycle,
            macrocycle_ring_size=screen.macrocycle_ring_size,
            macrocycle_carbon_fraction=screen.macrocycle_carbon_fraction,
            macrocycle_heteroatoms=screen.macrocycle_heteroatoms,
            alerts=screen.alerts,
        )
    return ScreenedCandidate(
        raw_smiles=smiles,
        isomeric_smiles=Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        canonical_smiles=Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False),
        molecule=molecule,
        chemical_screen=screen,
        novelty=NoveltyResult(novelty_status, smiles),
        conformer_ensemble=available_ensemble(),
    )


def test_review_found_and_unverified_never_reach_shortlist():
    labels = [f"label-{index}" for index in range(113)]
    model = OdorPredictor()
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    candidates = [
        candidate("CCO", ChemicalDecision.PASS, NoveltyStatus.NOT_FOUND),
        candidate("CCCO", ChemicalDecision.REVIEW, NoveltyStatus.NOT_FOUND),
        candidate("CCCCO", ChemicalDecision.PASS, NoveltyStatus.FOUND),
        candidate("CCOC", ChemicalDecision.PASS, NoveltyStatus.UNVERIFIED),
    ]
    ranked = rank_candidates(model, labels, ["label-0", "label-1"], candidates)
    assert len(ranked) == 1
    assert ranked[0].isomeric_smiles == "CCO"
    assert ranked[0].target_fit == pytest.approx(0.5)


def test_unresolved_stereo_variants_are_checked_in_target_fit_order():
    client = StaticPubChem()
    scores = []

    def scorer(variants):
        ordered = [
            Chem.MolToSmiles(item, canonical=True, isomericSmiles=True)
            for item in variants
        ]
        scores.extend(ordered)
        return [0.1, 0.9]

    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=client,
        consent=True,
        required_count=1,
        max_attempts=1,
        sampler=lambda **_kwargs: "CC(O)C(=O)O",
        variant_scorer=scorer,
        conformer_builder=available_ensemble,
    )
    events, result = exhaust(stream)

    assert len(scores) == 2
    assert client.calls[0][0] == scores[1]
    assert result.accepted_candidates[0].isomeric_smiles == scores[1]
    assert any(event.phase is GenerationPhase.STEREO_ENUMERATION for event in events)
    assert any(event.phase is GenerationPhase.RANKING for event in events)
    assert any(event.phase is GenerationPhase.PREPARING_3D for event in events)


def test_more_than_four_candidate_variants_goes_to_review(monkeypatch):
    base_screen = screen_molecule(Chem.MolFromSmiles("CCO"))
    monkeypatch.setattr(generation_module, "screen_molecule", lambda _: base_screen)
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=StaticPubChem(),
        consent=True,
        required_count=1,
        max_attempts=1,
        sampler=lambda **_kwargs: "CC(F)C(Cl)C(Br)C",
        conformer_builder=available_ensemble,
    )
    events, result = exhaust(stream)

    assert not result.accepted_candidates
    assert result.review_queue[0].chemical_screen.reason_codes == (
        "STEREO_VARIANT_LIMIT",
    )
    assert any(event.phase is GenerationPhase.STEREO_REVIEW for event in events)


def test_duplicate_connectivity_never_accepts_two_stereoisomers():
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=StaticPubChem(),
        consent=True,
        required_count=2,
        max_attempts=3,
        sampler=sampler_from(["C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O", "CCO"]),
        conformer_builder=available_ensemble,
    )
    _, result = exhaust(stream)

    assert len(result.accepted_candidates) == 2
    assert result.duplicates == 1
    assert len({item.canonical_smiles for item in result.accepted_candidates}) == 2


def test_candidate_without_converged_3d_is_not_accepted():
    unavailable = ConformerEnsembleResult(
        conformers=(),
        method=None,
        requested_count=50,
        embedded_count=12,
        converged_count=0,
        is_macrocycle=False,
        error="NO_CONVERGED_VALID_CONFORMER",
    )
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        pubchem_client=StaticPubChem(),
        consent=True,
        required_count=1,
        max_attempts=1,
        sampler=lambda **_kwargs: "CCO",
        conformer_builder=lambda _: unavailable,
    )
    _, result = exhaust(stream)

    assert not result.accepted_candidates
    assert result.review_queue[0].chemical_screen.reason_codes == (
        "CONFORMER_UNAVAILABLE",
    )


def reference_bundle(catalog_status):
    evidences = (
        ReferenceEvidence(
            "PUBCHEM",
            ReferenceStatus.NO_MATCH,
            None,
            "CCO",
        ),
        ReferenceEvidence(
            "TGSC",
            catalog_status,
            MatchLevel.EXACT_CONNECTIVITY
            if catalog_status is ReferenceStatus.MATCH
            else None,
            "LFQSCWFLJHTTHZ",
            error_code="TIMEOUT"
            if catalog_status is ReferenceStatus.UNVERIFIED
            else None,
        ),
    )
    gate_status = (
        ReferenceGateStatus.BLOCKED_MATCH
        if catalog_status is ReferenceStatus.MATCH
        else ReferenceGateStatus.REVIEW_REQUIRED
        if catalog_status is ReferenceStatus.UNVERIFIED
        else ReferenceGateStatus.PASS
    )
    blockers = () if gate_status is ReferenceGateStatus.PASS else ("TGSC",)
    return ReferenceCheckBundle(
        evidences,
        ReferenceGate(gate_status, blockers),
    )


@pytest.mark.parametrize(
    ("catalog_status", "phase", "reason_code"),
    [
        (
            ReferenceStatus.MATCH,
            GenerationPhase.CATALOG_MATCH,
            "KNOWN_IN_INDUSTRY_CATALOG",
        ),
        (
            ReferenceStatus.UNVERIFIED,
            GenerationPhase.REFERENCE_UNVERIFIED,
            "REFERENCE_UNVERIFIED",
        ),
    ],
)
def test_catalog_match_or_unverified_is_blocked_and_sent_to_reference_review(
    catalog_status,
    phase,
    reason_code,
):
    verifier = StaticReferenceVerifier(reference_bundle(catalog_status))
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        reference_verifier=verifier,
        reference_consents={"PUBCHEM"},
        required_count=1,
        max_attempts=1,
        sampler=lambda **_kwargs: "CCO",
        conformer_builder=available_ensemble,
    )

    events, result = exhaust(stream)

    assert not result.accepted_candidates
    assert len(result.review_queue) == 1
    assert result.review_queue[0].review_category == "REFERENCE"
    assert result.review_queue[0].chemical_screen.reason_codes == (reason_code,)
    assert any(event.phase is phase for event in events)


def test_all_enabled_reference_sources_must_return_no_match_before_acceptance():
    verifier = StaticReferenceVerifier(reference_bundle(ReferenceStatus.NO_MATCH))
    stream = generate_candidate_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        reference_verifier=verifier,
        reference_consents={"PUBCHEM"},
        required_count=1,
        max_attempts=1,
        sampler=lambda **_kwargs: "CCO",
        conformer_builder=available_ensemble,
    )

    events, result = exhaust(stream)

    assert len(result.accepted_candidates) == 1
    assert result.accepted_candidates[0].reference_gate.status is ReferenceGateStatus.PASS
    phases = [event.phase for event in events]
    assert phases.index(GenerationPhase.CHECKING_REFERENCES) < phases.index(
        GenerationPhase.REFERENCE_ACCEPTED
    ) < phases.index(GenerationPhase.PREPARING_3D)


def test_target_aligned_pool_scores_full_pool_before_reference_and_uses_best_scores():
    operation_log = []

    class OrderedVerifier(StaticReferenceVerifier):
        def verify(self, query, *, consents):
            operation_log.append(("reference", query.isomeric_smiles))
            return super().verify(query, consents=consents)

    def score(variants):
        operation_log.append(("score", len(variants)))
        return [float(molecule.GetNumHeavyAtoms()) for molecule in variants]

    stream = generate_target_aligned_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        reference_verifier=OrderedVerifier(reference_bundle(ReferenceStatus.NO_MATCH)),
        reference_consents={"PUBCHEM"},
        variant_scorer=score,
        required_count=2,
        provisional_pool_size=3,
        max_attempts=3,
        sampler=sampler_from(["CCO", "CCCO", "CCCCO"]),
        conformer_builder=available_ensemble,
    )

    events, result = exhaust(stream)

    first_reference = next(index for index, item in enumerate(operation_log) if item[0] == "reference")
    assert all(item[0] == "score" for item in operation_log[:first_reference])
    assert [item.isomeric_smiles for item in result.accepted_candidates] == [
        "CCCCO",
        "CCCO",
    ]
    phases = [event.phase for event in events]
    assert phases.count(GenerationPhase.TARGET_SCORING) == 3
    assert phases.index(GenerationPhase.RANKING) < phases.index(
        GenerationPhase.CHECKING_REFERENCES
    )


def test_target_aligned_pool_supplies_elite_prefix_during_refinement():
    starts = []
    values = iter(["CCO", "CCCO", "CCCCO", "CCCCCO"])

    def sampler(**kwargs):
        starts.append(kwargs.get("start_smiles"))
        return next(values)

    stream = generate_target_aligned_pool(
        creator_model=None,
        char_to_idx={},
        idx_to_char=(),
        temperature=0.8,
        existing_isomeric_smiles_set=set(),
        reference_verifier=StaticReferenceVerifier(reference_bundle(ReferenceStatus.NO_MATCH)),
        reference_consents={"PUBCHEM"},
        variant_scorer=lambda variants: [float(item.GetNumHeavyAtoms()) for item in variants],
        required_count=1,
        provisional_pool_size=4,
        max_attempts=4,
        sampler=sampler,
        conformer_builder=available_ensemble,
    )

    exhaust(stream)

    assert starts[:2] == [None, None]
    assert starts[2] in {"CCO", "CCCO"}
    assert starts[3] is not None
