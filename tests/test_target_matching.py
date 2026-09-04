import numpy as np
import pytest

from olfactory.target_matching import (
    DescriptorMaturity,
    TargetMatchTier,
    conservative_probability,
    descriptor_evidence,
    evaluate_target_match,
    maturity_from_support,
    select_target_matches,
)


def test_maturity_does_not_count_unassessed_rows_as_negative_support():
    assert maturity_from_support(80, 0) is DescriptorMaturity.LIMITED_EVIDENCE
    assert maturity_from_support(80, 50) is DescriptorMaturity.SUPPORTED
    assert maturity_from_support(9, 500) is DescriptorMaturity.INSUFFICIENT


def test_conservative_probability_uses_one_sided_ensemble_penalty():
    assert conservative_probability(0.50, 0.10) == pytest.approx(0.336)
    assert conservative_probability(0.10, 0.20) == 0.0


def test_supported_targets_require_each_target_and_geometric_fit_gate():
    evidence = descriptor_evidence(
        ["floral", "woody"],
        [100, 100],
        [100, 100],
        [0.2, 0.2],
        ["per_label_platt", "per_label_platt"],
    )
    passing = evaluate_target_match(
        [0.50, 0.40], [0.0, 0.0], evidence,
        calibrated=True,
    )
    low_fit = evaluate_target_match(
        [0.31, 0.31], [0.0, 0.0], evidence,
        calibrated=True,
    )

    assert passing.met_requested_gate is True
    assert passing.tier is TargetMatchTier.STRICT
    assert passing.robust_target_fit == pytest.approx(np.sqrt(0.2))
    assert low_fit.met_requested_gate is False
    assert low_fit.requested_fit_floor == 0.40


def test_limited_label_uses_its_calibration_threshold_not_absolute_thirty_percent():
    evidence = descriptor_evidence(
        ["musk"], [20], [0], [0.12], ["rare_tier_platt"]
    )
    match = evaluate_target_match(
        [0.15], [0.0], evidence,
        calibrated=True,
    )

    assert match.targets[0].maturity is DescriptorMaturity.LIMITED_EVIDENCE
    assert match.uses_absolute_probability_gate is False
    assert match.met_requested_gate is True
    assert match.requested_fit_floor == pytest.approx(0.12)


def test_insufficient_descriptor_cannot_be_used_as_target():
    evidence = descriptor_evidence(["unsupported"], [5], [100])
    with pytest.raises(ValueError, match="Insufficient-evidence"):
        evaluate_target_match([0.9], [0.0], evidence, calibrated=True)


def test_uncalibrated_score_is_never_presented_as_a_strict_match():
    evidence = descriptor_evidence(
        ["floral"], [100], [100], [0.30], ["uncalibrated"]
    )
    match = evaluate_target_match(
        [0.95], [0.0], evidence,
        calibrated=False,
    )

    assert match.met_requested_gate is False
    assert match.tier is TargetMatchTier.RELAXED
    assert match.calibrated is False


def test_selection_relaxes_transparently_until_three_rows_qualify():
    evidence = descriptor_evidence(
        ["floral"], [100], [100], [0.5], ["per_label_platt"]
    )
    probabilities = np.asarray([[0.50], [0.39], [0.35], [0.20]])
    selected = select_target_matches(
        probabilities,
        np.zeros_like(probabilities),
        evidence,
        count=3,
        calibrated=True,
    )

    assert [index for index, _ in selected] == [0, 1, 2]
    assert selected[0][1].tier is TargetMatchTier.STRICT
    assert selected[0][1].met_requested_gate is True
    assert selected[1][1].tier is TargetMatchTier.RELAXED
    assert selected[1][1].met_requested_gate is False
    assert selected[1][1].relaxation_factor == pytest.approx(0.85)
    assert selected[1][1].applied_fit_floor == pytest.approx(0.34)
