"""Explicit scientific promotion gates; no model is promoted implicitly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .metrics import multilabel_metrics


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    observed: Optional[float]
    requirement: str


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    checks: Sequence[GateCheck]
    blocked_reasons: Sequence[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "eligible": self.eligible,
            "checks": [asdict(check) for check in self.checks],
            "blocked_reasons": list(self.blocked_reasons),
        }


def grouped_bootstrap_delta(
    targets: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    group_ids: Sequence[str],
    *,
    metric: str = "macro_average_precision_supported",
    iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap molecule groups, keeping correlated stereo/connectivity rows together."""
    if len(group_ids) != len(targets):
        raise ValueError("Group IDs must align with target rows")
    groups: Dict[str, np.ndarray] = {}
    for index, group in enumerate(group_ids):
        groups.setdefault(str(group), []).append(index)
    groups = {key: np.asarray(value, dtype=int) for key, value in groups.items()}
    keys = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        indices = np.concatenate([groups[str(key)] for key in sampled])
        baseline = float(multilabel_metrics(targets[indices], baseline_probabilities[indices])[metric])
        candidate = float(multilabel_metrics(targets[indices], candidate_probabilities[indices])[metric])
        if np.isfinite(baseline) and np.isfinite(candidate):
            deltas.append(candidate - baseline)
    if not deltas:
        return {"mean": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
    values = np.asarray(deltas)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def judge_promotion_gate(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    *,
    bootstrap_macro_delta_lower: float,
    baseline_intensity_mae: float,
    candidate_intensity_mae: float,
) -> PromotionDecision:
    baseline_macro = float(baseline["macro_average_precision_supported"])
    candidate_macro = float(candidate["macro_average_precision_supported"])
    baseline_micro = float(baseline["micro_average_precision"])
    candidate_micro = float(candidate["micro_average_precision"])
    baseline_ece = float(baseline["mean_label_ece"])
    candidate_ece = float(candidate["mean_label_ece"])
    checks = (
        GateCheck("macro_ap_relative_gain", candidate_macro >= baseline_macro * 1.10, candidate_macro / max(baseline_macro, 1e-12) - 1.0, ">= 10%"),
        GateCheck("bootstrap_macro_delta", bootstrap_macro_delta_lower > 0, bootstrap_macro_delta_lower, "95% CI lower bound > 0"),
        GateCheck("micro_ap_retention", candidate_micro >= baseline_micro * 0.98, candidate_micro / max(baseline_micro, 1e-12) - 1.0, ">= -2%"),
        GateCheck("calibration", candidate_ece <= baseline_ece, candidate_ece - baseline_ece, "ECE must not increase"),
        GateCheck("intensity_mae", candidate_intensity_mae <= baseline_intensity_mae * 0.90, candidate_intensity_mae / max(baseline_intensity_mae, 1e-12) - 1.0, "<= -10%"),
    )
    blocked = tuple(check.name for check in checks if not check.passed)
    return PromotionDecision(not blocked, checks, blocked)


def creator_promotion_gate(
    benchmark: Dict[str, float],
    *,
    target_enrichment_ci_lower: Optional[float],
    diversity_not_degraded: bool,
    ood_not_increased: bool,
    blind_panel_effect_ci_lower: Optional[float],
) -> PromotionDecision:
    target_count = int(benchmark.get("targets_per_profile", 0))
    required_coverage = {1: 0.80, 2: 0.60, 3: 0.40}.get(target_count, 1.0)
    observed_coverage = benchmark.get("runs_with_three_strict")
    checks = (
        GateCheck("validity", benchmark.get("validity", 0.0) >= 0.98, benchmark.get("validity"), ">= 0.98"),
        GateCheck("canonical_uniqueness", benchmark.get("canonical_uniqueness", 0.0) >= 0.90, benchmark.get("canonical_uniqueness"), ">= 0.90"),
        GateCheck("chemistry_pass_rate", benchmark.get("chemistry_pass_rate", 0.0) >= 0.70, benchmark.get("chemistry_pass_rate"), ">= 0.70"),
        GateCheck("target_enrichment", target_enrichment_ci_lower is not None and target_enrichment_ci_lower > 0, target_enrichment_ci_lower, "bootstrap 95% CI lower bound > 0"),
        GateCheck(
            "strict_top3_coverage",
            observed_coverage is not None and float(observed_coverage) >= required_coverage,
            float(observed_coverage) if observed_coverage is not None else None,
            f">= {required_coverage:.2f} for {target_count or 'unknown'} target(s)",
        ),
        GateCheck("diversity_retention", diversity_not_degraded, float(diversity_not_degraded), "must not degrade"),
        GateCheck("ood_control", ood_not_increased, float(ood_not_increased), "must not increase materially"),
        GateCheck("blind_panel", blind_panel_effect_ci_lower is not None and blind_panel_effect_ci_lower > 0, blind_panel_effect_ci_lower, "prospective effect 95% CI lower bound > 0"),
    )
    blocked = tuple(check.name for check in checks if not check.passed)
    return PromotionDecision(not blocked, checks, blocked)
