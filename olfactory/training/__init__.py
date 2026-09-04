"""Reproducible benchmark, calibration, and model-development utilities."""

from ..prediction import MoleculePredictor, PredictionBatch
from .calibration import CalibrationBundle
from .metrics import multilabel_metrics
from .benchmark import (
    assert_no_leakage,
    build_benchmark_manifest,
    dataset_fingerprint,
    load_immutable_manifest,
    save_immutable_manifest,
    split_from_payload,
)
from .deepchem_judge import (
    DeepChemGraphJudge,
    DeepChemJudge,
    DeepChemJudgePredictor,
    load_deepchem_predictor,
    make_deepchem_featurizer,
)
from .registry import ModelRegistry, verify_artifact_manifest
from .gates import PromotionDecision, creator_promotion_gate, judge_promotion_gate
from .splits import (
    FoldManifest,
    SplitManifest,
    chemical_group_calibrated_split,
    chemical_group_folds,
    chemical_group_split,
)

__all__ = [
    "CalibrationBundle",
    "ModelRegistry",
    "verify_artifact_manifest",
    "FoldManifest",
    "PromotionDecision",
    "SplitManifest",
    "chemical_group_split",
    "chemical_group_calibrated_split",
    "chemical_group_folds",
    "creator_promotion_gate",
    "judge_promotion_gate",
    "multilabel_metrics",
    "dataset_fingerprint",
    "build_benchmark_manifest",
    "save_immutable_manifest",
    "load_immutable_manifest",
    "assert_no_leakage",
    "split_from_payload",
    "DeepChemGraphJudge",
    "DeepChemJudge",
    "DeepChemJudgePredictor",
    "make_deepchem_featurizer",
    "load_deepchem_predictor",
    "MoleculePredictor",
    "PredictionBatch",
]
