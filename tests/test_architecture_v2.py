"""Contracts for the accuracy-first model boundary and split manifest."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from olfactory.models import OdorPredictor
from olfactory.prediction import EnsemblePredictor, LegacyMorganPredictor, PredictionBatch
from olfactory.prediction_integrity import PredictionIdentity
from olfactory.training.benchmark import (
    assert_no_leakage,
    build_benchmark_manifest,
    load_immutable_manifest,
    save_immutable_manifest,
)
from olfactory.training.registry import sha256_file
from olfactory.training.dataset import MolecularTargetTable
from olfactory.training.deepchem_judge import DeepChemGraphJudge


def test_prediction_batch_has_113_outputs_and_preserves_missing_values():
    values = np.zeros((2, 113), dtype=np.float32)
    batch = PredictionBatch(
        "model", "data", "calibration", values, np.full_like(values, np.nan),
        np.full_like(values, np.nan), np.asarray([0.7, np.nan], dtype=np.float32),
        ("IN_DOMAIN", "OUT_OF_DOMAIN"), tuple(f"label-{i}" for i in range(113)),
    )
    assert batch.to_payload()["rows"][0]["presence_predictions"][0]["expected_intensity"] is None
    assert batch.to_payload()["rows"][1]["nearest_training_similarity"] is None


def test_legacy_predictor_implements_common_contract():
    model = OdorPredictor().eval()
    predictor = LegacyMorganPredictor(
        model,
        tuple(f"label-{i}" for i in range(113)),
        identity=PredictionIdentity("v1", "d1", "uncalibrated", "BASELINE"),
    )
    result = predictor.predict(["CCO"])
    assert result.presence_probability.shape == (1, 113)
    assert result.label_names[0] == "label-0"


def test_ensemble_reports_mean_and_epistemic_spread():
    class Stub:
        label_names = tuple(f"label-{i}" for i in range(113))

        def __init__(self, value):
            self.value = value

        def predict(self, smiles):
            matrix = np.full((len(smiles), 113), self.value, dtype=np.float32)
            return PredictionBatch(
                "stub", "data", "cal", matrix, np.full_like(matrix, np.nan),
                np.full_like(matrix, np.nan), np.full((len(smiles),), 0.8),
                ("IN_DOMAIN",) * len(smiles), self.label_names,
            )

    result = EnsemblePredictor([Stub(0.2), Stub(0.8)]).predict(["CCO"])
    assert result.presence_probability[0, 0] == pytest.approx(0.5)
    assert result.ensemble_uncertainty[0, 0] == pytest.approx(0.3)


def test_deepchem_graph_model_has_dual_113_heads_without_importing_deepchem():
    graph = SimpleNamespace(
        node_features=np.ones((3, 4), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        edge_features=np.ones((2, 2), dtype=np.float32),
    )
    model = DeepChemGraphJudge(4, 2)
    presence, intensity = model([graph])
    assert presence.shape == intensity.shape == (1, 113)


def test_split_manifest_is_immutable_and_validated(tmp_path):
    table = MolecularTargetTable(
        smiles=("CCO", "OCC", "c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CCN"),
        label_names=tuple(f"label-{i}" for i in range(113)),
        presence=np.zeros((6, 113), dtype=np.float32),
        intensity=np.full((6, 113), np.nan, dtype=np.float32),
        source=("test",) * 6,
        stereo_state=("RESOLVED",) * 6,
    )
    payload = build_benchmark_manifest(table, dataset_version="test-v1", seed=4)
    path = save_immutable_manifest(tmp_path / "split.json", payload)
    assert load_immutable_manifest(path, table=table)["dataset_version"] == "test-v1"
    with pytest.raises(FileExistsError):
        save_immutable_manifest(path, {**payload, "dataset_version": "different"})
    assert_no_leakage(payload)


def test_registry_requires_a_passing_quality_gate(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"candidate")
    from olfactory.training.registry import ModelRegistry

    registry = ModelRegistry(tmp_path / "registry.json")
    entry = {"weights_path": str(weights), "weights_sha256": sha256_file(weights)}
    with pytest.raises(ValueError):
        registry.promote_after_gate("judge", entry, SimpleNamespace(eligible=False, blocked_reasons=("macro_ap",)))
    registry.promote_after_gate("judge", entry, SimpleNamespace(eligible=True, blocked_reasons=()))
    assert registry.production("judge")["status"] == "PRODUCTION"
