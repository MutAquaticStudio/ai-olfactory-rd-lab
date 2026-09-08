import json
from pathlib import Path

import numpy as np
import pytest
import torch

import olfactory.training.judge_ensemble as ensemble_module
from olfactory.prediction_integrity import PredictionIdentity
from olfactory.training.benchmark import dataset_fingerprint
from olfactory.training.calibration import CalibrationBundle
from olfactory.training.dataset import MolecularTargetTable
from olfactory.training.judge_ensemble import (
    ChempropEnsemblePredictor,
    build_chemprop_ensemble_artifact,
)
from olfactory.training.registry import sha256_file
from olfactory.training.splits import SplitManifest


def identity_calibration(labels):
    return CalibrationBundle(
        tuple(labels),
        tuple(1.0 for _ in labels),
        tuple(0.0 for _ in labels),
        tuple(0.5 for _ in labels),
        tuple("fixture" for _ in labels),
        "ensemble-platt-tier-v1",
    )


def test_shadow_predictor_uses_mean_logits_and_reports_member_uncertainty(monkeypatch):
    labels = tuple(f"label-{index}" for index in range(113))
    monkeypatch.setattr(ensemble_module, "_inference_loader", lambda *_: object())
    monkeypatch.setattr(
        ensemble_module,
        "_member_outputs",
        lambda model, _loader: (
            np.full((1, 113), float(model), dtype=np.float32),
            np.zeros((1, 113), dtype=np.float32),
        ),
    )
    from rdkit import Chem
    from olfactory.features import create_morgan_tensor

    fingerprint = create_morgan_tensor(Chem.MolFromSmiles("CCO")).to(torch.uint8)
    predictor = ChempropEnsemblePredictor(
        models=(0, 1, 2, 3, 4),
        label_names=labels,
        identity=PredictionIdentity(
            "judge-v2-shadow-fixture",
            "fixture-data",
            "ensemble-platt-tier-v1",
            "SHADOW_ONLY",
        ),
        calibration=identity_calibration(labels),
        training_fingerprints=fingerprint.reshape(1, -1),
        intensity_available=False,
    )

    result = predictor.predict(["CCO"])

    assert result.presence_probability[0, 0] == pytest.approx(1 / (1 + np.exp(-2)))
    assert result.ensemble_uncertainty[0, 0] == pytest.approx(
        np.std([1 / (1 + np.exp(-value)) for value in range(5)])
    )
    assert np.isnan(result.expected_intensity).all()
    assert result.training_similarity[0] == pytest.approx(1.0)
    assert result.reliability_state == ("IN_DOMAIN",)


def test_ensemble_artifact_is_immutable_shadow_with_missing_intensity_gate(tmp_path):
    labels = tuple(f"label-{index}" for index in range(113))
    presence = np.zeros((4, 113), dtype=np.float32)
    presence[0, 0] = 1
    presence[1, 1] = 1
    table = MolecularTargetTable(
        smiles=("CCO", "CCN", "CCC", "CCCl"),
        label_names=labels,
        presence=presence,
        intensity=np.full_like(presence, np.nan),
        source=("fixture",) * 4,
        stereo_state=("RESOLVED",) * 4,
    )
    split = SplitManifest(
        train_indices=(0,),
        calibration_indices=(1,),
        validation_indices=(2,),
        test_indices=(3,),
        group_ids=("a", "b", "c", "d"),
        seed=42,
        ratios=(0.25, 0.25, 0.25, 0.25),
        similarity_threshold=0.6,
        split_hash="fixture-split",
    )
    member_paths = []
    for member, seed in enumerate((11, 17, 23, 31, 43)):
        run = tmp_path / f"member-{member}"
        run.mkdir()
        arrays = run / "raw_predictions.npz"
        np.savez_compressed(
            arrays,
            calibration_indices=np.array([1]),
            calibration_logits=np.full((1, 113), member / 10, dtype=np.float32),
            calibration_intensity=np.zeros((1, 113), dtype=np.float32),
            calibration_presence=presence[[1]],
            calibration_intensity_targets=np.full((1, 113), np.nan, dtype=np.float32),
            validation_indices=np.array([2]),
            validation_logits=np.full((1, 113), member / 10, dtype=np.float32),
            validation_intensity=np.zeros((1, 113), dtype=np.float32),
            validation_presence=presence[[2]],
            validation_intensity_targets=np.full((1, 113), np.nan, dtype=np.float32),
            locked_test_indices=np.array([3]),
            locked_test_logits=np.full((1, 113), member / 10, dtype=np.float32),
            locked_test_intensity=np.zeros((1, 113), dtype=np.float32),
            locked_test_presence=presence[[3]],
            locked_test_intensity_targets=np.full((1, 113), np.nan, dtype=np.float32),
        )
        curve = run / "learning_curve.json"
        curve.write_text(
            json.dumps(
                [
                    {
                        "epoch": 1.0,
                        "train_loss": 1.0,
                        "validation_loss": 1.0,
                        "validation_macro_ap": 0.1,
                        "validation_micro_ap": 0.1,
                        "validation_ece": 0.1,
                        "validation_intensity_mae": float("nan"),
                    }
                ]
            ),
            encoding="utf-8",
        )
        curve.with_suffix(".png").write_bytes(b"fixture")
        manifest = {
            "run_id": f"member-{member}",
            "model_version": f"member-{member}",
            "dataset_sha256": dataset_fingerprint(table),
            "split_hash": split.split_hash,
            "seed": seed,
            "intensity_weight": 0.3,
            "raw_predictions_path": str(arrays),
            "raw_predictions_sha256": sha256_file(arrays),
            "learning_curve_path": str(curve.with_suffix('.png')),
        }
        path = run / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        member_paths.append(path)

    artifact = build_chemprop_ensemble_artifact(
        table,
        split,
        member_paths,
        tmp_path / "artifacts",
        dataset_version="fixture-data",
        bootstrap_iterations=10,
    )

    assert artifact["status"] == "SHADOW_ONLY"
    assert artifact["production_promoted"] is False
    assert artifact["locked_test_status"] == "EXPOSED_RETROSPECTIVE_TEST"
    manifest_path = Path(artifact["manifest_path"])
    assert manifest_path.with_suffix(".sha256").is_file()
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted["calibration_path"] == "calibration.json"
    assert persisted["members"][0]["manifest_path"].startswith("../member-")
    gate = json.loads(Path(artifact["gate_report_path"]).read_text(encoding="utf-8"))
    assert gate["eligible"] is False
    intensity = next(check for check in gate["checks"] if check["name"] == "intensity_mae")
    assert intensity["status"] == "NOT_EVALUABLE"
