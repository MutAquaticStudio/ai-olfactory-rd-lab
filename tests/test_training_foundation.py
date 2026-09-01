import json

import numpy as np
import torch

from olfactory.training.calibration import CalibrationBundle
from olfactory.training.baselines import train_masked_morgan_mlp
from olfactory.training.creator_v2 import ConditionalSELFIESTransformer, robust_target_fit
from olfactory.training.judge_v2 import effective_positive_weights, masked_multitask_loss
from olfactory.training.gates import creator_promotion_gate, judge_promotion_gate
from olfactory.training.metrics import intensity_metrics, multilabel_metrics
from olfactory.training.registry import ModelRegistry, sha256_file
from olfactory.training.splits import chemical_group_folds, chemical_group_split


def test_chemical_split_is_deterministic_and_blocks_connectivity_leakage():
    smiles = [
        "C[C@H](O)C(=O)O",
        "C[C@@H](O)C(=O)O",
        "CCO",
        "CCCO",
        "CCCCO",
        "c1ccccc1",
        "Cc1ccccc1",
        "C1CCCCC1",
        "CC1CCCCC1",
    ]
    labels = np.eye(len(smiles), 3, dtype=float)
    first = chemical_group_split(smiles, labels, seed=7)
    second = chemical_group_split(smiles, labels, seed=7)

    assert first.split_hash == second.split_hash
    split_by_index = {}
    for split_name, indices in (
        ("train", first.train_indices),
        ("validation", first.validation_indices),
        ("test", first.test_indices),
    ):
        for index in indices:
            split_by_index[index] = split_name
    assert split_by_index[0] == split_by_index[1]
    assert set(first.train_indices).isdisjoint(first.validation_indices)
    assert set(first.train_indices).isdisjoint(first.test_indices)


def test_grouped_five_fold_split_is_deterministic_and_complete():
    smiles = ["CCO", "OCC", "CCN", "CCC", "c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CCCl", "CCBr", "CCF"]
    labels = np.eye(len(smiles), 3, dtype=float)
    first = chemical_group_folds(smiles, labels, fold_count=5, seed=17)
    second = chemical_group_folds(smiles, labels, fold_count=5, seed=17)
    assert first.fold_hash == second.fold_hash
    assert sorted(index for fold in first.folds for index in fold) == list(range(len(smiles)))
    membership = {}
    for fold_index, fold in enumerate(first.folds):
        for row in fold:
            membership.setdefault(first.group_ids[row], set()).add(fold_index)
    assert all(len(folds) == 1 for folds in membership.values())


def test_masked_metrics_ignore_unassessed_targets():
    targets = np.array([[1.0, np.nan], [0.0, 1.0], [1.0, np.nan]])
    probabilities = np.array([[0.9, 0.99], [0.1, 0.8], [0.7, 0.01]])
    metrics = multilabel_metrics(targets, probabilities, support_threshold=1)

    assert metrics["micro_average_precision"] == 1.0
    assert metrics["macro_average_precision_supported"] == 1.0
    assert metrics["positive_support"] == [2, 1]

    intensity = intensity_metrics(
        np.array([[5.0, np.nan], [7.0, 1.0]]),
        np.array([[4.0, 100.0], [8.0, 1.0]]),
    )
    assert intensity["masked_mae"] == 2 / 3
    assert intensity["count"] == 3


def test_calibration_round_trip_and_thresholds(tmp_path):
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(240, 2))
    targets = np.column_stack(
        [
            (logits[:, 0] + rng.normal(scale=0.6, size=240) > 0).astype(float),
            (logits[:, 1] + rng.normal(scale=1.2, size=240) > 1.4).astype(float),
        ]
    )
    bundle = CalibrationBundle.fit(logits, targets, ["common", "rare"], minimum_support=50)
    path = tmp_path / "calibration.json"
    bundle.save(path)
    loaded = CalibrationBundle.load(path)

    transformed = loaded.transform_logits(logits)
    assert transformed.shape == targets.shape
    assert np.all((transformed >= 0) & (transformed <= 1))
    assert all(0.02 <= threshold <= 0.98 for threshold in loaded.thresholds)
    assert loaded.methods[0] == "per_label_platt"


def test_model_registry_promotes_atomically_and_verifies_checksum(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights-v1")
    registry = ModelRegistry(tmp_path / "model_registry.json")
    entry = {
        "model_version": "judge-v2-test",
        "weights_path": str(weights),
        "weights_sha256": sha256_file(weights),
    }
    registry.promote("judge", entry)

    assert registry.production("judge") == entry
    assert registry.verify_entry(entry)
    payload = json.loads((tmp_path / "model_registry.json").read_text())
    assert payload["schema_version"] == 1


def test_judge_v2_loss_masks_unknown_presence_and_intensity():
    presence = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    intensity = torch.tensor([[6.0, float("nan")], [float("nan"), 4.0]])
    weights = effective_positive_weights(presence, torch.isfinite(presence))
    loss, parts = masked_multitask_loss(
        torch.zeros_like(presence),
        torch.zeros_like(intensity),
        presence,
        intensity,
        weights,
        0.3,
    )

    assert torch.isfinite(loss)
    assert parts["presence_loss"] > 0
    assert parts["intensity_loss"] > 0


def test_conditional_creator_contract_and_robust_fit_penalty():
    model = ConditionalSELFIESTransformer(
        vocab_size=12,
        condition_size=9,
        d_model=32,
        nhead=4,
        layers=2,
        max_length=12,
    )
    logits = model(torch.ones((2, 4), dtype=torch.long), torch.zeros((2, 9)))
    target_fit, robust = robust_target_fit(
        np.array([[0.8, 0.7], [0.4, 0.9], [0.9, 0.5]])
    )

    assert logits.shape == (2, 4, 12)
    assert 0 < robust <= target_fit <= 1


def test_promotion_gates_require_calibration_intensity_and_blind_panel():
    judge = judge_promotion_gate(
        {
            "macro_average_precision_supported": 0.30,
            "micro_average_precision": 0.43,
            "mean_label_ece": 0.08,
        },
        {
            "macro_average_precision_supported": 0.34,
            "micro_average_precision": 0.425,
            "mean_label_ece": 0.07,
        },
        bootstrap_macro_delta_lower=0.01,
        baseline_intensity_mae=2.0,
        candidate_intensity_mae=1.7,
    )
    assert judge.eligible

    creator = creator_promotion_gate(
        {"validity": 0.99, "canonical_uniqueness": 0.94, "chemistry_pass_rate": 0.75},
        target_enrichment_ci_lower=0.02,
        diversity_not_degraded=True,
        ood_not_increased=True,
        blind_panel_effect_ci_lower=None,
    )
    assert not creator.eligible
    assert "blind_panel" in creator.blocked_reasons


def test_morgan_training_smoke_is_deterministic():
    generator = torch.Generator().manual_seed(9)
    features = torch.randint(0, 2, (12, 2048), generator=generator).float()
    targets = torch.randint(0, 2, (12, 113), generator=generator).float()
    first = train_masked_morgan_mlp(
        features,
        targets,
        range(8),
        range(8, 12),
        seed=13,
        max_epochs=2,
        patience=2,
        device=torch.device("cpu"),
    )
    second = train_masked_morgan_mlp(
        features,
        targets,
        range(8),
        range(8, 12),
        seed=13,
        max_epochs=2,
        patience=2,
        device=torch.device("cpu"),
    )
    assert first.best_validation_loss == second.best_validation_loss
    assert all(
        torch.equal(first.model.state_dict()[key], second.model.state_dict()[key])
        for key in first.model.state_dict()
    )
