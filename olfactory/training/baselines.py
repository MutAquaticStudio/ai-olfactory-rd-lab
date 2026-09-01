"""Shared-split logistic and chiral Morgan MLP scientific baselines."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..models import OdorPredictor
from .judge_v2 import effective_positive_weights


def fit_ovr_logistic(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    evaluation_features: np.ndarray,
) -> Tuple[np.ndarray, Sequence[object]]:
    """Fit independent masked logistic models; constant labels use train prevalence."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError("Logistic baseline requires requirements-training.txt") from error
    probabilities = np.zeros((len(evaluation_features), train_targets.shape[1]), dtype=np.float32)
    estimators = []
    for label in range(train_targets.shape[1]):
        selected = np.isfinite(train_targets[:, label])
        target = train_targets[selected, label]
        if not len(target):
            prevalence = 0.0
            estimators.append({"constant": prevalence})
            probabilities[:, label] = prevalence
        elif len(np.unique(target)) < 2:
            prevalence = float(target[0])
            estimators.append({"constant": prevalence})
            probabilities[:, label] = prevalence
        else:
            estimator = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )
            estimator.fit(train_features[selected], target.astype(int))
            estimators.append(estimator)
            probabilities[:, label] = estimator.predict_proba(evaluation_features)[:, 1]
    return probabilities, estimators


@dataclass(frozen=True)
class MorganTrainingResult:
    model: OdorPredictor
    best_validation_loss: float
    epochs: int


def train_masked_morgan_mlp(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    seed: int,
    max_epochs: int = 100,
    patience: int = 20,
    device: Optional[torch.device] = None,
) -> MorganTrainingResult:
    """Reproducible current-architecture baseline on the leakage-resistant split."""
    torch.manual_seed(seed)
    selected_device = device or torch.device("cpu")
    model = OdorPredictor().to(selected_device)
    train_targets = targets[list(train_indices)]
    positive_weights = effective_positive_weights(train_targets, torch.isfinite(train_targets)).to(selected_device)
    train_loader = DataLoader(
        TensorDataset(features[list(train_indices)].float(), train_targets.float()),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TensorDataset(features[list(validation_indices)].float(), targets[list(validation_indices)].float()),
        batch_size=128,
        shuffle=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def masked_loss(logits, batch_targets):
        mask = torch.isfinite(batch_targets)
        safe = torch.nan_to_num(batch_targets, nan=0.0)
        raw = nn.functional.binary_cross_entropy_with_logits(
            logits,
            safe,
            pos_weight=positive_weights,
            reduction="none",
        )
        return raw[mask].mean() if mask.any() else logits.sum() * 0.0

    best_loss = float("inf")
    best_state = None
    stale = 0
    completed = 0
    for epoch in range(1, max_epochs + 1):
        completed = epoch
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(
                model(batch_features.to(selected_device)),
                batch_targets.to(selected_device),
            )
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.inference_mode():
            for batch_features, batch_targets in validation_loader:
                losses.append(float(masked_loss(model(batch_features.to(selected_device)), batch_targets.to(selected_device)).item()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Morgan baseline produced no checkpoint")
    model.load_state_dict(best_state)
    return MorganTrainingResult(model.cpu().eval(), best_loss, completed)
