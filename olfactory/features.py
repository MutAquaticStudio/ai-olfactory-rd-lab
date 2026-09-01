"""Molecular features and model-inference helpers."""

from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

import numpy as np
import torch
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem
from torch import nn

from .models import FINGERPRINT_SIZE, ODOR_LABEL_COUNT, OdorPredictor


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def create_morgan_tensor(molecule: Chem.Mol) -> torch.Tensor:
    """Create the exact 2,048-bit chiral Morgan feature used by the model."""
    with rdBase.BlockLogs():
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=2,
            nBits=FINGERPRINT_SIZE,
            useChirality=True,
        )
    values = np.zeros((FINGERPRINT_SIZE,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return torch.from_numpy(values)


def predict_probabilities(
    model: OdorPredictor,
    molecules: Sequence[Chem.Mol],
) -> torch.Tensor:
    """Return a CPU probability matrix with one row per molecule."""
    if not molecules:
        return torch.empty((0, ODOR_LABEL_COUNT), dtype=torch.float32)
    features = torch.stack([create_morgan_tensor(mol) for mol in molecules])
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(features.to(model_device(model))))
    return probabilities.cpu()


def top_descriptors(
    probabilities: torch.Tensor,
    label_names: Sequence[str],
    count: int,
    excluded_indices: Optional[Set[int]] = None,
) -> Tuple[Tuple[str, float], ...]:
    """Return the highest-scoring descriptors, optionally excluding targets."""
    excluded = excluded_indices or set()
    available = max(0, probabilities.numel() - len(excluded))
    if not available:
        return ()
    scores = probabilities.clone()
    for index in excluded:
        scores[index] = float("-inf")
    values, indices = torch.topk(scores, k=min(count, available))
    return tuple(
        (str(label_names[index.item()]), float(value.item()))
        for value, index in zip(values, indices)
    )


def canonical_isomeric_smiles(molecule: Chem.Mol) -> str:
    """Canonical key that intentionally preserves defined stereochemistry."""
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def smiles_representations(molecule: Chem.Mol) -> Tuple[str, str]:
    """Return canonical isomeric and connectivity-only SMILES."""
    return canonical_isomeric_smiles(molecule), Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )


def geometric_mean(probabilities: torch.Tensor) -> float:
    """Stable geometric mean used by the existing Target fit calculation."""
    safe = probabilities.clamp_min(1e-12)
    return float(torch.exp(torch.log(safe).mean()).item())
