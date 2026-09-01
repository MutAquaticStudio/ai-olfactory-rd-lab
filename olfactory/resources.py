"""Local model and reference-data loading without UI dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Set, Tuple

import pandas as pd
import torch
from rdkit import Chem, rdBase

from .features import canonical_isomeric_smiles
from .models import ODOR_LABEL_COUNT, OdorPredictor, SMILES_LSTM, select_device


PAD_TOKEN = "<PAD>"
END_TOKEN = "<END>"


def load_odor_model(
    dataset_path: Path,
    weights_path: Path,
) -> Tuple[OdorPredictor, Tuple[str, ...]]:
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    label_names = tuple(str(label) for label in dataset.label_names)
    if len(label_names) != ODOR_LABEL_COUNT or len(set(label_names)) != ODOR_LABEL_COUNT:
        raise ValueError("The odor dataset must expose 113 unique label names")
    model = OdorPredictor()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(select_device()).eval()
    return model, label_names


def load_training_fingerprints(dataset_path: Path) -> torch.Tensor:
    """Load the immutable CPU fingerprint matrix used by the legacy Judge."""
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    tensors = getattr(dataset, "tensors", None)
    if not tensors or len(tensors) < 2:
        raise ValueError("The odor dataset must expose X and Y tensors")
    features = tensors[0]
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 2048:
        raise ValueError("The odor dataset must expose a [N, 2048] fingerprint tensor")
    return features.detach().to(dtype=torch.bool, device="cpu")


def load_smiles_model(
    vocab_path: Path,
    weights_path: Path,
) -> Tuple[SMILES_LSTM, Dict[str, int], Tuple[str, ...]]:
    with vocab_path.open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    char_to_idx = {
        str(token): int(index)
        for token, index in vocabulary["char_to_idx"].items()
    }
    idx_to_char = tuple(str(token) for token in vocabulary["idx_to_char"])
    if PAD_TOKEN not in char_to_idx or END_TOKEN not in char_to_idx:
        raise ValueError("The SMILES vocabulary must contain <PAD> and <END>")
    if len(char_to_idx) != len(idx_to_char):
        raise ValueError("Vocabulary directions have different sizes")
    if any(idx_to_char[index] != token for token, index in char_to_idx.items()):
        raise ValueError("Vocabulary character/index mappings are inconsistent")

    model = SMILES_LSTM(len(idx_to_char), char_to_idx[PAD_TOKEN])
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(select_device()).eval()
    return model, char_to_idx, idx_to_char


def load_existing_isomeric_smiles_set(dataset_path: Path) -> Set[str]:
    frame = pd.read_csv(dataset_path)
    smiles_column = next(
        (column for column in frame.columns if column.lower() == "smiles"),
        None,
    )
    if smiles_column is None:
        raise ValueError("clean_dataset.csv has no SMILES column")
    keys: Set[str] = set()
    with rdBase.BlockLogs():
        for value in frame[smiles_column].dropna():
            molecule = Chem.MolFromSmiles(str(value).strip())
            if molecule is not None:
                keys.add(canonical_isomeric_smiles(molecule))
    if not keys:
        raise ValueError("No valid SMILES were loaded from clean_dataset.csv")
    return keys
