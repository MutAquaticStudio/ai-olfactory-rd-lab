"""Local model and reference-data loading without UI dependencies."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Sequence, Set, Tuple

import pandas as pd
import torch
from rdkit import Chem, rdBase

from .features import canonical_isomeric_smiles
from .models import ODOR_LABEL_COUNT, OdorPredictor, SMILES_LSTM, select_device
from .target_matching import DescriptorEvidence, descriptor_evidence_from_payload
from .training.creator_v2 import ConditionalSELFIESTransformer


PAD_TOKEN = "<PAD>"
END_TOKEN = "<END>"
RESOURCE_DIR_ENV = "SCENT_STUDIO_RESOURCE_DIR"
RESOURCE_MANIFEST_NAME = "resource_manifest.json"
PRIVATE_RESOURCE_FILES = (
    "odor_morgan_tensor_dataset.pt",
    "odor_predictor_weights.pth",
    "smiles_creator_weights.pth",
)


class ResourceBundleError(RuntimeError):
    """Stable, user-safe error raised when private model resources are invalid."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def default_resource_dir() -> Path:
    configured = os.environ.get(RESOURCE_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".scent-molecule-studio" / "resources").resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resource_bundle(resource_dir: Path | str | None = None) -> Path:
    """Validate the private model bundle and return its resolved directory."""

    root = Path(resource_dir or default_resource_dir()).expanduser().resolve()
    manifest_path = root / RESOURCE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ResourceBundleError(
            "RESOURCE_BUNDLE_MISSING",
            f"Private resource manifest is missing: {manifest_path}",
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ResourceBundleError(
            "RESOURCE_MANIFEST_INVALID",
            f"Private resource manifest cannot be read: {manifest_path}",
        ) from error

    if payload.get("schema_version") != 1:
        raise ResourceBundleError(
            "RESOURCE_MANIFEST_INVALID",
            "Private resource manifest has an unsupported schema version.",
        )

    files = payload.get("files")
    if not isinstance(files, dict):
        raise ResourceBundleError(
            "RESOURCE_MANIFEST_INVALID",
            "Private resource manifest must contain a files mapping.",
        )
    for name in PRIVATE_RESOURCE_FILES:
        expected = files.get(name)
        if not isinstance(expected, str) or not expected:
            raise ResourceBundleError(
                "RESOURCE_MANIFEST_INVALID",
                f"Private resource manifest has no checksum for {name}.",
            )
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ResourceBundleError(
                "RESOURCE_FILE_MISSING",
                f"Private resource file is missing: {path}",
            )
        if _sha256_file(path) != expected.lower():
            raise ResourceBundleError(
                "RESOURCE_CHECKSUM_MISMATCH",
                f"Private resource checksum mismatch: {name}",
            )
    return root


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


def load_label_positive_support(dataset_path: Path) -> Tuple[int, ...]:
    """Return positive counts without interpreting catalog zeros as assessed negatives."""
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    tensors = getattr(dataset, "tensors", None)
    if not tensors or len(tensors) < 2:
        raise ValueError("The odor dataset must expose X and Y tensors")
    labels = tensors[1]
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 2
        or labels.shape[1] != ODOR_LABEL_COUNT
    ):
        raise ValueError("The odor dataset must expose a [N, 113] label tensor")
    return tuple(int(value) for value in (labels > 0).sum(dim=0).tolist())


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


def load_conditioned_smiles_model(
    weights_path: Path,
) -> Tuple[ConditionalSELFIESTransformer, Tuple[str, ...]]:
    """Load a promoted masked-condition SELFIES artifact."""
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    tokens = tuple(str(value) for value in checkpoint.get("tokens", ()))
    if not tokens or len(set(tokens)) != len(tokens):
        raise ValueError("Conditional Creator artifact has an invalid vocabulary")
    condition_size = int(checkpoint.get("condition_size", 0))
    if condition_size != 455:
        raise ValueError("Conditional Creator artifact does not use condition schema v2")
    architecture = checkpoint.get("architecture", {})
    model = ConditionalSELFIESTransformer(
        len(tokens),
        condition_size=condition_size,
        d_model=int(architecture.get("d_model", 256)),
        nhead=int(architecture.get("heads", 8)),
        layers=int(architecture.get("layers", 6)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(select_device()).eval()
    return model, tokens


def load_descriptor_evidence(
    path: Path,
    label_names: Sequence[str],
) -> Tuple[DescriptorEvidence, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Descriptor evidence must be a JSON list")
    records = descriptor_evidence_from_payload(payload)
    if tuple(item.name for item in records) != tuple(str(name) for name in label_names):
        raise ValueError("Descriptor evidence label order does not match the model")
    return records


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
