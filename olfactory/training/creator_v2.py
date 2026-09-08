"""Conditional SELFIES Transformer and generator-quality evaluation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from torch import nn

from ..chemistry import ChemicalDecision, screen_molecule
from ..features import canonical_isomeric_smiles


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
END_TOKEN = "<END>"


class ConditionalSELFIESTransformer(nn.Module):
    """Six-layer autoregressive decoder conditioned on odor and property vectors."""

    def __init__(
        self,
        vocab_size: int,
        condition_size: int = 455,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 6,
        dropout: float = 0.1,
        max_length: int = 160,
    ):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length + 1, d_model)
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_size, d_model),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        batch, length = token_ids.shape
        if length > self.max_length:
            raise ValueError("Token sequence exceeds configured maximum")
        positions = torch.arange(length + 1, device=token_ids.device).unsqueeze(0)
        condition_token = self.condition_projection(conditions).unsqueeze(1)
        tokens = self.token_embedding(token_ids)
        sequence = torch.cat([condition_token, tokens], dim=1)
        sequence = sequence + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length + 1, length + 1, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.transformer(sequence, mask=causal_mask)
        return self.output(encoded[:, 1:])


def condition_vector(
    presence: np.ndarray,
    intensity: np.ndarray,
    molecule: Chem.Mol,
    *,
    presence_mask: Optional[np.ndarray] = None,
    intensity_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    raw_presence = np.asarray(presence, dtype=np.float32)
    raw_intensity = np.asarray(intensity, dtype=np.float32)
    if raw_presence.shape != raw_intensity.shape:
        raise ValueError("Presence and intensity vectors must align")
    assessed = (
        np.isfinite(raw_presence).astype(np.float32)
        if presence_mask is None
        else np.asarray(presence_mask, dtype=np.float32)
    )
    measured = (
        np.isfinite(raw_intensity).astype(np.float32)
        if intensity_mask is None
        else np.asarray(intensity_mask, dtype=np.float32)
    )
    if assessed.shape != raw_presence.shape or measured.shape != raw_intensity.shape:
        raise ValueError("Condition masks must align with descriptor vectors")
    presence_values = np.nan_to_num(raw_presence, nan=0.0)
    intensity_values = np.nan_to_num(raw_intensity, nan=0.0) / 10.0
    properties = np.asarray(
        [
            Descriptors.ExactMolWt(molecule) / 330.0,
            (Descriptors.MolLogP(molecule) + 1.0) / 8.0,
            Descriptors.TPSA(molecule) / 100.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [presence_values, assessed, intensity_values, measured, properties]
    )


def target_condition_vector(
    label_names: Sequence[str],
    target_descriptors: Sequence[str],
    *,
    reference_smiles: str = "CCOC(=O)C",
) -> np.ndarray:
    """Build a target condition without forcing unselected labels absent."""
    labels = tuple(str(value) for value in label_names)
    index_by_label = {label: index for index, label in enumerate(labels)}
    unknown = sorted(set(target_descriptors) - set(labels))
    if unknown:
        raise ValueError(f"Unknown target descriptors: {', '.join(unknown)}")
    if not 1 <= len(target_descriptors) <= 3:
        raise ValueError("Target-conditioned sampling requires one to three descriptors")
    presence = np.full(len(labels), np.nan, dtype=np.float32)
    intensity = np.full(len(labels), np.nan, dtype=np.float32)
    for label in target_descriptors:
        presence[index_by_label[label]] = 1.0
    molecule = Chem.MolFromSmiles(reference_smiles)
    if molecule is None:
        raise ValueError("Reference property SMILES is invalid")
    return condition_vector(presence, intensity, molecule)


def build_selfies_vocabulary(smiles: Sequence[str]) -> Dict[str, object]:
    try:
        import selfies as sf
    except ImportError as error:
        raise RuntimeError("Creator v2 requires requirements-training.txt") from error
    encoded = [sf.encoder(value) for value in smiles]
    alphabet = sorted(sf.get_alphabet_from_selfies(encoded))
    tokens = [PAD_TOKEN, BOS_TOKEN, END_TOKEN, *alphabet]
    return {
        "tokens": tokens,
        "token_to_idx": {token: index for index, token in enumerate(tokens)},
        "representation": "SELFIES",
        "schema_version": 1,
    }


def encode_selfies(smiles: str, token_to_idx: Mapping[str, int]) -> List[int]:
    try:
        import selfies as sf
    except ImportError as error:
        raise RuntimeError("Creator v2 requires requirements-training.txt") from error
    symbols = list(sf.split_selfies(sf.encoder(smiles)))
    return [token_to_idx[BOS_TOKEN], *[token_to_idx[symbol] for symbol in symbols], token_to_idx[END_TOKEN]]


def decode_selfies(token_ids: Sequence[int], tokens: Sequence[str]) -> Optional[str]:
    try:
        import selfies as sf
    except ImportError as error:
        raise RuntimeError("Creator v2 requires requirements-training.txt") from error
    symbols = []
    for token_id in token_ids:
        token = tokens[int(token_id)]
        if token == END_TOKEN:
            break
        if token not in {PAD_TOKEN, BOS_TOKEN}:
            symbols.append(token)
    try:
        return sf.decoder("".join(symbols))
    except sf.DecoderError:
        return None


@torch.inference_mode()
def sample_conditioned(
    model: ConditionalSELFIESTransformer,
    condition: torch.Tensor,
    tokens: Sequence[str],
    *,
    temperature: float = 0.8,
    max_length: int = 120,
    generator: Optional[torch.Generator] = None,
    prefix_smiles: Optional[str] = None,
    prefix_fraction: float = 0.4,
) -> Optional[str]:
    token_to_idx = {token: index for index, token in enumerate(tokens)}
    prefix_ids = [token_to_idx[BOS_TOKEN]]
    if prefix_smiles:
        try:
            encoded = encode_selfies(prefix_smiles, token_to_idx)[1:-1]
        except (KeyError, RuntimeError):
            encoded = []
        retained = max(
            1,
            int(len(encoded) * float(np.clip(prefix_fraction, 0.1, 0.8))),
        )
        prefix_ids.extend(encoded[:retained])
    current = torch.tensor([prefix_ids], dtype=torch.long, device=condition.device)
    for _ in range(max(0, max_length - len(prefix_ids) + 1)):
        logits = model(current, condition.unsqueeze(0))[:, -1, :]
        logits[:, token_to_idx[PAD_TOKEN]] = float("-inf")
        logits[:, token_to_idx[BOS_TOKEN]] = float("-inf")
        probabilities = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        sampled = torch.multinomial(probabilities, 1, generator=generator)
        current = torch.cat([current, sampled], dim=1)
        if sampled.item() == token_to_idx[END_TOKEN]:
            break
    return decode_selfies(current[0].tolist(), tokens)


def robust_target_fit(probability_ensemble: np.ndarray) -> Tuple[float, float]:
    """Return geometric target fit and one-sided uncertainty-penalized fit."""
    values = np.clip(np.asarray(probability_ensemble, dtype=float), 1e-12, 1.0)
    if values.ndim != 2:
        raise ValueError("Expected models by target descriptors")
    per_model_log_fit = np.log(values).mean(axis=1)
    target_fit = float(np.exp(per_model_log_fit.mean()))
    robust_fit = float(np.exp(per_model_log_fit.mean() - 1.64 * per_model_log_fit.std()))
    return target_fit, robust_fit


def target_alignment_benchmark(
    conditional_probabilities: np.ndarray,
    unconditional_probabilities: np.ndarray,
    *,
    target_floor: float = 0.30,
    fit_floor: float = 0.40,
    bootstrap_iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compare strict target yield against an equal-budget unconditional control.

    Inputs may be precomputed conservative probabilities shaped
    ``runs × samples × targets`` or calibrated ensemble probabilities shaped
    ``runs × samples × models × targets``. The evaluation Judge must not be the
    single model optimized online by the Creator.
    """
    conditional = np.asarray(conditional_probabilities, dtype=float)
    unconditional = np.asarray(unconditional_probabilities, dtype=float)
    if conditional.ndim not in {3, 4} or conditional.shape != unconditional.shape:
        raise ValueError(
            "Target benchmark arrays must align as runs × samples × targets "
            "or runs × samples × models × targets"
        )
    if not conditional.size:
        raise ValueError("Target benchmark arrays cannot be empty")

    def conservative_values(values: np.ndarray) -> np.ndarray:
        if values.ndim == 3:
            return np.clip(values, 0.0, 1.0)
        return np.clip(values.mean(axis=2) - 1.64 * values.std(axis=2), 0.0, 1.0)

    def strict_counts(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(conservative_values(values), 1e-12, 1.0)
        fits = np.exp(np.log(clipped).mean(axis=2))
        strict = (clipped >= target_floor).all(axis=2) & (fits >= fit_floor)
        return strict.sum(axis=1).astype(float)

    conditional_counts = strict_counts(conditional)
    unconditional_counts = strict_counts(unconditional)
    deltas = conditional_counts - unconditional_counts
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_iterations, dtype=float)
    for iteration in range(bootstrap_iterations):
        rows = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[iteration] = deltas[rows].mean()
    return {
        "runs": float(conditional.shape[0]),
        "samples_per_run": float(conditional.shape[1]),
        "targets_per_profile": float(conditional.shape[-1]),
        "conditional_strict_yield_mean": float(conditional_counts.mean()),
        "unconditional_strict_yield_mean": float(unconditional_counts.mean()),
        "runs_with_three_strict": float((conditional_counts >= 3).mean()),
        "target_enrichment_delta": float(deltas.mean()),
        "target_enrichment_ci_lower": float(np.quantile(bootstrap, 0.025)),
        "target_enrichment_ci_upper": float(np.quantile(bootstrap, 0.975)),
    }


def _internal_diversity(molecules: Sequence[Chem.Mol]) -> float:
    if len(molecules) < 2:
        return 0.0
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048)
        for molecule in molecules
    ]
    distances = []
    for index in range(1, len(fingerprints)):
        distances.extend(
            1.0 - similarity
            for similarity in DataStructs.BulkTanimotoSimilarity(
                fingerprints[index], fingerprints[:index]
            )
        )
    return float(np.mean(distances)) if distances else 0.0


def evaluate_generated_smiles(
    generated: Sequence[Optional[str]],
    training_smiles: Iterable[str],
) -> Dict[str, float]:
    training = set()
    for value in training_smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is not None:
            training.add(canonical_isomeric_smiles(molecule))
    valid_molecules = []
    canonical = []
    pass_count = 0
    for value in generated:
        molecule = Chem.MolFromSmiles(value) if value else None
        if molecule is None:
            continue
        valid_molecules.append(molecule)
        key = canonical_isomeric_smiles(molecule)
        canonical.append(key)
        if screen_molecule(molecule).decision == ChemicalDecision.PASS:
            pass_count += 1
    total = max(len(generated), 1)
    unique = set(canonical)
    return {
        "sample_count": len(generated),
        "validity": len(valid_molecules) / total,
        "canonical_uniqueness": len(unique) / max(len(canonical), 1),
        "chemistry_pass_rate": pass_count / max(len(valid_molecules), 1),
        "novelty_vs_training": len(unique - training) / max(len(unique), 1),
        "internal_diversity": _internal_diversity(
            [Chem.MolFromSmiles(value) for value in sorted(unique)]
        ),
    }
