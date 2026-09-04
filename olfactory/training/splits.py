"""Leakage-resistant split construction for small-molecule odor datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


@dataclass(frozen=True)
class SplitManifest:
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    test_indices: Tuple[int, ...]
    group_ids: Tuple[str, ...]
    seed: int
    ratios: Tuple[float, ...]
    similarity_threshold: float
    split_hash: str
    calibration_indices: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        return {key: list(value) if isinstance(value, tuple) else value for key, value in payload.items()}


@dataclass(frozen=True)
class FoldManifest:
    folds: Tuple[Tuple[int, ...], ...]
    group_ids: Tuple[str, ...]
    seed: int
    similarity_threshold: float
    fold_hash: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "folds": [list(fold) for fold in self.folds],
            "group_ids": list(self.group_ids),
            "seed": self.seed,
            "similarity_threshold": self.similarity_threshold,
            "fold_hash": self.fold_hash,
        }


def _fingerprint(molecule: Chem.Mol):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=True,
    )
    return generator.GetFingerprint(molecule)


def _acyclic_clusters(
    indices: Sequence[int],
    molecules: Sequence[Chem.Mol],
    threshold: float,
) -> List[Tuple[int, ...]]:
    if not indices:
        return []
    fingerprints = [_fingerprint(molecules[index]) for index in indices]
    distances: List[float] = []
    for position in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[position],
            fingerprints[:position],
        )
        distances.extend(1.0 - value for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        1.0 - threshold,
        isDistData=True,
    )
    return [tuple(indices[position] for position in cluster) for cluster in clusters]


def chemical_groups(
    smiles: Sequence[str],
    similarity_threshold: float = 0.6,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return split group and connectivity key for every structure."""
    molecules: List[Chem.Mol] = []
    connectivity_keys: List[str] = []
    cyclic_groups: Dict[str, List[int]] = {}
    acyclic_indices: List[int] = []
    with rdBase.BlockLogs():
        for index, value in enumerate(smiles):
            molecule = Chem.MolFromSmiles(str(value), sanitize=True)
            if molecule is None:
                raise ValueError(f"Invalid SMILES at index {index}")
            molecules.append(molecule)
            inchikey = Chem.MolToInchiKey(molecule)
            connectivity_keys.append(inchikey.split("-")[0])
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule,
                includeChirality=False,
            )
            if scaffold:
                cyclic_groups.setdefault(scaffold, []).append(index)
            else:
                acyclic_indices.append(index)

    group_ids = [""] * len(smiles)
    for scaffold, indices in sorted(cyclic_groups.items()):
        key = f"scaffold:{scaffold}"
        for index in indices:
            group_ids[index] = key
    for cluster_number, cluster in enumerate(
        _acyclic_clusters(acyclic_indices, molecules, similarity_threshold)
    ):
        key = f"acyclic:{cluster_number:05d}"
        for index in cluster:
            group_ids[index] = key

    # Connectivity identity is a stronger boundary than an accidental cluster split.
    connectivity_to_groups: Dict[str, set] = {}
    for index, connectivity in enumerate(connectivity_keys):
        connectivity_to_groups.setdefault(connectivity, set()).add(group_ids[index])
    aliases: Dict[str, str] = {}
    for connectivity, members in connectivity_to_groups.items():
        canonical = sorted(members)[0]
        for member in members:
            aliases[member] = canonical
    group_ids = [aliases.get(group, group) for group in group_ids]
    return tuple(group_ids), tuple(connectivity_keys)


def _greedy_assign_groups(
    groups: Dict[str, List[int]],
    labels: np.ndarray,
    ratios: Tuple[float, ...],
    seed: int,
) -> Dict[str, int]:
    rng = np.random.default_rng(seed)
    total_rows = labels.shape[0]
    total_labels = labels.sum(axis=0).astype(float)
    target_rows = np.asarray(ratios, dtype=float) * total_rows
    target_labels = np.asarray(ratios, dtype=float)[:, None] * total_labels[None, :]
    split_count = len(ratios)
    current_rows = np.zeros(split_count, dtype=float)
    current_labels = np.zeros((split_count, labels.shape[1]), dtype=float)

    prevalence = np.maximum(total_labels, 1.0)
    ordering = []
    for key, indices in groups.items():
        group_labels = labels[indices].sum(axis=0)
        rarity = float((group_labels / prevalence).sum())
        ordering.append((key, indices, group_labels, rarity, float(rng.random())))
    ordering.sort(key=lambda item: (-len(item[1]), -item[3], item[4], item[0]))

    assignment: Dict[str, int] = {}
    for key, indices, group_labels, _, _ in ordering:
        scores = []
        for split in range(split_count):
            rows_after = current_rows.copy()
            labels_after = current_labels.copy()
            rows_after[split] += len(indices)
            labels_after[split] += group_labels
            row_error = np.mean(((rows_after - target_rows) / np.maximum(target_rows, 1.0)) ** 2)
            observed = total_labels > 0
            if observed.any():
                label_error = np.mean(
                    (
                        (labels_after[:, observed] - target_labels[:, observed])
                        / np.maximum(target_labels[:, observed], 1.0)
                    )
                    ** 2
                )
            else:
                label_error = 0.0
            overflow = max(0.0, rows_after[split] - target_rows[split]) / max(target_rows[split], 1.0)
            scores.append(row_error + 1.5 * label_error + 2.0 * overflow**2)
        selected = int(np.argmin(scores))
        assignment[key] = selected
        current_rows[selected] += len(indices)
        current_labels[selected] += group_labels
    return assignment


def _build_group_split(
    smiles: Sequence[str],
    labels: np.ndarray,
    *,
    ratios: Tuple[float, ...],
    names: Tuple[str, ...],
    seed: int,
    similarity_threshold: float,
) -> SplitManifest:
    matrix = np.asarray(labels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(smiles):
        raise ValueError("Labels must be a two-dimensional matrix aligned with SMILES")
    if (
        len(ratios) != len(names)
        or any(value <= 0 for value in ratios)
        or not np.isclose(sum(ratios), 1.0)
    ):
        raise ValueError("Split ratios must be positive, named, and sum to one")

    group_ids, connectivity_keys = chemical_groups(smiles, similarity_threshold)
    grouped: Dict[str, List[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)
    assignment = _greedy_assign_groups(
        grouped,
        np.nan_to_num(matrix, nan=0.0),
        ratios,
        seed,
    )
    splits = [[] for _ in names]
    for index, group_id in enumerate(group_ids):
        splits[assignment[group_id]].append(index)

    connectivity_splits: Dict[str, set] = {}
    for split_number, indices in enumerate(splits):
        for index in indices:
            connectivity_splits.setdefault(connectivity_keys[index], set()).add(split_number)
    leaked = [key for key, members in connectivity_splits.items() if len(members) > 1]
    if leaked:
        raise RuntimeError(f"Connectivity leakage detected for {len(leaked)} groups")

    stable = {
        name: sorted(splits[position])
        for position, name in enumerate(names)
    }
    stable.update(
        {
            "group_ids": list(group_ids),
            "seed": seed,
            "ratios": list(ratios),
            "partition_order": list(names),
            "similarity_threshold": similarity_threshold,
        }
    )
    split_hash = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SplitManifest(
        train_indices=tuple(stable["train"]),
        validation_indices=tuple(stable["validation"]),
        test_indices=tuple(stable["test"]),
        group_ids=group_ids,
        seed=seed,
        ratios=ratios,
        similarity_threshold=similarity_threshold,
        split_hash=split_hash,
        calibration_indices=tuple(stable.get("calibration", [])),
    )


def chemical_group_split(
    smiles: Sequence[str],
    labels: np.ndarray,
    *,
    ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    similarity_threshold: float = 0.6,
) -> SplitManifest:
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must contain three positive values summing to one")
    return _build_group_split(
        smiles,
        labels,
        ratios=ratios,
        names=("train", "validation", "test"),
        seed=seed,
        similarity_threshold=similarity_threshold,
    )


def chemical_group_calibrated_split(
    smiles: Sequence[str],
    labels: np.ndarray,
    *,
    ratios: Tuple[float, float, float, float] = (0.60, 0.10, 0.15, 0.15),
    seed: int = 42,
    similarity_threshold: float = 0.6,
) -> SplitManifest:
    """Build train/calibration/validation/locked-test partitions.

    Early stopping belongs to validation, probability calibration belongs to
    calibration, and the locked test is evaluated only after model selection.
    """
    if len(ratios) != 4:
        raise ValueError("Calibrated split requires train/calibration/validation/test ratios")
    return _build_group_split(
        smiles,
        labels,
        ratios=ratios,
        names=("train", "calibration", "validation", "test"),
        seed=seed,
        similarity_threshold=similarity_threshold,
    )


def chemical_group_folds(
    smiles: Sequence[str],
    labels: np.ndarray,
    *,
    fold_count: int = 5,
    seed: int = 42,
    similarity_threshold: float = 0.6,
) -> FoldManifest:
    """Build deterministic multilabel-balanced folds without splitting chemical groups."""
    matrix = np.asarray(labels, dtype=float)
    if fold_count < 2:
        raise ValueError("At least two folds are required")
    if matrix.ndim != 2 or matrix.shape[0] != len(smiles):
        raise ValueError("Labels must be aligned with SMILES")
    group_ids, connectivity_keys = chemical_groups(smiles, similarity_threshold)
    grouped: Dict[str, List[int]] = {}
    for index, group_id in enumerate(group_ids):
        grouped.setdefault(group_id, []).append(index)

    rng = np.random.default_rng(seed)
    safe = np.nan_to_num(matrix, nan=0.0)
    total_labels = safe.sum(axis=0)
    target_rows = len(smiles) / fold_count
    target_labels = total_labels / fold_count
    rows = np.zeros(fold_count, dtype=float)
    counts = np.zeros((fold_count, matrix.shape[1]), dtype=float)
    ordering = []
    for group, indices in grouped.items():
        label_counts = safe[indices].sum(axis=0)
        rarity = float((label_counts / np.maximum(total_labels, 1.0)).sum())
        ordering.append((group, indices, label_counts, rarity, float(rng.random())))
    ordering.sort(key=lambda item: (-len(item[1]), -item[3], item[4], item[0]))

    assigned: List[List[int]] = [[] for _ in range(fold_count)]
    for _, indices, label_counts, _, _ in ordering:
        candidate_scores = []
        for fold in range(fold_count):
            next_rows = rows.copy()
            next_counts = counts.copy()
            next_rows[fold] += len(indices)
            next_counts[fold] += label_counts
            row_error = float(np.mean(((next_rows - target_rows) / max(target_rows, 1.0)) ** 2))
            observed = total_labels > 0
            label_error = float(
                np.mean(
                    ((next_counts[:, observed] - target_labels[observed]) / np.maximum(target_labels[observed], 1.0)) ** 2
                )
            ) if observed.any() else 0.0
            candidate_scores.append(row_error + 1.5 * label_error)
        selected = int(np.argmin(candidate_scores))
        assigned[selected].extend(indices)
        rows[selected] += len(indices)
        counts[selected] += label_counts

    seen_connectivity: Dict[str, int] = {}
    for fold, indices in enumerate(assigned):
        for index in indices:
            key = connectivity_keys[index]
            if key in seen_connectivity and seen_connectivity[key] != fold:
                raise RuntimeError("Connectivity leakage detected across CV folds")
            seen_connectivity[key] = fold
    stable_folds = tuple(tuple(sorted(indices)) for indices in assigned)
    payload = {
        "folds": [list(fold) for fold in stable_folds],
        "group_ids": list(group_ids),
        "seed": seed,
        "similarity_threshold": similarity_threshold,
    }
    fold_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return FoldManifest(stable_folds, group_ids, seed, similarity_threshold, fold_hash)
