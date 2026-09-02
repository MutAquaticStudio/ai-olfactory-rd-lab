"""Optional DeepChem graph Judge v2 adapter.

This module is intentionally not imported by the FastAPI application.  It is a
training-environment component: DeepChem provides the chemistry featurizer and
the small PyTorch message-passing model below keeps the dual-head/missing-label
contract explicit.  Production remains on the checked-in Morgan baseline
until the benchmark promotion gate passes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F

from ..prediction import PredictionBatch
from ..prediction_integrity import PredictionIdentity, reliability_state
from .dataset import MolecularTargetTable
from .calibration import CalibrationBundle
from .judge_v2 import effective_positive_weights, masked_multitask_loss
from .metrics import intensity_metrics, multilabel_metrics
from .benchmark import dataset_fingerprint
from .registry import sha256_file
from .splits import SplitManifest


DEEPCHEM_ARCHITECTURE: Dict[str, object] = {
    "featurizer": "MolGraphConvFeaturizer",
    "use_edges": True,
    "use_chirality": True,
    "hidden_size": 128,
    "message_layers": 3,
    "shared_hidden": 256,
    "dropout": 0.2,
    "presence_labels": 113,
    "intensity_labels": 113,
}


def _deepchem_featurizer_class():
    try:
        from deepchem.feat import MolGraphConvFeaturizer
    except ImportError as error:  # pragma: no cover - exercised in training env
        raise RuntimeError(
            "DeepChem is optional. Create the Python 3.11 training environment "
            "with requirements-deepchem.txt before running the graph benchmark."
        ) from error
    return MolGraphConvFeaturizer


def make_deepchem_featurizer():
    """Return the project-standard chiral graph featurizer."""
    cls = _deepchem_featurizer_class()
    return cls(use_edges=True, use_chirality=True)


def featurize_smiles(smiles: Sequence[str], featurizer=None) -> List[Any]:
    """Featurize SMILES with DeepChem, failing loudly on invalid chemistry."""
    current = featurizer or make_deepchem_featurizer()
    values = list(current.featurize(list(smiles)))
    if len(values) != len(smiles):
        raise RuntimeError("DeepChem returned a feature count different from the input")
    invalid = [index for index, graph in enumerate(values) if graph is None]
    if invalid:
        raise ValueError(f"DeepChem could not featurize rows: {invalid[:5]}")
    return values


def _graph_arrays(graph: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read DeepChem GraphData without depending on a specific DC version."""
    nodes = np.asarray(getattr(graph, "node_features"), dtype=np.float32)
    if nodes.ndim != 2:
        raise ValueError("GraphData.node_features must be a two-dimensional matrix")
    edge_index = np.asarray(getattr(graph, "edge_index", np.empty((2, 0))), dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    elif edge_index.ndim == 2 and edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("GraphData.edge_index must have shape (2, edges)")
    edge_features = getattr(graph, "edge_features", None)
    if edge_features is None:
        edge_features = np.zeros((edge_index.shape[1], 0), dtype=np.float32)
    edge_features = np.asarray(edge_features, dtype=np.float32)
    if edge_features.ndim == 1:
        edge_features = edge_features.reshape(-1, 1)
    if edge_features.shape[0] != edge_index.shape[1]:
        raise ValueError("GraphData edge features are not aligned with edge_index")
    return nodes, edge_index, edge_features


class GraphMessageEncoder(nn.Module):
    """Small deterministic edge-aware message passing encoder."""

    def __init__(self, node_features: int, edge_features: int, hidden_size: int = 128, layers: int = 3):
        super().__init__()
        if node_features <= 0:
            raise ValueError("node_features must be positive")
        self.node_projection = nn.Linear(node_features, hidden_size)
        self.edge_projection = nn.Linear(max(edge_features, 1), hidden_size)
        self.updates = nn.ModuleList(nn.Linear(hidden_size * 2, hidden_size) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(layers))
        self.hidden_size = hidden_size
        self.node_features = node_features
        self.edge_features = edge_features

    def forward(self, graphs: Sequence[Any]) -> torch.Tensor:
        device = self.node_projection.weight.device
        pooled: List[torch.Tensor] = []
        for graph in graphs:
            nodes, edge_index, edge_values = _graph_arrays(graph)
            node = torch.as_tensor(nodes, device=device, dtype=torch.float32)
            if node.shape[1] != self.node_features:
                raise ValueError("Graph node feature width does not match the trained encoder")
            state = F.relu(self.node_projection(node))
            if self.edge_features:
                edge = torch.as_tensor(edge_values, device=device, dtype=torch.float32)
                if edge.shape[1] != self.edge_features:
                    raise ValueError("Graph edge feature width does not match the trained encoder")
            else:
                edge = torch.zeros((edge_index.shape[1], 1), device=device)
            edge_state = self.edge_projection(edge)
            source = torch.as_tensor(edge_index[0], device=device, dtype=torch.long)
            target = torch.as_tensor(edge_index[1], device=device, dtype=torch.long)
            for update, norm in zip(self.updates, self.norms):
                messages = torch.zeros_like(state)
                if len(source):
                    messages.index_add_(0, target, state[source] + edge_state)
                    degree = torch.zeros((state.shape[0], 1), device=device)
                    degree.index_add_(0, target, torch.ones((len(target), 1), device=device))
                    messages = messages / degree.clamp_min(1.0)
                state = norm(F.relu(update(torch.cat([state, messages], dim=1))) + state)
            pooled.append(torch.cat([state.mean(dim=0), state.max(dim=0).values], dim=0))
        if not pooled:
            return torch.empty((0, self.hidden_size * 2), device=device)
        return torch.stack(pooled)


class DeepChemGraphJudge(nn.Module):
    """Shared graph representation with presence and conditional-intensity heads."""

    def __init__(self, node_features: int, edge_features: int, label_count: int = 113):
        super().__init__()
        if label_count != 113:
            raise ValueError("The project contract requires exactly 113 descriptors")
        self.label_count = label_count
        self.encoder = GraphMessageEncoder(
            node_features,
            edge_features,
            int(DEEPCHEM_ARCHITECTURE["hidden_size"]),
            int(DEEPCHEM_ARCHITECTURE["message_layers"]),
        )
        representation = int(DEEPCHEM_ARCHITECTURE["hidden_size"]) * 2
        self.shared = nn.Sequential(
            nn.Linear(representation, int(DEEPCHEM_ARCHITECTURE["shared_hidden"])),
            nn.ReLU(),
            nn.Dropout(float(DEEPCHEM_ARCHITECTURE["dropout"])),
        )
        shared = int(DEEPCHEM_ARCHITECTURE["shared_hidden"])
        self.presence_head = nn.Linear(shared, label_count)
        self.intensity_head = nn.Linear(shared, label_count)

    def forward(self, graphs: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(self.encoder(graphs))
        return self.presence_head(hidden), self.intensity_head(hidden)


# Short alias for notebooks and benchmark configuration files.
DeepChemJudge = DeepChemGraphJudge


def _graph_dimensions(graphs: Sequence[Any]) -> Tuple[int, int]:
    if not graphs:
        raise ValueError("At least one graph is required to infer feature dimensions")
    nodes, _, edges = _graph_arrays(graphs[0])
    return int(nodes.shape[1]), int(edges.shape[1])


@dataclass
class DeepChemJudgePredictor:
    """Inference adapter implementing the common MoleculePredictor contract."""

    model: DeepChemGraphJudge
    featurizer: Any
    label_names: Tuple[str, ...]
    identity: PredictionIdentity

    @classmethod
    def from_artifact(cls, run_dir: Path) -> "DeepChemJudgePredictor":
        """Load a graph candidate artifact without touching the model registry."""
        root = Path(run_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        weights_path = root / "weights.pth"
        expected_checksum = manifest.get("weights_sha256")
        if expected_checksum and sha256_file(weights_path) != expected_checksum:
            raise ValueError("DeepChem artifact weights checksum verification failed")
        weights = torch.load(weights_path, map_location="cpu", weights_only=False)
        labels = tuple(str(value) for value in weights["label_names"])
        model = DeepChemGraphJudge(int(weights["node_features"]), int(weights["edge_features"]), len(labels))
        model.load_state_dict(weights["state_dict"])
        model.eval()
        return cls(
            model=model,
            featurizer=make_deepchem_featurizer(),
            label_names=labels,
            identity=PredictionIdentity(
                model_version=str(manifest.get("model_version", manifest.get("run_id", "judge-deepchem"))),
                dataset_version=str(manifest.get("dataset_version", "unknown")),
                calibration_version=str(manifest.get("calibration_version", "uncalibrated")),
                model_status=str(manifest.get("status", "CANDIDATE")),
            ),
        )

    def predict(self, isomeric_smiles: Sequence[str]) -> PredictionBatch:
        graphs = featurize_smiles(isomeric_smiles, self.featurizer)
        self.model.eval()
        with torch.inference_mode():
            logits, intensity = self.model(graphs)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            values = intensity.cpu().numpy()
        shape = probabilities.shape
        return PredictionBatch(
            model_version=self.identity.model_version,
            dataset_version=self.identity.dataset_version,
            calibration_version=self.identity.calibration_version,
            presence_probability=probabilities,
            expected_intensity=values,
            ensemble_uncertainty=np.full(shape, np.nan, dtype=np.float32),
            training_similarity=np.full((len(graphs),), np.nan, dtype=np.float32),
            reliability_state=tuple(reliability_state(None) for _ in graphs),
            label_names=self.label_names,
        )


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _evaluate(model: DeepChemGraphJudge, graphs: Sequence[Any], targets: np.ndarray, indices: Sequence[int], label_count: int):
    model.eval()
    with torch.inference_mode():
        logits, intensity = model([graphs[index] for index in indices])
    probabilities = torch.sigmoid(logits).cpu().numpy()
    intensity_values = intensity.cpu().numpy()
    presence_targets = targets[list(indices), :label_count]
    intensity_targets = targets[list(indices), label_count:]
    return {
        "presence": multilabel_metrics(presence_targets, probabilities),
        "intensity": intensity_metrics(intensity_targets, intensity_values),
        "logits": logits.cpu().numpy(),
        "intensity_values": intensity_values,
    }


def train_deepchem_judge(
    table: MolecularTargetTable,
    split: SplitManifest,
    artifact_root: Path,
    *,
    dataset_version: str,
    seed: int = 42,
    intensity_weight: float = 0.3,
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 32,
) -> Dict[str, object]:
    """Train one graph run and write an immutable candidate artifact.

    No registry promotion occurs here.  A separate quality-gate step must
    compare this artifact with the locked Morgan baseline first.
    """
    if len(table.label_names) != 113:
        raise ValueError("DeepChem Judge requires exactly 113 odor descriptors")
    if intensity_weight not in (0.1, 0.3, 1.0):
        raise ValueError("intensity_weight must be one of 0.1, 0.3 or 1.0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    featurizer = make_deepchem_featurizer()
    graphs = featurize_smiles(table.smiles, featurizer)
    node_features, edge_features = _graph_dimensions(graphs)
    model = DeepChemGraphJudge(node_features, edge_features, len(table.label_names))
    targets = np.concatenate([table.presence, table.intensity], axis=1).astype(np.float32)
    train_indices, validation_indices, test_indices = map(list, (split.train_indices, split.validation_indices, split.test_indices))
    positive_weights = effective_positive_weights(
        torch.from_numpy(table.presence[train_indices]),
        torch.from_numpy(table.presence_mask[train_indices]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = np.random.default_rng(seed)
    best_state = None
    best_loss = float("inf")
    stale = 0
    completed = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = np.asarray(train_indices, dtype=int)
        generator.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size].tolist()
            logits, intensity = model([graphs[index] for index in batch])
            presence = torch.from_numpy(table.presence[batch])
            intensity_target = torch.from_numpy(table.intensity[batch])
            loss, _ = masked_multitask_loss(logits, intensity, presence, intensity_target, positive_weights, intensity_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_logits, validation_intensity = model([graphs[index] for index in validation_indices])
            validation_loss, _ = masked_multitask_loss(
                validation_logits,
                validation_intensity,
                torch.from_numpy(table.presence[validation_indices]),
                torch.from_numpy(table.intensity[validation_indices]),
                positive_weights,
                intensity_weight,
            )
        value = float(validation_loss.item())
        completed = epoch
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("DeepChem graph training produced no checkpoint")
    model.load_state_dict(best_state)
    validation = _evaluate(model, graphs, targets, validation_indices, len(table.label_names))
    locked_test = _evaluate(model, graphs, targets, test_indices, len(table.label_names))
    # Calibration parameters and per-label thresholds are fitted on validation
    # only.  The locked test remains untouched until this final report.
    calibration = CalibrationBundle.fit(
        validation["logits"],
        table.presence[validation_indices],
        table.label_names,
        minimum_support=50,
    )
    validation["presence"] = multilabel_metrics(
        table.presence[validation_indices],
        calibration.transform_logits(validation["logits"]),
    )
    locked_test["presence"] = multilabel_metrics(
        table.presence[test_indices],
        calibration.transform_logits(locked_test["logits"]),
    )

    root = Path(__file__).resolve().parents[2]
    run_id = f"judge-deepchem-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-s{seed}"
    run_dir = Path(artifact_root) / "judge" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    weights_path = run_dir / "weights.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "node_features": node_features,
            "edge_features": edge_features,
            "label_names": list(table.label_names),
            "architecture": DEEPCHEM_ARCHITECTURE,
        },
        weights_path,
    )
    (run_dir / "config.json").write_text(json.dumps({"architecture": DEEPCHEM_ARCHITECTURE, "seed": seed, "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience, "intensity_weight": intensity_weight}, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    calibration_path = run_dir / "calibration.json"
    calibration.save(calibration_path)
    metrics = {
        "validation": {key: value for key, value in validation.items() if key not in {"logits", "intensity_values"}},
        "locked_test": {key: value for key, value in locked_test.items() if key not in {"logits", "intensity_values"}},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    checksums = {
        name: sha256_file(run_dir / name)
        for name in ("weights.pth", "config.json", "split.json", "metrics.json", "calibration.json")
    }
    manifest: Dict[str, object] = {
        "run_id": run_id,
        "model_family": "judge-deepchem-graph",
        "model_version": run_id,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_fingerprint(table),
        "split_hash": split.split_hash,
        "seed": seed,
        "git_commit": _git_commit(root),
        "architecture": DEEPCHEM_ARCHITECTURE,
        "node_features": node_features,
        "edge_features": edge_features,
        "weights_path": str(weights_path),
        "weights_sha256": sha256_file(weights_path),
        "checksums": checksums,
        "calibration_version": calibration.calibration_version,
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE",
        "production_promoted": False,
        "deepchem_is_training_only": True,
        "epochs": completed,
        "best_validation_loss": best_loss,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_deepchem_predictor(run_dir: Path) -> DeepChemJudgePredictor:
    """Public loader used by offline evaluation notebooks and shadow mode."""
    return DeepChemJudgePredictor.from_artifact(run_dir)


__all__ = [
    "DEEPCHEM_ARCHITECTURE",
    "make_deepchem_featurizer",
    "featurize_smiles",
    "GraphMessageEncoder",
    "DeepChemGraphJudge",
    "DeepChemJudge",
    "DeepChemJudgePredictor",
    "train_deepchem_judge",
    "load_deepchem_predictor",
]
