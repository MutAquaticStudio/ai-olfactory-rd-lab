#!/usr/bin/env python3
"""Train and benchmark the conditional SELFIES Transformer candidate model."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from olfactory.training.creator_v2 import (
    BOS_TOKEN,
    END_TOKEN,
    PAD_TOKEN,
    ConditionalSELFIESTransformer,
    build_selfies_vocabulary,
    condition_vector,
    encode_selfies,
    evaluate_generated_smiles,
    sample_conditioned,
    target_alignment_benchmark,
    target_condition_vector,
)
from olfactory.training.dataset import load_versioned_snapshot
from olfactory.training.gates import creator_promotion_gate
from olfactory.training.splits import chemical_group_calibrated_split
from olfactory.training.tracking import log_manifest_to_mlflow
from olfactory.resources import validate_resource_bundle


ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = validate_resource_bundle()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--dataset-version")
    parser.add_argument("--pretrain-smiles", type=Path, help="Approved one-SMILES-per-line pretraining corpus")
    parser.add_argument("--pretrain-license-manifest", type=Path, help="JSON approval and checksum for the pretraining corpus")
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--target-descriptors", default="fruity")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--benchmark-samples", type=int, default=1000)
    parser.add_argument(
        "--target-score-benchmark",
        type=Path,
        help="NPZ with conditional and unconditional calibrated score arrays",
    )
    parser.add_argument(
        "--target-enrichment-ci-lower",
        type=float,
        help="Deprecated; manual promotion metrics are ignored",
    )
    parser.add_argument("--blind-panel-effect-ci-lower", type=float)
    parser.add_argument("--diversity-not-degraded", action="store_true")
    parser.add_argument("--ood-not-increased", action="store_true")
    parser.add_argument("--allow-pre-panel-data", action="store_true")
    return parser.parse_args()


def padded_sequences(encoded, pad_idx):
    maximum = max(len(sequence) for sequence in encoded)
    matrix = torch.full((len(encoded), maximum), pad_idx, dtype=torch.long)
    for row, sequence in enumerate(encoded):
        matrix[row, : len(sequence)] = torch.tensor(sequence)
    return matrix[:, :-1], matrix[:, 1:]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    legacy = torch.load(
        RESOURCE_DIR / "odor_morgan_tensor_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    labels = tuple(str(name) for name in legacy.label_names)
    table = load_versioned_snapshot(
        args.snapshot,
        labels,
        strict_panel_gate=not args.allow_pre_panel_data,
    )
    usable = [
        index
        for index, state in enumerate(table.stereo_state)
        if state != "UNRESOLVED" and np.isfinite(table.presence[index]).any()
    ]
    if len(usable) < 10:
        raise SystemExit("Too few stereo-resolved, assessed molecules passed the Creator v2 data gate.")
    smiles = [table.smiles[index] for index in usable]
    presence = table.presence[usable]
    intensity = table.intensity[usable]
    pretrain_manifest = None
    if args.pretrain_smiles:
        if args.pretrain_license_manifest is None:
            raise SystemExit("--pretrain-license-manifest is required with --pretrain-smiles")
        pretrain_manifest = json.loads(args.pretrain_license_manifest.read_text(encoding="utf-8"))
        corpus_sha256 = hashlib.sha256(args.pretrain_smiles.read_bytes()).hexdigest()
        if pretrain_manifest.get("approved_for_training") is not True:
            raise SystemExit("The pretraining corpus is not approved for training.")
        if pretrain_manifest.get("corpus_sha256") != corpus_sha256:
            raise SystemExit("The pretraining corpus checksum does not match its license manifest.")
        approved = []
        for line in args.pretrain_smiles.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and Chem.MolFromSmiles(value) is not None:
                approved.append(value)
    else:
        approved = []
    vocabulary = build_selfies_vocabulary([*smiles, *approved])
    tokens = vocabulary["tokens"]
    token_to_idx = vocabulary["token_to_idx"]
    encoded = [encode_selfies(value, token_to_idx) for value in smiles]
    input_ids, target_ids = padded_sequences(encoded, token_to_idx[PAD_TOKEN])
    conditions = []
    for row, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        conditions.append(condition_vector(presence[row], intensity[row], molecule))
    condition_tensor = torch.from_numpy(np.stack(conditions)).float()
    split = chemical_group_calibrated_split(
        smiles,
        np.nan_to_num(presence, nan=0.0),
        seed=args.seed,
    )

    def loader(indices, shuffle):
        dataset = TensorDataset(input_ids[list(indices)], target_ids[list(indices)], condition_tensor[list(indices)])
        return DataLoader(dataset, batch_size=64, shuffle=shuffle)

    train_loader = loader(split.train_indices, True)
    validation_loader = loader(split.validation_indices, False)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = ConditionalSELFIESTransformer(len(tokens)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_idx[PAD_TOKEN])

    if approved:
        pretrain_encoded = [encode_selfies(value, token_to_idx) for value in approved]
        retained = [
            (value, sequence)
            for value, sequence in zip(approved, pretrain_encoded)
            if len(sequence) <= model.max_length
        ]
        if not retained:
            raise SystemExit("No approved pretraining sequence fits the configured maximum length.")
        pretrain_inputs, pretrain_targets = padded_sequences(
            [sequence for _, sequence in retained],
            token_to_idx[PAD_TOKEN],
        )
        zero_presence = np.zeros(len(labels), dtype=np.float32)
        zero_intensity = np.zeros(len(labels), dtype=np.float32)
        pretrain_conditions = torch.from_numpy(
            np.stack(
                [
                    condition_vector(
                        zero_presence,
                        zero_intensity,
                        Chem.MolFromSmiles(value),
                        presence_mask=np.zeros(len(labels), dtype=np.float32),
                        intensity_mask=np.zeros(len(labels), dtype=np.float32),
                    )
                    for value, _ in retained
                ]
            )
        ).float()
        pretrain_loader = DataLoader(
            TensorDataset(pretrain_inputs, pretrain_targets, pretrain_conditions),
            batch_size=64,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        for epoch in range(1, args.pretrain_epochs + 1):
            model.train()
            losses = []
            for inputs, targets, batch_conditions in pretrain_loader:
                optimizer.zero_grad(set_to_none=True)
                output = model(inputs.to(device), batch_conditions.to(device))
                loss = criterion(output.reshape(-1, len(tokens)), targets.to(device).reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            print(f"Pretrain {epoch:03d} | Loss {np.mean(losses):.5f}")
    best_loss = float("inf")
    best_state = None
    stale = 0
    training_history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for inputs, targets, batch_conditions in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device), batch_conditions.to(device))
            loss = criterion(logits.reshape(-1, len(tokens)), targets.to(device).reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for inputs, targets, batch_conditions in validation_loader:
                logits = model(inputs.to(device), batch_conditions.to(device))
                loss = criterion(logits.reshape(-1, len(tokens)), targets.to(device).reshape(-1))
                validation_losses.append(float(loss.item()))
        validation_loss = float(np.mean(validation_losses))
        train_loss = float(np.mean(train_losses))
        training_history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        print(f"Epoch {epoch:03d} | Train {train_loss:.5f} | Validation {validation_loss:.5f}")
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    if best_state is None:
        raise RuntimeError("Creator v2 did not produce a checkpoint")
    model.load_state_dict(best_state)

    selected_targets = [
        value.strip()
        for value in args.target_descriptors.split(",")
        if value.strip()
    ]
    condition = torch.from_numpy(
        target_condition_vector(labels, selected_targets)
    ).to(device)
    generator = None
    if device.type in {"cpu", "cuda"}:
        generator = torch.Generator(device=device.type).manual_seed(args.seed)
    generated = [
        sample_conditioned(model, condition, tokens, temperature=0.8, generator=generator)
        for _ in range(args.benchmark_samples)
    ]
    benchmark = evaluate_generated_smiles(generated, smiles)
    if args.target_score_benchmark:
        score_arrays = np.load(args.target_score_benchmark)
        alignment = target_alignment_benchmark(
            score_arrays["conditional"],
            score_arrays["unconditional"],
            seed=args.seed,
        )
        benchmark.update(alignment)
        target_enrichment_ci_lower = alignment["target_enrichment_ci_lower"]
    else:
        target_enrichment_ci_lower = None
    promotion = creator_promotion_gate(
        benchmark,
        target_enrichment_ci_lower=target_enrichment_ci_lower,
        diversity_not_degraded=args.diversity_not_degraded,
        ood_not_increased=args.ood_not_increased,
        blind_panel_effect_ci_lower=args.blind_panel_effect_ci_lower,
    )
    run_id = f"creator-v2-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-s{args.seed}"
    run_dir = args.artifact_root / "creator" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    history_path = run_dir / "learning_history.json"
    history_path.write_text(
        json.dumps(training_history, indent=2),
        encoding="utf-8",
    )
    history_csv_path = run_dir / "learning_history.csv"
    with history_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "validation_loss"))
        writer.writeheader()
        writer.writerows(training_history)
    curve_path = run_dir / "learning_curve.png"
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        [item["epoch"] for item in training_history],
        [item["train_loss"] for item in training_history],
        label="Train loss",
    )
    axis.plot(
        [item["epoch"] for item in training_history],
        [item["validation_loss"] for item in training_history],
        label="Validation loss",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Conditional SELFIES Transformer learning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)
    weights_path = run_dir / "creator_v2_weights.pth"
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "tokens": tokens,
            "label_names": labels,
            "condition_size": 455,
            "condition_schema": "presence+assessed_mask+intensity+measured_mask+properties-v2",
            "architecture": {"layers": 6, "d_model": 256, "heads": 8},
        },
        weights_path,
    )
    (run_dir / "selfies_vocab.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    split_path = run_dir / "split.json"
    split_path.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "model_version": run_id,
        "dataset_version": args.dataset_version or args.snapshot.stem,
        "split_hash": split.split_hash,
        "seed": args.seed,
        "validation_loss": best_loss,
        "learning_curve_path": str(curve_path),
        "learning_curve_sha256": hashlib.sha256(curve_path.read_bytes()).hexdigest(),
        "learning_history_path": str(history_path),
        "learning_history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "learning_history_csv_path": str(history_csv_path),
        "learning_history_csv_sha256": hashlib.sha256(history_csv_path.read_bytes()).hexdigest(),
        "split_path": str(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "pretraining": {
            "used": bool(approved),
            "approved_molecules": len(approved),
            "epochs": args.pretrain_epochs if approved else 0,
            "corpus_sha256": pretrain_manifest.get("corpus_sha256") if pretrain_manifest else None,
            "license_approval_ticket": pretrain_manifest.get("license_approval_ticket") if pretrain_manifest else None,
        },
        "benchmark": benchmark,
        "target_enrichment_ci_lower": target_enrichment_ci_lower,
        "manual_target_enrichment_ignored": args.target_enrichment_ci_lower,
        "promotion_gate": promotion.to_dict(),
        "promotion_eligible": promotion.eligible,
        "prospective_panel_status": "PASSED" if args.blind_panel_effect_ci_lower and args.blind_panel_effect_ci_lower > 0 else "REQUIRED",
        "weights_path": str(weights_path),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "status": "CANDIDATE",
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["mlflow_run_id"] = log_manifest_to_mlflow(
        manifest,
        manifest_path,
        args.artifact_root / "tracking",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
