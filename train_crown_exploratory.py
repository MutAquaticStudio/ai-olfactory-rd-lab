#!/usr/bin/env python3
"""Explicitly non-production CROWN response-rate experiment with learning curves.

This separate entry point does NOT relax the reviewed Judge loader or its gates.
Targets are condition-specific observed yes fractions, not molecular consensus.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from olfactory.data_foundation.evidence import checked_file, content_hash, file_hash
from olfactory.features import create_morgan_tensor
from olfactory.models import OdorPredictor, select_device
from olfactory.resources import validate_resource_bundle
from olfactory.training.benchmark import assert_no_leakage
from olfactory.training.splits import chemical_group_calibrated_split, chemical_groups

ROOT = Path(__file__).resolve().parent


def condition_targets(frame, labels):
    if not set(frame.source_presence_state) <= {'PRESENT_AT_TRIAL', 'ABSENT_AT_TRIAL', 'UNASSESSED'}:
        raise ValueError('Unexpected source response state')
    if (frame.groupby('condition_id').isomeric_smiles.nunique() != 1).any():
        raise ValueError('A condition has multiple identities')
    if not set(frame.descriptor) <= set(labels):
        raise ValueError('Unknown mapped descriptor')
    units = frame[['condition_id', 'isomeric_smiles']].drop_duplicates().sort_values('condition_id').reset_index(drop=True)
    indices = {name: i for i, name in enumerate(units.condition_id)}
    columns = {name: i for i, name in enumerate(labels)}
    targets = np.full((len(units), len(labels)), np.nan, dtype=np.float32)
    counts = np.zeros_like(targets, dtype=np.int64)
    for (condition, term), rows in frame.groupby(['condition_id', 'descriptor']):
        observed = rows.source_presence_state.ne('UNASSESSED')
        n = int(observed.sum())
        if n:
            i, j = indices[condition], columns[term]
            targets[i, j] = rows.source_presence_state.eq('PRESENT_AT_TRIAL').sum() / n
            counts[i, j] = n
    return units, targets, counts


def prepare_data(source_dir, labels):
    source_dir = Path(source_dir)
    manifest = json.loads((source_dir / 'manifest.json').read_text())
    if manifest.get('status') != 'AUDIT_ONLY' or manifest.get('training_eligible') is not False:
        raise ValueError('This exploratory adapter expects the audit source, not a production release')
    for name, checksum in manifest['files'].items():
        checked_file(source_dir, name, checksum)
    policy = json.loads((source_dir / 'policy.json').read_text())
    if policy['encoding']['status'] != 'RESOLVED_BY_PAPER_AND_AUTHOR_CODE':
        raise ValueError('Source encoding unresolved')
    frame = pd.read_parquet(source_dir / 'source_responses.parquet')
    identities = {r['source_csv_row']: r for r in json.loads((source_dir / 'identity_review.json').read_text())}
    conditions = json.loads((source_dir / 'condition_review.json').read_text())
    links, exclusions = [], []
    keys = ['study', 'sampling_group', 'inclusion', 'odor_set', 'molcode']
    for condition in conditions:
        reason = None
        if condition['inclusion'] != '1':
            reason = 'SOURCE_EXCLUDED'
        elif condition['study'] != 'main' or condition['sampling_group'] not in ('home', 'lab'):
            reason = 'PATIENT_OR_RETEST_CONTEXT_HOLD'
        elif condition['context_status'] not in ('SET_MATCH_PROPOSED', 'ANCHOR_MATCH_PROPOSED'):
            reason = 'UNRESOLVED_CONDITION'
        elif condition['molcode'] in ('Tribut', 'Allycap'):
            reason = 'PUBLISHED_MATERIAL_OR_NAME_DISCREPANCY_HOLD'
        refs = condition['candidate_stimulus_csv_rows']
        if reason is None and len(refs) != 1:
            reason = 'AMBIGUOUS_STIMULUS'
        record = identities.get(refs[0]) if len(refs) == 1 else None
        if reason is None and (record is None or record['flags']):
            reason = 'STEREO_OR_IDENTITY_FLAG'
        if reason:
            exclusions.append({**{k: condition[k] for k in keys}, 'source_rows': condition['rows'], 'reason': reason})
            continue
        raw = record['raw_record']
        context = {**{k: condition[k] for k in keys}, 'stimulus_csv_row': refs[0],
            'source_concentration': raw.get('concentration_final'), 'source_volume': raw.get('volume_final')}
        links.append({**{k: condition[k] for k in keys},
            'condition_id': content_hash(context), 'context': json.dumps(context, sort_keys=True),
            'isomeric_smiles': record['identity']['isomeric_smiles']})
    if not links:
        raise ValueError('No unflagged source conditions available for exploration')
    link_frame = pd.DataFrame(links)
    selected = frame.merge(link_frame, on=keys, how='inner', validate='many_to_one')
    selected = selected[selected.descriptor.notna()].copy()
    # Deduplicate source keys is forbidden: duplicates mean the adapter is wrong.
    if selected.duplicated(['condition_id', 'assessor_id', 'descriptor']).any():
        raise ValueError('Duplicate participant/condition/descriptor')
    table, targets, counts = condition_targets(selected, labels)
    table = table.merge(link_frame[['condition_id', 'context']], on='condition_id', validate='one_to_one')
    usable = np.isfinite(targets).any(axis=1)
    table, targets, counts = table[usable].reset_index(drop=True), targets[usable], counts[usable]
    summary = {'source_response_slots': len(frame), 'selected_descriptor_slots': len(selected),
        'selected_trial_rows': int(selected.source_csv_row.nunique()),
        'condition_rows': len(table), 'unique_isomeric_smiles': int(table.isomeric_smiles.nunique()),
        'trained_descriptors': [label for j, label in enumerate(labels) if np.isfinite(targets[:, j]).any()],
        'excluded_source_rows_by_reason': dict(Counter({reason: sum(r['source_rows'] for r in exclusions if r['reason'] == reason)
            for reason in {r['reason'] for r in exclusions}})),
        'target_definition': 'Unweighted per-condition observed yes fraction; UNASSESSED masked; no consensus threshold',
        'identity_scope': 'Unflagged source SMILES only; proposed context links, not verified physical material identity'}
    return table, targets, counts, summary, exclusions


def masked_bce(logits, targets):
    mask = torch.isfinite(targets)
    if not mask.any():
        return logits.sum() * 0
    return nn.functional.binary_cross_entropy_with_logits(logits[mask], targets[mask])


def evaluate(model, features, targets, indices, device):
    model.eval()
    with torch.inference_mode():
        logits = model(features[list(indices)].to(device))
        target = targets[list(indices)].to(device)
        loss = masked_bce(logits, target).item()
        probabilities = torch.sigmoid(logits).cpu().numpy()
    probabilities[:, ~torch.isfinite(targets).any(dim=0).numpy()] = np.nan
    y = targets[list(indices)].numpy()
    mask = np.isfinite(y)
    if not mask.any():
        raise ValueError('Partition has no assessed responses')
    metrics = {'bce': float(loss), 'response_rate_mae': float(np.abs(probabilities[mask] - y[mask]).mean()),
               'condition_descriptor_cells': int(mask.sum())}
    return metrics, probabilities


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def train(args):
    if not args.acknowledge_exploratory:
        raise ValueError('Explicit --acknowledge-exploratory is required; this source failed production data gates')
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError('Epochs and patience must be positive')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment')
    bundle = validate_resource_bundle()
    protected = [bundle / name for name in ('odor_predictor_weights.pth', 'smiles_creator_weights.pth', 'odor_morgan_tensor_dataset.pt')]
    protected += [p for p in (ROOT / 'model_registry.json',) if p.is_file()]
    before = {str(p): file_hash(p) for p in protected}
    dataset = torch.load(bundle / 'odor_morgan_tensor_dataset.pt', map_location='cpu', weights_only=False)
    labels = list(dataset.label_names)
    del dataset
    if len(labels) != 113 or len(set(labels)) != 113:
        raise ValueError('Expected production label order with 113 unique entries')
    table, y, counts, summary, exclusions = prepare_data(args.source, labels)
    # Split uses observed positive and negative support, never a missing->negative cast.
    support = np.concatenate((np.isfinite(y) & (y > 0), np.isfinite(y) & (y < 1)), axis=1).astype(float)
    split = chemical_group_calibrated_split(table.isomeric_smiles.tolist(), support, seed=args.seed)
    payload = split.to_dict()
    assert_no_leakage(payload)
    partitions = {'train': split.train_indices, 'calibration': split.calibration_indices,
                  'validation': split.validation_indices, 'locked_test': split.test_indices}
    if any(not values for values in partitions.values()):
        raise ValueError('Chemical grouping produced an empty partition; do not weaken the split automatically')
    _, connectivity = chemical_groups(table.isomeric_smiles.tolist())
    owner = {}
    for name, indices in partitions.items():
        for i in indices:
            if connectivity[i] in owner and owner[connectivity[i]] != name:
                raise ValueError('Connectivity leakage')
            owner[connectivity[i]] = name
    features = torch.stack([create_morgan_tensor(Chem.MolFromSmiles(s)) for s in table.isomeric_smiles])
    targets = torch.from_numpy(y)
    device = select_device() if args.device == 'auto' else torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cpu':
        torch.use_deterministic_algorithms(True)
    model = OdorPredictor().to(device)  # Fresh initialization, no production weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_indices = list(split.train_indices)
    loader = DataLoader(TensorDataset(features[train_indices], targets[train_indices]), batch_size=64,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    output.mkdir(parents=True, exist_ok=False)
    table['partition'] = [next(name for name, ids in partitions.items() if i in ids) for i in range(len(table))]
    table['group_id'] = split.group_ids
    table['connectivity_key'] = connectivity
    table.to_parquet(output / 'conditions.parquet', index=False)
    np.savez_compressed(output / 'targets.npz', response_rates=y, assessed_counts=counts, features=features.numpy())
    save_json(output / 'split.json', payload)
    save_json(output / 'exclusions.json', exclusions)
    save_json(output / 'data_summary.json', summary)
    config = {'status': 'EXPLORATORY_NOT_FOR_PRODUCTION', 'architecture': '2048-1024-512-113; ReLU; dropout 0.3',
        'features': {'radius': 2, 'bits': 2048, 'useChirality': True}, 'seed': args.seed,
        'max_epochs': args.max_epochs, 'patience': args.patience, 'batch_size': 64, 'learning_rate': 0.001,
        'loss': 'Masked unweighted BCE on per-condition response rates', 'device': str(device),
        'label_names': labels, 'calibration': 'NOT_FITTED_INSUFFICIENT_MOLECULE_SUPPORT',
        'intensity': 'NOT_EVALUABLE_NO_DESCRIPTOR_INTENSITY', 'source': str(args.source),
        'source_manifest_sha256': file_hash(Path(args.source) / 'manifest.json'),
        'script_sha256': file_hash(Path(__file__)), 'production_checksums_before': before,
        'python': platform.python_version(), 'torch': torch.__version__, 'rdkit': rdBase.rdkitVersion,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())}
    save_json(output / 'config.json', config)
    print(json.dumps({'data': summary, 'partitions': {k: len(v) for k, v in partitions.items()}, 'device': str(device)}, indent=2), flush=True)
    history, best_state, best_loss, stale, best_epoch = [], None, float('inf'), 0, 0
    started = time.monotonic()
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(batch_x.to(device)), batch_y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
        training, _ = evaluate(model, features, targets, split.train_indices, device)
        validation, _ = evaluate(model, features, targets, split.validation_indices, device)
        row = {'epoch': epoch, 'train_loss': training['bce'], 'validation_loss': validation['bce'],
               'train_response_rate_mae': training['response_rate_mae'], 'validation_response_rate_mae': validation['response_rate_mae']}
        history.append(row)
        pd.DataFrame(history).to_csv(output / 'history.csv', index=False)
        if validation['bce'] < best_loss - 1e-6:
            best_loss, best_epoch, stale = validation['bce'], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, output / 'weights.tmp')
            os.replace(output / 'weights.tmp', output / 'weights.pth')
        else:
            stale += 1
        print(f'Epoch {epoch:03d} | train={training["bce"]:.6f} | validation={validation["bce"]:.6f} | best={best_epoch}', flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    # First locked-test access is after checkpoint selection is finished.
    metrics, predictions = {}, {}
    for name in ('train', 'validation', 'locked_test'):
        metrics[name], predictions[name] = evaluate(model, features, targets, partitions[name], device)
    mean = np.nansum(y[train_indices], axis=0) / np.maximum(np.isfinite(y[train_indices]).sum(axis=0), 1)
    test_y = y[list(split.test_indices)]
    mask = np.isfinite(test_y)
    p = np.broadcast_to(np.clip(mean, 1e-6, 1-1e-6), test_y.shape)
    metrics['train_mean_baseline_locked_test'] = {
        'bce': float(-(test_y[mask]*np.log(p[mask])+(1-test_y[mask])*np.log(1-p[mask])).mean()),
        'response_rate_mae': float(np.abs(p[mask]-test_y[mask]).mean())}
    np.savez_compressed(output / 'predictions.npz', **predictions)
    metrics.update({'best_epoch': best_epoch, 'epochs_completed': len(history),
        'early_stopped': len(history) < args.max_epochs, 'runtime_seconds': time.monotonic()-started,
        'locked_test_policy': 'Evaluated once after model selection; exploratory retrospective results',
        'calibration_partition_policy': 'Held out, not read for fitting or metrics',
        'promotion_eligible': False})
    save_json(output / 'metrics.json', metrics)
    save_json(output / 'history.json', history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    h = pd.DataFrame(history)
    for ax, names, title, ylabel in [(axes[0], ('train_loss', 'validation_loss'), 'Masked BCE', 'Loss'),
                                    (axes[1], ('train_response_rate_mae', 'validation_response_rate_mae'), 'Response-rate error', 'MAE')]:
        ax.plot(h.epoch, h[names[0]], color='#008080', label='Train (evaluation mode)')
        ax.plot(h.epoch, h[names[1]], color='#D97706', label='Validation')
        ax.axvline(best_epoch, color='#64748B', ls='--', label=f'Best epoch {best_epoch}')
        ax.set(xlabel='Epoch', ylabel=ylabel, title=title)
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle('CROWN exploratory Morgan MLP — not calibrated or production-validated')
    fig.savefig(output / 'learning_curve.png', dpi=180)
    plt.close(fig)
    after = {str(p): file_hash(p) for p in protected}
    if before != after:
        raise RuntimeError('Protected production files changed during experiment')
    save_json(output / 'manifest.json', {'status': 'EXPLORATORY_COMPLETE', 'promotion_eligible': False,
        'created_at': datetime.now(timezone.utc).isoformat(), 'production_unchanged': True,
        'files': {p.name: file_hash(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'output': str(output), 'metrics': metrics, 'production_unchanged': True}, indent=2), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acknowledge-exploratory', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    train(parser.parse_args())
