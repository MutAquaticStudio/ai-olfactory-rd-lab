#!/usr/bin/env python3
"""Local 113-label PU benchmark. Explicit opt-in; never promotes a model.

Example: .venv-training/bin/python train_pu_exploratory.py --acknowledge-exploratory
Resume:  same command with --output <existing directory> --resume
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
import torch
from torch import nn

from olfactory.data_foundation.evidence import file_hash
from olfactory.features import create_morgan_tensor
from olfactory.models import OdorPredictor
from olfactory.resources import validate_resource_bundle
from olfactory.training.benchmark import assert_no_leakage
from olfactory.training.splits import chemical_group_calibrated_split
from olfactory.training.pu_exploratory import (
    grouped_bootstrap, nnpu_risk, prepare_snapshot, prior_assumption,
    retrieval_metrics, retrieval_rows, save_checkpoint, write_json,
)

ROOT = Path(__file__).resolve().parent
PARTITIONS = {'train': 'train_indices', 'calibration': 'calibration_indices',
              'validation': 'validation_indices', 'locked_test': 'test_indices'}


def make_model(architecture):
    if architecture == 'linear':
        return nn.Linear(2048, 113)
    if architecture == 'mlp':
        return OdorPredictor()
    raise ValueError('Unknown architecture')


def choose_device():
    if torch.backends.mps.is_available():
        try:
            model = make_model('mlp').to('mps')
            z = model(torch.zeros(3, 2048, device='mps'))
            p = torch.ones(3, 113, dtype=torch.bool, device='mps')
            nnpu_risk(z, p, torch.full((113,), .2, device='mps'), p[0]).backward()
            torch.mps.synchronize()
            del model, z, p
            torch.mps.empty_cache()
            return torch.device('mps'), None
        except (RuntimeError, NotImplementedError) as error:
            return torch.device('cpu'), str(error)
    return torch.device('cpu'), None


def evaluate(model, x, p, indices, pi, active, device):
    model.eval()
    with torch.inference_mode():
        z = model(x[indices].to(device))
        loss = nnpu_risk(z, p[indices].to(device), pi, active).item()
        scores = z.sigmoid().cpu().numpy()
    scores[:, ~active.cpu().numpy()] = np.nan
    metrics = retrieval_metrics(scores, p[indices].numpy(), active.cpu().numpy())
    metrics['pu_risk'] = loss if (p[indices][:, active.cpu()].sum(0) > 0).any() else None
    return metrics, scores


def rng_state(device):
    return {'cpu': torch.get_rng_state(), 'mps': torch.mps.get_rng_state() if device.type == 'mps' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    if device.type == 'mps' and state['mps'] is not None:
        torch.mps.set_rng_state(state['mps'])


def train_one(directory, architecture, offset, seed, x, p, partitions, active_np,
              priors_np, device, deadline, max_epochs=100, patience=20):
    """Only train + validation are accessible to model selection.

    Checkpoints are atomic transactions containing optimizer, RNG, best weights
    and history. Resume repeats no completed epoch and never fits on test/cal.
    """
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == 'mps':
        torch.mps.manual_seed(seed)
    model = make_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    active = torch.tensor(active_np, device=device)
    pi = torch.tensor(priors_np, dtype=torch.float32, device=device)
    history, best, best_epoch, best_score, stale, epoch = [], None, 0, -1., 0, 0
    checkpoint = directory / 'checkpoint.pth'
    identity = {'architecture': architecture, 'prior_offset': offset, 'seed': seed,
                'max_epochs': max_epochs, 'patience': patience, 'device': str(device)}
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume configuration changed')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history, best = state['history'], state['best']
        epoch, best_epoch, best_score, stale = [state[k] for k in ('epoch', 'best_epoch', 'best_score', 'stale')]
        restore_rng(state['rng'], device)
    train_ids, validation_ids = partitions['train'], partitions['validation']
    tx, tp = x[train_ids].to(device), p[train_ids].to(device)
    status = 'COMPLETE'
    while epoch < max_epochs and stale < patience:
        # One complete molecular batch per epoch; no label-starved mini-batches.
        if time.monotonic() >= deadline:
            status = 'BUDGET_EXHAUSTED'
            break
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nnpu_risk(model(tx), tp, pi, active)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite PU training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        train_metrics, _ = evaluate(model, x, p, train_ids, pi, active, device)
        validation, _ = evaluate(model, x, p, validation_ids, pi, active, device)
        epoch += 1
        score = validation['recall_at_5']
        if score is None:
            raise ValueError('No validation positives in trained-label universe')
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        row = {'epoch': epoch, 'seconds': time.monotonic() - started,
               **{f'train_{k}': v for k, v in train_metrics.items()},
               **{f'validation_{k}': v for k, v in validation.items()}}
        history.append(row)
        state = {'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'rng': rng_state(device), 'history': history, 'best': best,
                 'epoch': epoch, 'best_epoch': best_epoch, 'best_score': best_score, 'stale': stale}
        save_checkpoint(checkpoint, state)
        print(f'{directory.name} epoch={epoch} PU={train_metrics["pu_risk"]:.4f}'
              f' validation R@5={score:.4f} best={best_score:.4f}', flush=True)
    if best is not None:
        save_checkpoint(directory / 'weights.pth', best)
    pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
    write_json(directory / 'history.json', history)
    result = {**identity, 'status': status, 'epochs': epoch, 'best_epoch': best_epoch,
              'best_validation_recall_at_5': best_score if best is not None else None,
              'promotion_eligible': False}
    write_json(directory / 'run.json', result)
    return result


def curves(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = pd.read_csv(directory / 'history.csv')
    if history.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(axes.flat,
            ('pu_risk', 'recall_at_5', 'recall_at_10', 'mean_reciprocal_rank'),
            ('Nonnegative PU risk', 'Observed-positive recall@5', 'Observed-positive recall@10', 'First known-positive reciprocal rank')):
        for partition, color in [('train', '#008080'), ('validation', '#cb7849')]:
            ax.plot(history.epoch, history[f'{partition}_{metric}'], label=partition, color=color)
        ax.set(title=title, xlabel='Epoch')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle(f'{directory.name} — exploratory scores, not calibrated probabilities')
    fig.tight_layout()
    fig.savefig(directory / 'learning_curve.png', dpi=160)
    plt.close(fig)


def protected_checksums():
    bundle = validate_resource_bundle()
    paths = [bundle / n for n in ('odor_predictor_weights.pth', 'smiles_creator_weights.pth', 'odor_morgan_tensor_dataset.pt')]
    paths += [p for p in (ROOT / 'model_registry.json',) if p.exists()]
    return {str(p): file_hash(p) for p in paths}, bundle


def verify_files(root, files):
    for name, checksum in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != checksum:
            raise ValueError(f'Artifact checksum mismatch: {name}')


def main(args):
    if not args.acknowledge_exploratory:
        raise ValueError('Required: --acknowledge-exploratory; audit data are NOT production eligible')
    if not 0 < args.hours <= 12 or not 1 <= args.max_epochs <= 100 or not 1 <= args.patience <= 20:
        raise ValueError('Budget must be <=12h, epochs <=100, patience <=20 and positive')
    started = time.monotonic()
    output = Path(args.output).resolve()
    protected, bundle = protected_checksums()
    dataset = torch.load(bundle / 'odor_morgan_tensor_dataset.pt', map_location='cpu', weights_only=False)
    labels = list(dataset.label_names)
    del dataset
    torch.set_num_threads(2)
    device, fallback = choose_device()
    if device.type == 'cpu':
        torch.use_deterministic_algorithms(True)
    source = Path(args.source).resolve()
    source_hash = file_hash(source / 'manifest.json')
    code_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in (
        Path(__file__), ROOT / 'olfactory/training/pu_exploratory.py', ROOT / 'olfactory/training/splits.py',
        ROOT / 'olfactory/models.py', ROOT / 'olfactory/features.py')}
    contract = {'source_manifest_sha256': source_hash, 'label_names': labels, 'hours': args.hours,
        'max_epochs': args.max_epochs, 'patience': args.patience, 'code_sha256': code_hashes,
        'device': str(device), 'torch': torch.__version__, 'rdkit': rdBase.rdkitVersion}
    # Revalidate source on resume too; never trust only a cached feature matrix.
    table, positive, exclusions, source_manifest = prepare_snapshot(source, labels)
    if output.exists():
        if not args.resume:
            raise FileExistsError('Experiment exists. Use --resume; never overwrite runs')
        config = json.loads((output / 'config.json').read_text())
        if config['contract'] != contract or config['protected_before'] != protected:
            raise ValueError('Resume contract or protected resources changed')
        verify_files(output, config['prepared_files'])
        state = json.loads((output / 'state.json').read_text())
        if state['status'] == 'COMPLETE':
            print(f'Already complete: {output}', flush=True)
            return
        # Include any unclosed invocation after a process crash in consumed time.
        consumed = state['consumed_seconds']
        if state.get('invocation_started_utc'):
            consumed += max(0., time.time() - state['invocation_started_utc'])
        payload = json.loads((output / 'split.json').read_text())
        arrays = np.load(output / 'prepared.npz')
        x = torch.from_numpy(arrays['features'])
        if not np.array_equal(arrays['positive_observed'], positive):
            raise ValueError('Resume positive membership changed')
    else:
        if args.resume:
            raise FileNotFoundError('Cannot resume missing experiment')
        output.mkdir(parents=True)
        split = chemical_group_calibrated_split(table.isomeric_smiles.tolist(), positive.astype(float), seed=42)
        payload = split.to_dict()
        x = torch.stack([create_morgan_tensor(Chem.MolFromSmiles(s)) for s in table.isomeric_smiles])
        np.savez_compressed(output / 'prepared.npz', features=x.numpy(), positive_observed=positive)
        table['group_id'] = split.group_ids
        table.to_parquet(output / 'molecules.parquet', index=False)
        write_json(output / 'split.json', payload)
        write_json(output / 'exclusions.json', exclusions)
        config = {'contract': contract, 'protected_before': protected, 'created_at': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'mps_fallback_reason': fallback,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'status': 'AUDIT_ONLY_PU_EXPLORATORY', 'promotion_eligible': False,
            'test_status': 'RETROSPECTIVE_EXPLORATORY', 'calibration': 'NOT_EVALUABLE', 'intensity': 'NOT_EVALUABLE',
            'batch_policy': 'Full training molecular batch; marginal includes known positives',
            'prior_policy': 'r + offset*(1-r), offsets [0,.05,.15], unverified catalog prior/SCAR assumptions',
            'selection': 'validation observed-positive recall@5; ties use declared schedule order',
            'features': {'radius': 2, 'bits': 2048, 'chirality': True}, 'learning_rate': .001,
            'prepared_files': {name: file_hash(output / name) for name in
                ('prepared.npz', 'molecules.parquet', 'split.json', 'exclusions.json')}}
        write_json(output / 'config.json', config)
        consumed = 0.
    assert_no_leakage(payload)
    partitions = {k: list(payload[v]) for k, v in PARTITIONS.items()}
    if any(not ids for ids in partitions.values()):
        raise ValueError('Empty chemical partition; refuse to weaken split')
    owners = {}
    for name, ids in partitions.items():
        for i in ids:
            old = owners.setdefault(table.iloc[i].connectivity_key, name)
            if old != name:
                raise ValueError('Connectivity leakage')
    train_observed = positive[partitions['train']]
    counts = train_observed.sum(0)
    # Independent connectivity support, not repeated source observations.
    independent = np.array([table.iloc[np.array(partitions['train'])[train_observed[:, j]]].connectivity_key.nunique()
                            for j in range(113)])
    active = (independent >= 10) & (counts < len(train_observed))
    if not active.any():
        raise ValueError('No labels eligible for PU training')
    rates = counts / len(train_observed)
    p = torch.from_numpy(positive)
    support = []
    for j, label in enumerate(labels):
        support.append({'descriptor': label, 'trained': bool(active[j]), 'train_connectivity_positive': int(independent[j]),
            **{f'{name}_positive': int(positive[ids, j].sum()) for name, ids in partitions.items()},
            'assessed_negative': 0, 'output_type': 'MODEL_SCORE' if active[j] else 'NOT_TRAINED',
            **{f'prior_{offset}': float(prior_assumption(rates, offset)[j]) for offset in (0., .05, .15)}})
    write_json(output / 'label_support.json', support)
    pd.DataFrame(support).to_csv(output / 'label_support.csv', index=False)
    data_summary = {'source_rows': int(pd.read_parquet(source / 'molecules.parquet').shape[0]),
        'excluded_stereo_rows': len(exclusions), 'unique_resolved_structures': len(table),
        'trained_labels': int(active.sum()), 'partitions': {k: len(v) for k, v in partitions.items()},
        'groups': len(set(payload['group_ids'])), 'connectivity_overlap': False,
        'butina_note': 'Grouping does not guarantee every cross-partition similarity is <0.60'}
    write_json(output / 'data_summary.json', data_summary)
    print(json.dumps(data_summary), flush=True)
    deadline = started + args.hours * 3600 - consumed
    # Reserve final reporting/evaluation time inside the same 12h budget.
    train_deadline = deadline - min(120., args.hours * 3600 * .1)
    write_json(output / 'state.json', {'status': 'RUNNING', 'consumed_seconds': consumed,
                                     'invocation_started_utc': time.time() - (time.monotonic() - started)})
    results = []
    schedule = [(architecture, offset, 42) for architecture in ('linear', 'mlp') for offset in (0., .05, .15)]
    try:
        for architecture, offset, seed in schedule:
            result = train_one(output / f'{architecture}-prior{offset:.2f}-s{seed}', architecture, offset, seed,
                x, p, partitions, active, prior_assumption(rates, offset), device, train_deadline,
                args.max_epochs, args.patience)
            results.append(result)
            if result['status'] != 'COMPLETE':
                break
        complete = [r for r in results if r['status'] == 'COMPLETE']
        if len(complete) == 6:
            winner = max(complete, key=lambda r: r['best_validation_recall_at_5'])
            for seed in (17, 23):
                r = train_one(output / f'{winner["architecture"]}-prior{winner["prior_offset"]:.2f}-s{seed}',
                    winner['architecture'], winner['prior_offset'], seed, x, p, partitions, active,
                    prior_assumption(rates, winner['prior_offset']), device, train_deadline, args.max_epochs, args.patience)
                results.append(r)
                if r['status'] != 'COMPLETE':
                    break
        # Freeze configuration selection BEFORE any retrospective test access.
        winner = max(complete, key=lambda r: r['best_validation_recall_at_5']) if complete else None
        write_json(output / 'selection.json', {'winner_seed42': winner, 'runs': results,
            'selection_partition': 'validation', 'test_status': 'RETROSPECTIVE_EXPLORATORY',
            'calibration_used': False, 'production_promoted': False})
        if time.monotonic() < deadline:
            report(output, results, winner, x, p, partitions, active, rates, payload, device, deadline)
        elapsed = consumed + time.monotonic() - started
        status = 'COMPLETE' if len(results) == 8 and all(r['status'] == 'COMPLETE' for r in results) and (output / 'metrics.json').exists() else 'BUDGET_EXHAUSTED'
        write_json(output / 'state.json', {'status': status, 'consumed_seconds': elapsed, 'invocation_started_utc': None})
    finally:
        after, _ = protected_checksums()
        write_json(output / 'production_integrity.json', {'before': protected, 'after': after, 'unchanged': protected == after})
        if protected != after:
            raise RuntimeError('Protected production resources changed during experiment')
    files = {str(f.relative_to(output)): file_hash(f) for f in output.rglob('*') if f.is_file() and f.name != 'manifest.json'}
    write_json(output / 'manifest.json', {'status': status, 'promotion_eligible': False,
        'dataset_version': source_manifest['dataset_version'], 'files': files})
    print(f'{status}: {output}', flush=True)


def report(output, results, winner, x, p, partitions, active, rates, payload, device, deadline):
    metrics, predictions = {}, {}
    for r in results:
        if time.monotonic() >= deadline:
            return
        directory = output / f'{r["architecture"]}-prior{r["prior_offset"]:.2f}-s{r["seed"]}'
        if r['best_epoch'] == 0:
            continue
        model = make_model(r['architecture']).to(device)
        model.load_state_dict(torch.load(directory / 'weights.pth', weights_only=True, map_location='cpu'))
        pi = torch.tensor(prior_assumption(rates, r['prior_offset']), dtype=torch.float32, device=device)
        mask = torch.tensor(active, device=device)
        run_metrics = {}
        for name in ('train', 'validation', 'locked_test'):
            m, scores = evaluate(model, x, p, partitions[name], pi, mask, device)
            run_metrics[name] = m
            if name == 'locked_test':
                predictions[directory.name] = scores
                np.savez_compressed(directory / 'test_scores.npz', model_scores=scores, row_indices=partitions[name])
        metrics[directory.name] = run_metrics
        write_json(directory / 'metrics.json', run_metrics)
        curves(directory)
    if winner is None:
        return
    key = f'{winner["architecture"]}-prior{winner["prior_offset"]:.2f}-s42'
    ids = partitions['locked_test']
    observed = p[ids].numpy()
    frequency = np.broadcast_to(rates, (len(ids), 113))
    scores = predictions[key]
    rows = retrieval_rows(scores, observed, active)
    per_label = []
    ranking = np.argsort(-np.nan_to_num(scores, nan=-np.inf), axis=1, kind='stable')
    for j, label in enumerate(json.loads((output / 'config.json').read_text())['contract']['label_names']):
        known = observed[:, j]
        per_label.append({'descriptor': label, 'test_known_positive': int(known.sum()), 'trained': bool(active[j]),
            'retrieved_at_5_fraction': float((ranking[known, :5] == j).any(1).mean()) if active[j] and known.any() else None,
            'evidence_status': 'EXPLORATORY' if known.sum() >= 10 and active[j] else 'LIMITED_SUPPORT'})
    pd.DataFrame(per_label).to_csv(output / 'per_label_retrieval.csv', index=False)
    write_json(output / 'per_label_retrieval.json', per_label)
    groups = np.array(payload['group_ids'])[ids]
    summary = {'runs': metrics, 'selected_configuration': key,
        'frequency_baseline': retrieval_metrics(frequency, observed, active),
        'selected_vs_frequency_group_bootstrap': grouped_bootstrap(scores, frequency, observed, active, groups),
        'metric_scope': 'Known-positive retrieval only; UNASSESSED are not negative; no calibrated probability claim',
        'calibration': 'NOT_EVALUABLE', 'intensity': 'NOT_EVALUABLE', 'production_promoted': False,
        'test_status': 'RETROSPECTIVE_EXPLORATORY', 'random_expected_recall_at_5': min(5, int(active.sum())) / int(active.sum())}
    linear = [r for r in results if r['architecture'] == 'linear' and r['status'] == 'COMPLETE']
    if linear:
        linear_winner = max(linear, key=lambda r: r['best_validation_recall_at_5'])
        linear_key = f'linear-prior{linear_winner["prior_offset"]:.2f}-s42'
        summary['selected_vs_linear_group_bootstrap'] = grouped_bootstrap(scores, predictions[linear_key], observed, active, groups)
    write_json(output / 'metrics.json', summary)
    # Keep the private report beside artifacts; no source data enter Git.
    test = metrics[key]['locked_test']
    baseline = summary['frequency_baseline']
    lines = ['# PU Judge exploratory benchmark', '', f'Selected by validation recall@5: `{key}`.', '',
        f'Test observed-positive recall@5: {test["recall_at_5"]:.4f}; frequency baseline: {baseline["recall_at_5"]:.4f}.',
        f'Test recall@10: {test["recall_at_10"]:.4f}; MRR: {test["mean_reciprocal_rank"]:.4f}.', '',
        'These are retrospective known-positive retrieval metrics, NOT sensory accuracy or calibrated probabilities.',
        'Prior/SCAR assumptions are unverified. No assessed negatives or intensity evidence were manufactured.',
        'Calibration partition was not fitted. Production resources and registry are unchanged.',
        'CROWN response-rate BCE/MAE and prior 254-output catalog metrics are different tasks and not directly comparable.', '',
        '## Runs', '', '| Run | Epochs | Best epoch | Validation R@5 | Test R@5 |', '|---|---:|---:|---:|---:|']
    for r in results:
        name = f'{r["architecture"]}-prior{r["prior_offset"]:.2f}-s{r["seed"]}'
        if name in metrics:
            lines.append(f'| {name} | {r["epochs"]} | {r["best_epoch"]} | {r["best_validation_recall_at_5"]:.4f} | {metrics[name]["locked_test"]["recall_at_5"]:.4f} |')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--acknowledge-exploratory', action='store_true')
    p.add_argument('--source', default=str(ROOT / 'artifacts/data/clean-master-113-reviewed-v1'))
    p.add_argument('--output', default=str(ROOT / 'artifacts/judge' / ('pu-113-' + datetime.now().strftime('%Y%m%dT%H%M%S'))))
    p.add_argument('--hours', type=float, default=12.)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    main(parser().parse_args())
