"""Positive/unlabeled exploration; never a production data-release adapter.

The marginal risk uses ALL training molecules (including labeled positives).
Unknown catalog entries are not assessed negatives. Class priors and SCAR are
unverified assumptions, so exported sigmoid values are scores, not calibrated
sensory probabilities. No training code is imported by the production API.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.nn import functional as F

from olfactory.data_foundation.evidence import checked_file, file_hash


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prior_assumption(rate, offset):
    rate = np.asarray(rate, dtype=float)
    if offset not in (0., .05, .15) or not np.isfinite(rate).all() or ((rate < 0) | (rate > 1)).any():
        raise ValueError('Invalid PU prior sensitivity configuration')
    return rate + offset * (1 - rate)


def nnpu_risk(logits, positive_observed, priors, active):
    """pi*E_P[softplus(-f)] + max(0, E_X[softplus(f)]-pi*E_P[softplus(f)]).

    Full molecular batches avoid undefined positive expectations for rare
    labels. Evaluation omits labels with no positives in that partition.
    This implements the nonnegative objective, not an ABSENT label conversion.
    """
    count = positive_observed.sum(dim=0)
    eligible = active & (count > 0)
    if not bool(eligible.any()):
        return logits.sum() * 0
    z = logits[:, eligible]
    mask = positive_observed[:, eligible].to(z.dtype)
    denominator = count[eligible].to(z.dtype)
    pi = priors[eligible]
    positive = (F.softplus(-z) * mask).sum(0) / denominator
    positive_as_negative = (F.softplus(z) * mask).sum(0) / denominator
    marginal = F.softplus(z).mean(0)
    return (pi * positive + (marginal - pi * positive_as_negative).clamp_min(0)).mean()


def retrieval_rows(scores, observed, active):
    """Per-molecule known-positive retrieval in the fixed trained-label universe."""
    scores, observed = np.asarray(scores), np.asarray(observed, bool)
    selected = np.flatnonzero(active)
    if not len(selected):
        raise ValueError('No trained labels')
    if not np.isfinite(scores[:, selected]).all():
        raise ValueError('Nonfinite model score')
    known = observed[:, selected]
    order = np.argsort(-scores[:, selected], axis=1, kind='stable')
    ranked = np.take_along_axis(known, order, axis=1)
    count = known.sum(1)
    valid = count > 0
    rr = np.full(len(scores), np.nan)
    rr[valid] = 1. / (ranked[valid].argmax(1) + 1)
    result = {'mean_reciprocal_rank': rr}
    for k in (5, 10):
        recall = np.full(len(scores), np.nan)
        recall[valid] = ranked[valid, :k].sum(1) / count[valid]
        result[f'recall_at_{k}'] = recall
        result[f'enrichment_at_{k}'] = recall / (min(k, len(selected)) / len(selected))
    return result


def retrieval_metrics(scores, observed, active):
    rows = retrieval_rows(scores, observed, active)
    result = {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in rows.items()}
    total = int(np.asarray(observed).sum())
    result.update(evaluated_rows=int(np.isfinite(rows['recall_at_5']).sum()),
                  trained_label_count=int(np.asarray(active).sum()),
                  observed_positive_coverage=float(observed[:, active].sum() / total) if total else None)
    return result


def grouped_bootstrap(scores, baseline, observed, active, groups, seed=42, repeats=1000):
    left = retrieval_rows(scores, observed, active)
    right = retrieval_rows(baseline, observed, active)
    group_values, codes = np.unique(groups, return_inverse=True)
    # Sufficient statistics make group bootstrap cheap and preserve whole groups.
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(group_values), (repeats, len(group_values)))
    report = {}
    for name, values in left.items():
        delta = values - right[name]
        valid = np.isfinite(delta)
        sums = np.bincount(codes[valid], weights=delta[valid], minlength=len(group_values))
        counts = np.bincount(codes[valid], minlength=len(group_values))
        denominators = counts[draws].sum(1)
        bootstrap = sums[draws].sum(1)[denominators > 0] / denominators[denominators > 0]
        report[name] = {'delta': float(delta[valid].mean()) if valid.any() else None,
                        'ci95': np.quantile(bootstrap, [.025, .975]).tolist() if len(bootstrap) else None}
    return report


def prepare_snapshot(source, labels):
    """Checksum-checked, stereo-resolved, deduplicated *experimental* view.

    Raw snapshot remains untouched. Boolean observations mean ONLY known
    positive membership; false is UNASSESSED, never an assessed negative.
    """
    source = Path(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest.get('training_eligible') is not False:
        raise ValueError('PU exploration expects the ineligible audit snapshot')
    label_hash = hashlib.sha256(json.dumps(list(labels), separators=(',', ':')).encode()).hexdigest()
    if len(labels) != 113 or len(set(labels)) != 113 or label_hash != manifest['production_label_order_sha256']:
        raise ValueError('Production label order mismatch')
    for name, checksum in manifest['files'].items():
        checked_file(source, name, checksum)
    if file_hash(Path(manifest['input_csv'])) != manifest['input_csv_sha256']:
        raise ValueError('Source CSV checksum mismatch')
    mapping = json.loads((source / 'mapping.json').read_text())
    if mapping['production_label_order_sha256'] != label_hash or mapping['mapping_version'] != manifest['mapping_version']:
        raise ValueError('Mapping contract mismatch')
    molecules = pd.read_parquet(source / 'molecules.parquet')
    assessments = pd.read_parquet(source / 'draft_assessments.parquet')
    if set(assessments.presence_state) - {'PRESENT', 'UNASSESSED'}:
        raise ValueError('PU catalog adapter does not pool assessed negatives')
    if set(assessments.descriptor) != set(labels) or assessments.duplicated(['source_row', 'descriptor']).any():
        raise ValueError('Invalid assessment label membership')
    if molecules.source_row.duplicated().any() or set(assessments.source_row) != set(molecules.source_row):
        raise ValueError('Invalid source row membership')
    if len(assessments) != len(molecules) * len(labels):
        raise ValueError('Missing assessment rows')
    pivot = assessments.pivot(index='source_row', columns='descriptor', values='presence_state').reindex(columns=labels)
    units, exclusions = {}, []
    for record in molecules.to_dict('records'):
        i = record['source_row']
        mol = Chem.MolFromSmiles(record['canonical_isomeric_smiles'])
        if mol is None:
            raise ValueError(f'Invalid snapshot structure: {i}')
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if canonical != record['canonical_isomeric_smiles'] or Chem.MolToInchiKey(mol) != record['inchikey']:
            raise ValueError(f'Snapshot identity mismatch: {i}')
        if any(str(s.specified) != 'Specified' for s in Chem.FindPotentialStereo(mol)):
            exclusions.append({'source_row': i, 'reason': 'UNRESOLVED_STEREO'})
            continue
        observed = pivot.loc[i].eq('PRESENT').to_numpy()
        mapped = set(json.loads(record['mapped_terms']))
        if mapped != {labels[j] for j in np.flatnonzero(observed)}:
            raise ValueError('Mapped terms and evidence disagree')
        item = units.setdefault(canonical, {'isomeric_smiles': canonical, 'connectivity_key': record['connectivity_key'],
            'source_rows': [], 'sources': set(), 'positive_observed': np.zeros(113, bool)})
        item['source_rows'].append(i)
        item['sources'].update(json.loads(record['sources']))
        item['positive_observed'] |= observed
    ordered = [units[k] for k in sorted(units)]
    positive = np.stack([r.pop('positive_observed') for r in ordered])
    for r in ordered:
        r['source_rows'] = json.dumps(r['source_rows'])
        r['sources'] = json.dumps(sorted(r['sources']))
    return pd.DataFrame(ordered), positive, exclusions, manifest
