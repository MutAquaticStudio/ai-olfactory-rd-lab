#!/usr/bin/env python3
"""Resolve source encoding/mapping and run the existing repeated-panel preflight.

Technical source review is not human evidence approval or a training release.
No network requests, model loading, registry changes or source overwrites.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from olfactory.data_foundation.evidence import checked_file, file_hash
from olfactory.training.reliability import krippendorff_alpha_nominal
from stage_crown_review import DESCRIPTORS

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / 'data/crown_descriptor_policy_v1.json'


def apply_source_policy(frame, policy):
    mappings = policy['mappings']
    if len(mappings) != 16 or {r['source'] for r in mappings} != set(DESCRIPTORS):
        raise ValueError('Policy must cover exactly 16 source descriptors')
    label_names = json.loads((ROOT / 'data/odor_taxonomy_mapping_v1_2.json').read_text())['labels']
    if any(r['target'] is not None and r['target'] not in label_names for r in mappings):
        raise ValueError('Mapping outside 113-output contract')
    if policy['encoding']['status'] != 'RESOLVED_BY_PAPER_AND_AUTHOR_CODE':
        raise ValueError('Unresolved source encoding')
    if (policy['encoding']['zero'], policy['encoding']['one'], policy['encoding']['missing']) != (
            'ABSENT_AT_TRIAL', 'PRESENT_AT_TRIAL', 'UNASSESSED'):
        raise ValueError('Encoding differs from reviewed author evidence')
    if not set(frame.source_descriptor) <= set(DESCRIPTORS) or not set(frame.raw_binary_code.dropna()) <= {'0', '1'}:
        raise ValueError('Unknown descriptor or numeric code')
    result = frame.copy()
    result['source_presence_state'] = result.raw_binary_code.map({'0': 'ABSENT_AT_TRIAL', '1': 'PRESENT_AT_TRIAL'}).fillna('UNASSESSED')
    result['descriptor'] = result.source_descriptor.map({r['source']: r['target'] for r in mappings})
    result['mapping_decision'] = result.source_descriptor.map({r['source']: r['decision'] for r in mappings})
    result['policy_version'] = policy['policy_version']
    result['mapping_review'] = 'TECHNICAL_SOURCE_HARMONIZATION'
    result['presence_state'] = 'UNASSESSED'
    result['intensity'] = None
    result['evidence_review'] = 'UNREVIEWED'
    result['training_eligible'] = False
    return result


def agreement_audit(frame):
    """Source-specific descriptive alpha; missing ratings excluded, not zero-filled.

Main cohorts/sets remain separate. Retest unit = molecule and occasion; paired
assessor counts are checked per molecule. Retest context/material review is
still required even if this necessary statistical gate passes.
    """
    data = frame[frame.inclusion.eq('1') & ~frame.sampling_group.eq('pat')].copy()
    data['domain'] = data.sampling_group
    data.loc[data.study.eq('retest'), 'domain'] = 'paired_retest'
    data['numeric'] = pd.to_numeric(data.raw_binary_code, errors='raise')
    results = []
    for (study, domain, odor_set, term), group in data.groupby(['study', 'domain', 'odor_set', 'source_descriptor'], sort=True):
        keys = ['molcode', 'sampling_group', 'assessor_id']
        if group.duplicated(keys).any():
            raise ValueError('Duplicate rating in agreement unit')
        matrix = group.pivot(index=['molcode', 'sampling_group'], columns='assessor_id', values='numeric').to_numpy(dtype=float)
        alpha = krippendorff_alpha_nominal(matrix)
        repeated_counts = []
        for _, molecule in group.groupby('molcode'):
            observed = molecule[molecule.numeric.notna()]
            visits = observed.groupby('assessor_id').sampling_group.nunique()
            repeated_counts.append(int((visits >= 2).sum()) if study == 'retest' else 0)
        minimum = min(repeated_counts, default=0)
        passes = study == 'retest' and minimum >= 8 and np.isfinite(alpha) and alpha >= 0.5
        results.append({'study': study, 'cohort': domain, 'odor_set': odor_set, 'source_descriptor': term,
            'source_molecule_codes': int(group.molcode.nunique()), 'rating_units': len(matrix),
            'assessors': int(group.assessor_id.nunique()), 'observed_ratings': int(group.numeric.notna().sum()),
            'missing_ratings': int(group.numeric.isna().sum()),
            'alpha': float(alpha) if np.isfinite(alpha) else None,
            'alpha_status': 'EVALUATED' if np.isfinite(alpha) else 'NOT_EVALUABLE',
            'assessors_with_two_sessions_minimum': minimum,
            'required_alpha': 0.5, 'required_paired_assessors': 8,
            'passes_repeated_panel_gate': bool(passes), 'training_eligible': False,
            'reason': ('Single-occasion main study' if study != 'retest' else
                       'Necessary statistical checks only; context and identity approval still required')})
    return results


def finalize(staging_dir, audit_dir, notebook_dir, output, policy_path=POLICY_PATH):
    staging_dir, audit_dir, notebook_dir, output = map(Path, (staging_dir, audit_dir, notebook_dir, output))
    if output.exists():
        raise FileExistsError('Immutable harmonized source already exists')
    parents = {}
    source_digests = []
    for directory in (staging_dir, audit_dir):
        m = json.loads((directory / 'manifest.json').read_text())
        if m['training_eligible'] is not False:
            raise ValueError('Expected audit-only source')
        source_digests.append(m['source_manifest_sha256'])
        for name, checksum in m['files'].items():
            checked_file(directory, name, checksum)
        parents[str(directory / 'manifest.json')] = file_hash(directory / 'manifest.json')
    if len(set(source_digests)) != 1:
        raise ValueError('Staging and context audit use different raw sources')
    source_report = json.loads((audit_dir / 'report.json').read_text())
    policy = json.loads(Path(policy_path).read_text())
    if policy['record_id'] != 15657278 or policy['training_eligible'] is not False:
        raise ValueError('Policy source/release mismatch')
    checked_file(notebook_dir, 'analyses.ipynb', policy['encoding']['notebook_sha256'])
    parents[str(notebook_dir / 'analyses.ipynb')] = policy['encoding']['notebook_sha256']
    frame = pd.read_parquet(staging_dir / 'source_responses.parquet')
    if len(frame) != source_report['source_rows'] * 16:
        raise ValueError('Staging and context audit row counts differ')
    normalized = apply_source_policy(frame, policy)
    agreement = agreement_audit(normalized)
    mapping = {r['source']: r['target'] for r in policy['mappings']}
    repeated = [r for r in agreement if r['cohort'] == 'paired_retest' and mapping[r['source_descriptor']] is not None]
    gate_count = sum(r['passes_repeated_panel_gate'] for r in repeated)
    # Never release merely because the statistical necessary condition is met.
    gate = {'status': 'BLOCKED_DATA_GATE', 'training_eligible': False,
        'encoding': policy['encoding']['status'], 'mapped_descriptors': sum(v is not None for v in mapping.values()),
        'source_only_descriptors': sum(v is None for v in mapping.values()),
        'repeated_mapped_descriptor_groups': len(repeated), 'repeated_groups_passing_statistics': gate_count,
        'alpha_threshold': 0.5, 'paired_assessors_threshold': 8,
        'source_presence_counts': normalized.source_presence_state.value_counts().to_dict(),
        'identity_flags': source_report['identity_flags'],
        'context_status_rows': source_report['context_status_rows'],
        'training_started': False, 'production_changed': False,
        'blockers': ['Main study lacks independent repeated sessions',
            'Retest statistical gates not sufficient for a release; see per-descriptor agreement',
            'Material identity, held contexts and human evidence review incomplete',
            'No eligible training/calibration/validation snapshot and split'],
        'prohibited_workarounds': ['Do not lower alpha/repeat thresholds silently',
            'Do not turn trial-level absence into molecule-level consensus',
            'Do not count assessors as independent molecules', 'Do not train from audit-only source']}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.crown-harmonized-', dir=output.parent))
    try:
        normalized.to_parquet(stage / 'source_responses.parquet', index=False)
        for name, payload in [('policy.json', policy), ('agreement.json', agreement), ('training_preflight.json', gate)]:
            (stage / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        pd.DataFrame(agreement).to_csv(stage / 'agreement.csv', index=False)
        for name in ('condition_review.json', 'identity_review.json'):
            shutil.copyfile(audit_dir / name, stage / name)
        out = {'schema_version': 1, 'dataset_version': output.name, 'status': 'AUDIT_ONLY',
            'created_at': datetime.now(timezone.utc).isoformat(), 'training_eligible': False,
            'parent_checksums': parents, 'policy_sha256': file_hash(Path(policy_path)),
            'script_sha256': file_hash(Path(__file__)), 'row_count': len(normalized),
            'files': {p.name: file_hash(p) for p in stage.iterdir()}}
        (stage / 'manifest.json').write_text(json.dumps(out, indent=2, sort_keys=True))
        if output.exists():
            raise FileExistsError('Immutable harmonized source already exists')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return gate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('staging-dir', 'audit-dir', 'notebook-dir', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.staging_dir, args.audit_dir, args.notebook_dir, args.output), indent=2))
