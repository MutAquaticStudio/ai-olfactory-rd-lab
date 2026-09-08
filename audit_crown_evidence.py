#!/usr/bin/env python3
"""Offline, immutable CROWN identity/context audit; never releases training labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd

from olfactory.data_foundation.evidence import checked_file, file_hash, molecular_identity
from stage_crown_review import DESCRIPTORS, PROPOSED_MAPPING, normalize_for_review


def cas_valid(value):
    """Syntax/check digit only, not CAS-to-molecule identity verification."""
    if not isinstance(value, str) or not re.fullmatch(r'[1-9][0-9]{1,6}-[0-9]{2}-[0-9]', value.strip()):
        return False
    body, digit = value.strip().rsplit('-', 1)
    return sum(i * int(c) for i, c in enumerate(reversed(body.replace('-', '')), 1)) % 10 == int(digit)


def audit_identities(stimuli):
    rows = []
    for index, source in enumerate(stimuli.to_dict('records'), 2):
        source = {k: None if pd.isna(v) else v for k, v in source.items()}
        flags = []
        raw = source.get('SMILES')
        identity = {}
        try:
            identity = molecular_identity(raw)
            if identity['stereo_state'] == 'UNRESOLVED':
                flags.append('UNRESOLVED_STEREO')
            name = source.get('name') or ''
            if re.search(r'±|\+/-|racem', name, re.I) and identity['stereo_state'] == 'DEFINED':
                flags.append('RACEMIC_NAME_WITH_SPECIFIED_STEREO')
            if identity['identity_review_required']:
                flags.append('FRAGMENT_CHARGE_OR_RADICAL')
        except (ValueError, TypeError):
            flags.append('INVALID_SMILES')
        valid_cas = cas_valid(source.get('cas'))
        if not valid_cas:
            flags.append('INVALID_CAS_CHECK_DIGIT_OR_FORMAT')
        rows.append({'source_csv_row': index, 'molcode': source.get('molcode'),
            'raw_record': source, 'raw_smiles': raw, 'identity': identity,
            'cas_checksum_valid': valid_cas, 'flags': flags,
            'identity_status': 'DATABASE_CROSSCHECK_REQUIRED', 'training_eligible': False})
    for field, flag in [('cas', 'CAS_MULTIPLE_ISOMERIC_STRUCTURES'), ('cid', 'CID_MULTIPLE_ISOMERIC_STRUCTURES')]:
        structures = {}
        for row in rows:
            value = row['raw_record'].get(field)
            iso = row['identity'].get('isomeric_smiles')
            if value and iso:
                structures.setdefault(value.strip(), set()).add(iso)
        for row in rows:
            value = row['raw_record'].get(field)
            if value and len(structures.get(value.strip(), set())) > 1:
                row['flags'].append(flag)
    for row in rows:
        if row['flags']:
            row['identity_status'] = 'REVIEW_REQUIRED'
    return rows


def condition_candidates(study, cohort, odor_set, code, stimuli):
    """Return source row references, never an approved join or inferred concentration."""
    matching = stimuli[stimuli.molcode.eq(code)]
    if study == 'retest':
        return [int(i) + 2 for i in matching.index], 'RETEST_CONTEXT_REVIEW'
    if cohort == 'pat':
        return [int(i) + 2 for i in matching.index], 'PATIENT_CONTEXT_REVIEW'
    exact = matching[matching.odor_group.eq(odor_set)]
    anchors = matching[matching.odor_group.eq('0')]
    candidates = pd.concat([exact, anchors]).drop_duplicates()
    # Count source records even when identical: a duplicate needs review.
    count = len(exact) + len(anchors)
    if count != 1:
        return [int(i) + 2 for i in pd.concat([exact, anchors]).index], ('NO_CONTEXT_MATCH' if count == 0 else 'AMBIGUOUS_CONTEXT')
    return [int(candidates.index[0]) + 2], ('SET_MATCH_PROPOSED' if len(exact) else 'ANCHOR_MATCH_PROPOSED')


def run_audit(raw_dir, output, notebook_dir=None):
    raw_dir, output = Path(raw_dir), Path(output)
    if output.exists():
        raise FileExistsError('Immutable CROWN audit already exists')
    manifest = json.loads((raw_dir / 'source-manifest.json').read_text())
    config = manifest['source_config']
    if config['record_id'] != 15657278 or config['license_id'] != 'cc-by-4.0' or manifest['training_eligible'] is not False:
        raise ValueError('Unapproved source')
    checked_file(raw_dir, 'zenodo_metadata.json', manifest['metadata_sha256'])
    files = {r['filename']: checked_file(raw_dir, r['filename'], r['sha256']) for r in manifest['files']}
    # Only use source fields needed for identity, cohort and sensory response audit.
    from stage_crown_review import IDENTIFIERS
    data = pd.read_csv(files['data.csv'], sep=';', dtype=str, usecols=list(IDENTIFIERS) + list(DESCRIPTORS))
    normalize_for_review(data)  # Enforce source code/cohort/duplicate invariants.
    stimuli = pd.read_csv(files['odors.csv'], sep=';', dtype=str).dropna(axis=1, how='all')
    identities = audit_identities(stimuli)
    conditions = []
    keys = ['study', 'sampling_group', 'inclusion', 'odor_set', 'molcode']
    for key, frame in data.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, key))
        refs, status = condition_candidates(record['study'], record['sampling_group'], record['odor_set'], record['molcode'], stimuli)
        conditions.append({**record, 'rows': len(frame), 'participants': int(frame.code.nunique()),
            'candidate_stimulus_csv_rows': refs, 'context_status': status,
            'training_eligible': False, 'descriptor_counts': {term: {
                'recorded_zero': int(frame[term].eq('0').sum()),
                'recorded_one': int(frame[term].eq('1').sum()),
                'missing': int(frame[term].isna().sum())} for term in DESCRIPTORS}})
    mappings = [{'source_descriptor': term, 'proposed_descriptor': PROPOSED_MAPPING.get(term),
        'status': 'PENDING_SOURCE_SPECIFIC_REVIEW' if term in PROPOSED_MAPPING else 'SOURCE_ONLY',
        'training_eligible': False} for term in DESCRIPTORS]
    notebook = {'status': 'NOT_INSPECTED', 'training_eligible': False}
    if notebook_dir:
        root = Path(notebook_dir)
        nm = json.loads((root / 'source-manifest.json').read_text())
        if nm['source_config']['record_id'] != 15657278 or nm['training_eligible'] is not False:
            raise ValueError('Notebook source mismatch')
        entry = next(r for r in nm['files'] if r['filename'] == 'analyses.ipynb')
        path = checked_file(root, entry['filename'], entry['sha256'])
        # Pin the manually inspected author code. Changed code requires a new audit.
        if file_hash(path) != 'f3bc594a944f6276aba68bce0bb2ec5700fffaa7535e55d34687840efb001c9c':
            raise ValueError('Unreviewed notebook version')
        cells = json.loads(path.read_text())['cells']
        notebook = {'status': 'STATIC_CODE_INSPECTED_NOT_EXECUTED', 'sha256': file_hash(path),
            'source_manifest_sha256': file_hash(root / 'source-manifest.json'),
            'cell_index_basis': 'zero_based',
            'cell_hashes': {str(i): hashlib.sha256(''.join(cells[i]['source']).encode()).hexdigest() for i in (6, 19, 21, 26)},
            'findings': ['Cell 6 separates healthy included main, patients, excluded and retest',
                'Cells 21/26 average binary source descriptor values without explicit yes/no recoding',
                'Supports but does not explicitly document zero=no and one=yes; needs encoding evidence'],
            'binary_polarity_status': 'INDIRECT_SUPPORT_NOT_EXPLICIT_ENCODING', 'training_eligible': False}
    report = {'status': 'AUDIT_ONLY', 'training_eligible': False, 'calibration_eligible': False,
        'source_rows': len(data), 'stimulus_records': len(identities), 'condition_groups': len(conditions),
        'source_molecule_codes': int(data.molcode.nunique()),
        'cas_checksum_valid_rows': sum(r['cas_checksum_valid'] for r in identities),
        'stereo_states': dict(Counter(r['identity'].get('stereo_state', 'INVALID') for r in identities)),
        'identity_flags': dict(Counter(f for r in identities for f in r['flags'])),
        'context_status_rows': {status: sum(c['rows'] for c in conditions if c['context_status'] == status)
            for status in sorted({c['context_status'] for c in conditions})},
        'notebook': notebook,
        'release_blockers': ['Explicit numeric polarity evidence', 'Source-specific descriptor mapping review',
            'External identity/material review including racemate and declared stereo',
            'Condition and consensus review; no participant-level OR aggregation',
            'Support audit and leakage-resistant split before training']}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.crown-audit-', dir=output.parent))
    try:
        for name, value in [('report.json', report), ('identity_review.json', identities),
                            ('condition_review.json', conditions), ('descriptor_review.json', mappings)]:
            (stage / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        out_manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'source_manifest_sha256': file_hash(raw_dir / 'source-manifest.json'),
            'script_sha256': file_hash(Path(__file__)), 'training_eligible': False,
            'files': {p.name: file_hash(p) for p in stage.iterdir()}}
        (stage / 'manifest.json').write_text(json.dumps(out_manifest, indent=2, sort_keys=True))
        if output.exists():
            raise FileExistsError('Immutable CROWN audit already exists')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--notebook-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.raw_dir, args.output, args.notebook_dir), indent=2))
