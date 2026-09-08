#!/usr/bin/env python3
"""Stage minimal CROWN sensory evidence. Does not approve numeric polarity or train."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from olfactory.data_foundation.evidence import checked_file, content_hash, file_hash, molecular_identity

DESCRIPTORS = ('sweet', 'sour', 'fruit', 'spices', 'bakery', 'garlic', 'fish', 'burnt',
               'decayed', 'grass', 'wood', 'chemical', 'flower', 'musky', 'sweaty', 'ammonia/urinous')
PROPOSED_MAPPING = dict(sweet='sweet', sour='sour', fruit='fruity', spices='spicy', garlic='garlic',
                        fish='fishy', burnt='burnt', grass='grassy', wood='woody', flower='floral')
IDENTIFIERS = ('inclusion', 'study', 'sampling_group', 'code', 'odor_set', 'molcode',
               'odor_order', 'days_storage', 'days_diff')


def normalize_for_review(data):
    """Data minimization and source-specific state, not molecular consensus."""
    data = data[list(IDENTIFIERS) + list(DESCRIPTORS)].copy()
    if data[list(IDENTIFIERS[:6])].isna().any().any():
        raise ValueError('Missing source identity/cohort fields')
    if not set(data.inclusion) <= {'0', '1'} or not set(data.study) <= {'main', 'retest'}:
        raise ValueError('Unknown inclusion/study code')
    if not set(data.sampling_group) <= {'home', 'lab', 'pat', 'test', 'retest'}:
        raise ValueError('Unknown cohort code')
    keys = ['study', 'sampling_group', 'code', 'molcode']
    if data.duplicated(keys).any():
        raise ValueError('Duplicate source participant/molecule/occasion')
    for term in DESCRIPTORS:
        if not set(data[term].dropna()) <= {'0', '1'}:
            raise ValueError(f'Unexpected binary source value: {term}')
    data['source_csv_row'] = [str(i + 2) for i in range(len(data))]
    data['assessor_id'] = data.code.map(lambda x: content_hash({'source': 'crown-15657278', 'code': x}))
    data = data.drop(columns='code')
    long = data.melt(id_vars=[c for c in data if c not in DESCRIPTORS], value_vars=list(DESCRIPTORS),
                     var_name='source_descriptor', value_name='raw_binary_code')
    long['source_response_status'] = long.raw_binary_code.map({'0': 'RECORDED_ZERO', '1': 'RECORDED_ONE'}).fillna('MISSING')
    long['proposed_descriptor'] = long.source_descriptor.map(PROPOSED_MAPPING)
    long['mapping_review'] = 'PENDING_SOURCE_SPECIFIC_REVIEW'
    long['presence_state'] = 'UNASSESSED'
    long['intensity'] = None
    long['evidence_review'] = 'UNREVIEWED'
    long['cohort_disposition'] = 'HEALTHY_SOURCE_INCLUDED_REVIEW_REQUIRED'
    long.loc[long.sampling_group.eq('pat'), 'cohort_disposition'] = 'SEPARATE_PATIENT_DOMAIN'
    long.loc[long.inclusion.eq('0'), 'cohort_disposition'] = 'EXCLUDED_BY_SOURCE'
    return long


def stage_source(raw_root, output):
    raw_root, output = Path(raw_root), Path(output)
    if output.exists():
        raise FileExistsError('Immutable CROWN review exists')
    manifest = json.loads((raw_root / 'source-manifest.json').read_text())
    config = manifest['source_config']
    if config['record_id'] != 15657278 or config['license_id'] != 'cc-by-4.0' or manifest['training_eligible'] is not False:
        raise ValueError('Unapproved source configuration')
    checked_file(raw_root, 'zenodo_metadata.json', manifest['metadata_sha256'])
    paths = {r['filename']: checked_file(raw_root, r['filename'], r['sha256']) for r in manifest['files']}
    data = pd.read_csv(paths['data.csv'], sep=';', dtype=str, usecols=list(IDENTIFIERS) + list(DESCRIPTORS))
    stimuli = pd.read_csv(paths['odors.csv'], sep=';', dtype=str).dropna(axis=1, how='all')
    long = normalize_for_review(data)
    normalized = []
    for row in stimuli.to_dict('records'):
        row = {k: None if pd.isna(v) else v for k, v in row.items()}
        try:
            identity = molecular_identity(row['SMILES'])
            row.update({'canonical_source_smiles': identity['isomeric_smiles'], 'source_inchikey': identity['inchikey'],
                        'stereo_state': identity['stereo_state'], 'identity_review': 'SOURCE_STRUCTURE_UNVERIFIED'})
        except (ValueError, TypeError):
            row.update({'canonical_source_smiles': None, 'source_inchikey': None,
                        'stereo_state': 'UNKNOWN', 'identity_review': 'INVALID_SOURCE_SMILES'})
        normalized.append(row)
    if set(data.molcode) - set(stimuli.molcode):
        raise ValueError('Ratings lack corresponding stimulus metadata')
    # No molcode-only join: reused PEA has different concentrations across sets.
    stats = []
    for (cohort, included, term), frame in long.groupby(['sampling_group', 'inclusion', 'source_descriptor']):
        stats.append({'sampling_group': cohort, 'source_inclusion': included, 'source_descriptor': term,
                      'source_zero': int(frame.raw_binary_code.eq('0').sum()),
                      'source_one': int(frame.raw_binary_code.eq('1').sum()),
                      'missing': int(frame.raw_binary_code.isna().sum()), 'molecule_codes': int(frame.molcode.nunique())})
    report = {'status': 'AUDIT_STAGING_NOT_TRAINING_SNAPSHOT', 'source_rows': len(data), 'response_slots': len(long),
              'stimulus_metadata_rows': len(stimuli), 'stimulus_metadata_codes': int(stimuli.molcode.nunique()),
              'observed_molecule_codes': int(data.molcode.nunique()),
              'metadata_only_codes': sorted(set(stimuli.molcode) - set(data.molcode)),
              'binary_descriptors': len(DESCRIPTORS), 'source_response_counts': long.source_response_status.value_counts().to_dict(),
              'cohorts': data.groupby(['study', 'sampling_group', 'inclusion']).agg(rows=('molcode', 'size'),
                   participants=('code', 'nunique'), molecules=('molcode', 'nunique')).reset_index().to_dict('records'),
              'source_semantics': 'Article and dictionary specify yes/no questions, with numeric 0/1 in CSV',
              'binary_polarity_review': 'PENDING_AUTHOR_CODE_CROSSCHECK; notebook retrieval timed out',
              'mapping_status': '10 broad/literal candidates proposed; MUSKY not mapped to perfumery musk',
              'training_eligible': False, 'calibration_eligible': False,
              'requirements_before_release': ['Verify 0/1 polarity with author code or explicit encoding documentation',
                    'Review source-specific mapping and German/English terms',
                    'Resolve stimulus metadata by cohort/set, not molcode alone; verify stereo/CID/CAS',
                    'Keep healthy main, patient, excluded and retest domains separate',
                    'Choose and validate condition-specific molecular consensus',
                    'Recompute support and leakage-resistant split; do not count assessors as molecules']}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.crown-review-', dir=output.parent))
    try:
        long.to_parquet(stage / 'source_responses.parquet', index=False)
        pd.DataFrame(normalized).to_parquet(stage / 'source_stimuli.parquet', index=False)
        for name, value in [('report.json', report), ('descriptor_cohort_counts.json', stats),
                            ('source_config.json', config)]:
            (stage / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        out_manifest = {'schema_version': 1, 'source_manifest_sha256': file_hash(raw_root / 'source-manifest.json'),
                        'source_directory': str(raw_root), 'training_eligible': False,
                        'files': {p.name: file_hash(p) for p in stage.iterdir()}, 'script_sha256': file_hash(Path(__file__))}
        (stage / 'manifest.json').write_text(json.dumps(out_manifest, indent=2, sort_keys=True))
        if output.exists():
            raise FileExistsError('Immutable CROWN review exists')
        stage.rename(output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_source(args.raw_dir, args.output), indent=2))
