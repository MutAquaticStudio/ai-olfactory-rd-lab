"""Freeze source harmonization and database-reference decisions, not sample approval.

No network, model, calibration or training imports. Historical snapshots remain
unchanged; a derived audit snapshot carries every decision and its evidence hash.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .evidence import checked_file, content_hash, file_hash, molecular_identity
from .keller_trials import DESCRIPTORS
from .repository import utc_now

TARGETS = {'SWEET': 'sweet', 'FRUIT': 'fruity', 'FISH': 'fishy', 'GARLIC': 'garlic',
           'SPICES': 'spicy', 'SOUR': 'sour', 'BURNT': 'burnt', 'WARM': 'warm',
           'WOOD': 'woody', 'GRASS': 'grassy', 'FLOWER': 'floral'}
HOLD = {
    'STEREO_OR_MATERIAL_REVIEW_REQUIRED': 'HOLD_STEREO_OR_MATERIAL',
    'IDENTITY_CONFLICT_REVIEW_REQUIRED': 'HOLD_IDENTITY_CONFLICT',
    'IDENTITY_UNVERIFIED_REVIEW_REQUIRED': 'HOLD_AMBIGUOUS_IDENTITY',
}


def validate_policy(policy, labels):
    mappings = policy['mappings']
    if (len(mappings) != 20 or {x['source'] for x in mappings} != set(DESCRIPTORS)
            or any(x['target'] != TARGETS.get(x['source']) for x in mappings)
            or not set(TARGETS.values()) <= set(labels)
            or policy.get('policy_version') != 'keller-source-harmonization-locked-v1'
            or policy.get('authorization') != 'USER_REQUESTED_IDENTITY_AND_MAPPING_FINALIZATION'):
        raise ValueError('Keller mapping/authorization does not match locked policy')
    for field in ('human_expert_approval', 'sample_identity_approved', 'training_eligible', 'calibration_eligible'):
        if policy.get(field) is not False:
            raise ValueError('Harmonization cannot approve samples or training')
    rating = policy['rating_policy']
    if (any(rating.get(k) is not False for k in ('negative_labels_authorized', 'molecule_consensus_authorized', 'pool_concentrations'))
            or rating.get('blank') != 'UNASSESSED' or rating.get('zero') != 'UNASSESSED'
            or rating.get('training_presence_state') != 'UNASSESSED'
            or rating.get('training_intensity') is not None):
        raise ValueError('Harmonization cannot release sensory targets')


def annotate_ratings(frame, policy):
    """Change only harmonization fields; never infer training targets or stereo."""
    if not frame.presence_state.eq('UNASSESSED').all() or not frame.intensity.isna().all():
        raise ValueError('Expected audit-only, unassessed Keller input')
    result = frame.copy()
    mapping = {item['source']: item for item in policy['mappings']}
    result['descriptor_before_review'] = frame.descriptor
    result['mapping_state_before_review'] = frame.mapping_state
    result['descriptor'] = frame.source_term.map({k: v['target'] for k, v in mapping.items()})
    result['mapping_state'] = frame.source_term.map({k: 'LOCKED_' + v['decision'] for k, v in mapping.items()}).fillna('NOT_DESCRIPTOR')
    result['mapping_policy_version'] = policy['policy_version']
    result['source_applicability_state'] = 'NOT_DESCRIPTOR'
    descriptor = frame.source_term.isin(mapping)
    result.loc[descriptor, 'source_applicability_state'] = 'UNASSESSED'
    numeric = pd.to_numeric(frame.raw_numeric_value, errors='coerce')
    recorded = frame.raw_value.notna()
    invalid = descriptor & recorded & (numeric.isna() | ~numeric.between(0, 100))
    detected = frame.detection_response.eq('I smell something')
    result.loc[descriptor & numeric.gt(0) & numeric.le(100) & detected, 'source_applicability_state'] = 'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY'
    result.loc[descriptor & recorded & ~detected, 'source_applicability_state'] = 'CONFLICT_REVIEW_REQUIRED'
    result.loc[invalid, 'source_applicability_state'] = 'INVALID_REVIEW_REQUIRED'
    return result


def finalize_identity(record, correction):
    status = record['status']
    decision = HOLD.get(status)
    structure = None
    effective_cas = record['source_cas']
    correction_id = None
    if status == 'DATABASE_STRUCTURE_CORROBORATED':
        reference = record['pubchem_identity']
        identity = molecular_identity(reference['isomeric_smiles'])
        if (identity['inchikey'] != reference['inchikey'] or identity['identity_review_required']
                or identity['stereo_state'] == 'UNRESOLVED' or record['flags']
                or not record['source_cas_exact_synonym'] or not record['source_cas_checksum_valid']
                or record['cas_lookup_cids'] != [record['returned_cid']]
                or record['queried_cid'] != record['returned_cid']):
            raise ValueError('Database reference failed independent identity checks')
        decision = 'ACCEPT_DATABASE_REFERENCE_ONLY'
        structure = identity
    elif status == 'IDENTIFIER_CORRECTION_PROPOSED':
        decision = 'HOLD_IDENTIFIER_CORRECTION'
        if correction is not None:
            if (record['source_cas'] != correction['old_cas']
                    or record['raw_source_identifier'] != correction['source_identifier']
                    or set(record['source_names']) != set(correction['source_names'])):
                raise ValueError('CAS correction does not match review identity')
            decision = 'CAS_CORRECTED_IDENTITY_REVIEW_REQUIRED'
            effective_cas = correction['new_cas']
            correction_id = correction['correction_id']
    if decision is None:
        raise ValueError('Unknown identity decision')
    return {
        'review_id': content_hash({'cas': record['source_cas'], 'identifier': record['raw_source_identifier']}),
        'raw_source_cas': record['source_cas'], 'effective_cas': effective_cas,
        'raw_source_identifier': record['raw_source_identifier'], 'source_names': record['source_names'],
        'measurement_rows': record['measurement_rows'], 'decision': decision,
        'flags': record['flags'], 'accepted_reference_structure': structure,
        'evidence_record_sha256': content_hash(record), 'record_url': record.get('record_url'),
        'correction_id': correction_id, 'sample_identity_approved': False,
        'training_eligible': False,
    }


def finalize_review(service, *, parent_path, evidence_dir, policy_path, dataset_version):
    """Publish an append-only review snapshot in the existing Data Foundation."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', dataset_version):
        raise ValueError('Invalid dataset version')
    target = service.snapshot_root / dataset_version
    if target.exists():
        raise FileExistsError('Immutable review snapshot exists')
    parent_path, evidence_dir, policy_path = map(Path, (parent_path, evidence_dir, policy_path))
    parent = json.loads(parent_path.read_text())
    unsigned = {k: v for k, v in parent.items() if k != 'manifest_sha256'}
    if (parent.get('manifest_sha256') != content_hash(unsigned)
            or parent.get('status') != 'AUDIT_ONLY' or parent.get('training_eligible') is not False
            or parent.get('label_names') != list(service.label_names)
            or 'harmonization_review' in parent):
        raise ValueError('Parent snapshot/config mismatch')
    for name, sha in parent['files'].items():
        checked_file(parent_path.parent, name, sha)
    policy = json.loads(policy_path.read_text())
    validate_policy(policy, service.label_names)
    evidence_manifest = json.loads((evidence_dir / 'manifest.json').read_text())
    if (evidence_manifest.get('training_eligible') is not False
            or evidence_manifest['input_sha256']['original_workbook'] != parent['original_workbook']['sha256']):
        raise ValueError('Identity evidence source mismatch')
    for name, sha in evidence_manifest['artifacts_sha256'].items():
        checked_file(evidence_dir, name, sha)
    for name, sha in evidence_manifest['response_files_sha256'].items():
        checked_file(evidence_dir / 'responses', name, sha)
    evidence = json.loads((evidence_dir / 'identity_evidence.json').read_text())
    if len(evidence) != evidence_manifest['source_rows']:
        raise ValueError('Identity evidence coverage mismatch')
    decisions = [finalize_identity(r, parent.get('cas_correction')) for r in evidence]
    by_key = {(r['effective_cas'], r['raw_source_identifier']): r for r in decisions}
    if len(by_key) != len(decisions):
        raise ValueError('Duplicate review identity')
    # The review is exactly the unresolved/missing-identity queue, not all molecules.
    queue = pd.read_csv(parent_path.parent / 'structure_review_queue.csv', dtype=str)
    queue_keys = set(zip(queue.source_cas, queue.raw_source_identifier))
    if queue_keys != set(by_key):
        raise ValueError('Identity evidence does not cover the parent review queue')
    stage = Path(tempfile.mkdtemp(prefix='.keller-review-', dir=service.snapshot_root))
    writer = None
    counts, seen = Counter(), Counter()
    try:
        for batch in pq.ParquetFile(parent_path.parent / 'ratings.parquet').iter_batches(batch_size=10000):
            frame = annotate_ratings(batch.to_pandas(), policy)
            keys = list(zip(frame.source_cas, frame.raw_source_identifier))
            frame['identity_review_id'] = [by_key[k]['review_id'] if k in by_key else None for k in keys]
            frame['reference_identity_decision'] = [by_key[k]['decision'] if k in by_key else 'NOT_IN_REVIEW_SCOPE' for k in keys]
            seen.update(k for k in keys if k in by_key)
            counts['measurement_rows'] += len(frame)
            counts['harmonized_descriptor_rows'] += int(frame.descriptor.notna().sum())
            counts['harmonized_positive_trial_observations'] += int((frame.descriptor.notna() & frame.source_applicability_state.eq('DESCRIPTOR_APPLIED_AT_TRIAL_ONLY')).sum())
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = pa.schema([(c, pa.float64() if c in ('raw_numeric_value', 'intensity') else pa.string()) for c in frame.columns])
                writer = pq.ParquetWriter(stage / 'ratings.parquet', schema, compression='zstd')
            writer.write_table(table.cast(schema))
        if writer is None:
            raise ValueError('Empty Keller snapshot')
        writer.close()
        writer = None
        if counts['measurement_rows'] != parent['row_count'] or any(seen[k] != r['measurement_rows'] for k, r in by_key.items()):
            raise ValueError('Reviewed row coverage mismatch')
        decision_counts = dict(Counter(r['decision'] for r in decisions))
        report = {'counts': dict(counts), 'identity_decisions': decision_counts,
                  'review_scope_identities': len(decisions), 'descriptor_count': 20, 'mapped_descriptors': 11,
                  'source_only_descriptors': 9, 'status': 'HARMONIZATION_LOCKED_WITH_IDENTITY_HOLDS',
                  'training_eligible': False, 'calibration_eligible': False,
                  'remaining_gates': ['Sample/stereo and conflicting identifiers require evidence',
                                     'No assessed descriptor negatives released',
                                     'No molecular consensus or intensity targets approved']}
        for filename, value in [('descriptor_policy.json', policy), ('identity_decisions.json', decisions),
                                ('identity_evidence.json', evidence), ('identity_evidence_manifest.json', evidence_manifest),
                                ('review_report.json', report)]:
            (stage / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
        if 'cas_correction.json' in parent['files']:
            shutil.copyfile(parent_path.parent / 'cas_correction.json', stage / 'cas_correction.json')
        pd.DataFrame(policy['mappings']).to_csv(stage / 'descriptor_mapping.csv', index=False)
        flat = [{**{k: v for k, v in r.items() if k != 'accepted_reference_structure'},
                 'reference_isomeric_smiles': (r['accepted_reference_structure'] or {}).get('isomeric_smiles'),
                 'reference_inchikey': (r['accepted_reference_structure'] or {}).get('inchikey')}
                for r in decisions]
        pd.DataFrame(flat).to_csv(stage / 'identity_decisions.csv', index=False)
        pd.DataFrame([r for r in flat if r['decision'] != 'ACCEPT_DATABASE_REFERENCE_ONLY']).to_csv(stage / 'identity_hold_queue.csv', index=False)
        manifest = {**unsigned, 'dataset_version': dataset_version, 'created_at': utc_now(),
                    'public_panel_schema_version': 4,
                    'parent_manifest_sha256': file_hash(parent_path), 'parent_dataset_version': parent['dataset_version'],
                    'harmonization_review': {'policy_version': policy['policy_version'],
                        'policy_sha256': file_hash(policy_path), 'identity_evidence_manifest_sha256': file_hash(evidence_dir / 'manifest.json'),
                        'evidence_cache_directory': str(evidence_dir), 'authorization': policy['authorization'],
                        'sample_identity_approved': False},
                    'files': {p.name: file_hash(p) for p in stage.iterdir()}}
        manifest['manifest_sha256'] = content_hash(manifest)
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
        with service.repository.transaction() as connection:
            if target.exists():
                raise FileExistsError('Immutable review snapshot exists')
            connection.execute('INSERT INTO dataset_snapshots VALUES (?, ?, ?, ?, ?, ?)',
                (dataset_version, str(target / 'ratings.parquet'), manifest['files']['ratings.parquet'],
                 str(target / 'manifest.json'), counts['measurement_rows'], utc_now()))
            service.repository.audit(connection, 'KELLER_HARMONIZATION_FINALIZED', 'dataset_snapshot', dataset_version,
                {'manifest_sha256': manifest['manifest_sha256'], 'policy_version': policy['policy_version'],
                 'identity_decisions': decision_counts, 'training_eligible': False})
            stage.rename(target)
        return {'manifest_path': str(target / 'manifest.json'), 'report': report}
    finally:
        if writer is not None:
            writer.close()
        if stage.exists():
            shutil.rmtree(stage)
