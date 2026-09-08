import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from olfactory.data_foundation.keller_review import annotate_ratings, finalize_identity, validate_policy
from olfactory.data_foundation.keller_review import finalize_review
from olfactory.data_foundation import DataFoundationService
from olfactory.data_foundation.evidence import content_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT / 'data/keller_descriptor_policy_v1.json').read_text())


def labels():
    return tuple(json.loads((ROOT / 'data/odor_taxonomy_mapping_v1_2.json').read_text())['labels'])


def test_mapping_is_complete_without_broadening_or_training_approval():
    p = policy()
    validate_policy(p, labels())
    m = {x['source']: x['target'] for x in p['mappings']}
    assert len(m) == 20 and sum(v is not None for v in m.values()) == 11
    assert m['FLOWER'] == 'floral' and m['MUSKY'] is None and m['ACID'] is None
    for mutate in ('musky', 'missing', 'training', 'negative'):
        bad = copy.deepcopy(p)
        if mutate == 'musky':
            next(x for x in bad['mappings'] if x['source'] == 'MUSKY')['target'] = 'musk'
        elif mutate == 'missing':
            bad['mappings'].pop()
        elif mutate == 'training':
            bad['training_eligible'] = True
        else:
            bad['rating_policy']['negative_labels_authorized'] = True
        with pytest.raises(ValueError):
            validate_policy(bad, labels())


def test_ratings_keep_unknowns_context_and_structures():
    frame = pd.DataFrame({
        'source_term': ['FLOWER', 'MUSKY', 'SWEET', 'SWEET', 'SWEET', 'HOW STRONG IS THE SMELL?'],
        'descriptor': [None, None, 'sweet', 'sweet', 'sweet', None],
        'mapping_state': ['REVIEW_REQUIRED'] * 6,
        'raw_numeric_value': [25., 60., None, 0., 90., 0.],
        'raw_value': ['25', '60', None, '0', '90', '0'],
        'detection_response': ['I smell something'] * 4 + ["I can't smell anything", 'I smell something'],
        'presence_state': ['UNASSESSED'] * 6, 'intensity': [None] * 6,
        'evidence_review': ['UNREVIEWED'] * 6, 'context': ['original-context'] * 6,
        'isomeric_smiles': ['CCO'] * 6,
    })
    result = annotate_ratings(frame, policy())
    assert result.descriptor.iloc[0] == 'floral' and pd.isna(result.descriptor.iloc[1])
    assert result.source_applicability_state.tolist() == [
        'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY', 'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY',
        'UNASSESSED', 'UNASSESSED', 'CONFLICT_REVIEW_REQUIRED', 'NOT_DESCRIPTOR']
    for col in frame.columns.difference(['descriptor', 'mapping_state']):
        pd.testing.assert_series_equal(frame[col], result[col])


def identity():
    return {'source_cas': '64-17-5', 'raw_source_identifier': '702',
            'source_names': ['ethanol'], 'measurement_rows': 2,
            'status': 'DATABASE_STRUCTURE_CORROBORATED', 'flags': [],
            'queried_cid': 702, 'returned_cid': 702, 'cas_lookup_cids': [702],
            'source_cas_exact_synonym': True, 'source_cas_checksum_valid': True,
            'pubchem_identity': {'isomeric_smiles': 'CCO', 'inchikey': 'LFQSCWFLJHTTHZ-UHFFFAOYSA-N'},
            'record_url': 'https://pubchem.ncbi.nlm.nih.gov/compound/702'}


def test_reference_acceptance_never_assigns_sample_stereo():
    record = identity()
    result = finalize_identity(record, None)
    assert result['decision'] == 'ACCEPT_DATABASE_REFERENCE_ONLY'
    assert not result['sample_identity_approved'] and not result['training_eligible']
    bad = copy.deepcopy(record)
    bad['pubchem_identity']['inchikey'] = 'wrong'
    with pytest.raises(ValueError):
        finalize_identity(bad, None)
    for status in ('STEREO_OR_MATERIAL_REVIEW_REQUIRED', 'IDENTITY_CONFLICT_REVIEW_REQUIRED',
                   'IDENTITY_UNVERIFIED_REVIEW_REQUIRED'):
        record['status'] = status
        assert finalize_identity(record, None)['decision'].startswith('HOLD_')
    record['status'] = 'invented'
    with pytest.raises(ValueError):
        finalize_identity(record, None)


def test_cas_correction_does_not_upgrade_to_database_identity():
    record = identity()
    record.update(source_cas='109-19-0', raw_source_identifier='109-19-0',
                  source_names=['isobutyl acetate'], status='IDENTIFIER_CORRECTION_PROPOSED')
    correction = {'old_cas': '109-19-0', 'new_cas': '110-19-0', 'source_identifier': '109-19-0',
                  'source_names': ['isobutyl acetate'], 'correction_id': 'fixture'}
    result = finalize_identity(record, correction)
    assert result['decision'] == 'CAS_CORRECTED_IDENTITY_REVIEW_REQUIRED'
    assert result['effective_cas'] == '110-19-0'
    assert result['accepted_reference_structure'] is None


def review_fixture(tmp_path):
    service = DataFoundationService(tmp_path / 'data', labels())
    parent = service.snapshot_root / 'parent'
    parent.mkdir()
    frame = pd.DataFrame({
        'source_cas': ['64-17-5'] * 2, 'raw_source_identifier': ['702'] * 2,
        'source_term': ['FLOWER', 'MUSKY'], 'descriptor': [None, None],
        'mapping_state': ['REVIEW_REQUIRED'] * 2, 'raw_value': ['25', None],
        'raw_numeric_value': [25., None], 'detection_response': ['I smell something'] * 2,
        'presence_state': ['UNASSESSED'] * 2, 'intensity': [None] * 2,
        'evidence_review': ['UNREVIEWED'] * 2, 'isomeric_smiles': ['CCO'] * 2,
    })
    frame.to_parquet(parent / 'ratings.parquet', index=False)
    pd.DataFrame([{'source_cas': '64-17-5', 'raw_source_identifier': '702'}]).to_csv(parent / 'structure_review_queue.csv', index=False)
    manifest = {'dataset_version': 'parent', 'row_count': 2, 'status': 'AUDIT_ONLY',
                'training_eligible': False, 'label_names': list(labels()),
                'original_workbook': {'sha256': 'fixture-workbook'},
                'files': {p.name: file_hash(p) for p in parent.iterdir()}}
    manifest['manifest_sha256'] = content_hash(manifest)
    (parent / 'manifest.json').write_text(json.dumps(manifest))
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'identity_evidence.json').write_text(json.dumps([identity()]))
    em = {'training_eligible': False, 'source_rows': 1,
          'input_sha256': {'original_workbook': 'fixture-workbook'},
          'artifacts_sha256': {'identity_evidence.json': file_hash(evidence / 'identity_evidence.json')},
          'response_files_sha256': {}}
    (evidence / 'manifest.json').write_text(json.dumps(em))
    return service, parent, evidence, frame


def test_review_publication_is_immutable_registered_and_preserves_input(tmp_path):
    service, parent, evidence, frame = review_fixture(tmp_path)
    before = file_hash(parent / 'ratings.parquet')
    kwargs = dict(parent_path=parent / 'manifest.json', evidence_dir=evidence,
                  policy_path=ROOT / 'data/keller_descriptor_policy_v1.json', dataset_version='finalized')
    result = finalize_review(service, **kwargs)
    snapshot = Path(result['manifest_path']).parent
    out = pd.read_parquet(snapshot / 'ratings.parquet')
    assert out.descriptor.iloc[0] == 'floral'
    assert out.reference_identity_decision.eq('ACCEPT_DATABASE_REFERENCE_ONLY').all()
    assert out.presence_state.eq('UNASSESSED').all() and out.intensity.isna().all()
    assert out.isomeric_smiles.equals(frame.isomeric_smiles)
    assert file_hash(parent / 'ratings.parquet') == before
    assert len(service.list_snapshots()) == 1
    assert result['report']['training_eligible'] is False
    with pytest.raises(FileExistsError):
        finalize_review(service, **kwargs)


@pytest.mark.parametrize('failure', ['tampered_evidence', 'missing_coverage', 'wrong_row_count'])
def test_invalid_review_does_not_publish_or_register(tmp_path, failure):
    service, parent, evidence, _ = review_fixture(tmp_path)
    records = [identity()]
    if failure == 'missing_coverage':
        records[0]['source_cas'] = 'another'
    if failure == 'wrong_row_count':
        records[0]['measurement_rows'] = 900
    (evidence / 'identity_evidence.json').write_text(json.dumps(records) + '\n')
    if failure != 'tampered_evidence':
        manifest = json.loads((evidence / 'manifest.json').read_text())
        manifest['artifacts_sha256']['identity_evidence.json'] = file_hash(evidence / 'identity_evidence.json')
        (evidence / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        finalize_review(service, parent_path=parent / 'manifest.json', evidence_dir=evidence,
                        policy_path=ROOT / 'data/keller_descriptor_policy_v1.json', dataset_version='bad')
    assert service.list_snapshots() == [] and not (service.snapshot_root / 'bad').exists()
    assert not list(service.snapshot_root.glob('.keller-review-*'))
