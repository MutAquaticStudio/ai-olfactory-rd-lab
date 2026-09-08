import json
from pathlib import Path
import pandas as pd
import pytest

from olfactory.data_foundation.keller_readiness import summarize_panel, coverage_rows, build_readiness_report
from olfactory.data_foundation import DataFoundationService
from olfactory.data_foundation.evidence import content_hash, file_hash


def fixture():
    return pd.DataFrame({
        'source_cas': ['example'] * 4, 'raw_source_identifier': ['source-1'] * 4,
        'context': ['{"concentration": 0.01}'] * 3 + ['{"concentration": 0.001}'],
        'descriptor': ['floral'] * 4, 'source_applicability_state': [
            'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY', 'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY',
            'UNASSESSED', 'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY'],
        'raw_value': ['50', '40', None, '30'], 'raw_numeric_value': [50., 40., None, 30.],
        'presence_state': ['UNASSESSED'] * 4, 'intensity': [None] * 4,
        'evidence_review': ['UNREVIEWED'] * 4, 'inchikey': ['ABC-DEF-G'] * 4,
        'stereo_state': ['UNRESOLVED'] * 4, 'reference_identity_decision': ['HOLD_STEREO_OR_MATERIAL'] * 4,
        'assessor_id': ['a', 'a', 'b', 'a'], 'trial_id': ['t1', 't2', 't3', 't4'],
        'session_id': [None] * 4, 'replicate_number': [None] * 4,
        'source_replicate_marker': ['false', 'true', 'false', 'false'],
        'detection_response': ['I smell something'] * 4,
    })


def test_count_identities_not_assessors_and_keep_conditions_separate():
    frame = fixture()
    summary, conditions = summarize_panel([frame.iloc[:2], frame.iloc[2:]], ('floral', 'musk'))
    floral = summary['floral']
    assert floral['positive_trial_observations'] == 3
    assert floral['positive_source_identities'] == 1
    assert floral['positive_source_connectivity_groups'] == 1
    assert len(conditions) == 2
    high = next(c for c in conditions if json.loads(c['context'])['concentration'] == 0.01)
    assert high['source_trials'] == 3
    assert high['positive_assessors'] == 1
    assert all(x['agreement_status'] == 'NOT_EVALUABLE' for x in conditions)
    assert all(x['krippendorff_alpha'] is None and x['intensity_icc'] is None for x in conditions)
    rows = coverage_rows(('floral', 'musk'), summary, {})
    assert rows[0]['eligible_positive'] == rows[0]['eligible_negative'] == 0
    assert rows[0]['maturity'] == rows[1]['maturity'] == 'INSUFFICIENT'
    assert rows[1]['panel_response_slots'] == 0


def test_weak_catalog_positive_does_not_open_gate():
    summary, _ = summarize_panel([fixture()], ('floral',))
    result = coverage_rows(('floral',), summary, {'floral': {'catalog_positive': 500}})[0]
    assert result['weak_catalog_positive_molecules'] == 500
    assert result['eligible_positive'] == 0 and result['maturity'] == 'INSUFFICIENT'


@pytest.mark.parametrize('column,value', [('presence_state', 'ABSENT'), ('intensity', 5), ('evidence_review', 'APPROVED')])
def test_audit_only_analyzer_rejects_unexpected_released_evidence(column, value):
    frame = fixture()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match='audit-only'):
        summarize_panel([frame], ('floral',))


def test_report_roundtrip_checksums_and_no_snapshot_mutation(tmp_path):
    service = DataFoundationService(tmp_path / 'data', ('floral', 'musk'))
    service.import_evidence_records(
        {'source_id': 'catalog', 'kind': 'CATALOG', 'version': '1', 'license_status': 'REVIEW_REQUIRED', 'evidence_review': 'UNREVIEWED'},
        [{'record_id': 'one', 'raw_smiles': 'CCO', 'observations': [{'descriptor': 'floral', 'presence_state': 'PRESENT', 'intensity': None}]}])
    catalog = service.create_evidence_snapshot('catalog-v1', ['catalog'])['manifest_path']
    panel = tmp_path / 'panel'
    panel.mkdir()
    fixture().to_parquet(panel / 'ratings.parquet', index=False)
    m = {'status': 'AUDIT_ONLY', 'training_eligible': False, 'dataset_version': 'panel-v5',
         'public_panel_schema_version': 4, 'label_names': ['floral', 'musk'], 'row_count': 4,
         'files': {'ratings.parquet': file_hash(panel / 'ratings.parquet')}}
    m['manifest_sha256'] = content_hash(m)
    (panel / 'manifest.json').write_text(json.dumps(m))
    hashes = {p: file_hash(p) for p in (Path(catalog), panel / 'manifest.json', panel / 'ratings.parquet')}
    result = build_readiness_report(panel / 'manifest.json', catalog, tmp_path / 'report')
    assert result['gate']['eligible_negative'] == 0
    output = tmp_path / 'report'
    data = pd.read_csv(output / 'coverage_113.csv')
    assert len(data) == 2 and data.weak_catalog_positive_molecules.sum() == 1
    audit = json.loads((output / 'audit_manifest.json').read_text())
    for name, checksum in audit['files'].items():
        assert file_hash(output / name) == checksum
    assert all(file_hash(p) == checksum for p, checksum in hashes.items())
    assert len(service.list_snapshots()) == 1  # audit did not publish training data
    with pytest.raises(FileExistsError):
        build_readiness_report(panel / 'manifest.json', catalog, output)
    (panel / 'ratings.parquet').write_bytes(b'tampered fixture')
    with pytest.raises(ValueError, match='checksum'):
        build_readiness_report(panel / 'manifest.json', catalog, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
