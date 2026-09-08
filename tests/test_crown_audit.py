import json
import pandas as pd
import pytest

from audit_crown_evidence import audit_identities, condition_candidates, cas_valid, run_audit
from olfactory.data_foundation.evidence import file_hash
from stage_crown_review import DESCRIPTORS, IDENTIFIERS


def test_cas_checksum_is_not_identity_approval():
    assert cas_valid('64-17-5')
    assert not cas_valid('64-17-6')
    assert not cas_valid(None)
    rows = audit_identities(pd.DataFrame([{'molcode': 'a', 'cas': '64-17-5',
        'SMILES': 'CCO', 'cid': '702', 'name': 'ethanol'}]))
    assert rows[0]['cas_checksum_valid']
    assert rows[0]['identity_status'] == 'DATABASE_CROSSCHECK_REQUIRED'
    assert rows[0]['training_eligible'] is False


def test_identity_flags_keep_racemic_source_and_never_assign_stereo():
    data = pd.DataFrame([
        {'molcode': 'a', 'SMILES': 'C[C@H](O)CC', 'name': '(±)-sample', 'cas': '64-17-5'},
        {'molcode': 'b', 'SMILES': 'CC(O)CC', 'name': 'sample', 'cas': '64-17-5'},
        {'molcode': 'c', 'SMILES': 'bad', 'name': 'bad', 'cas': 'bad'},
    ])
    result = audit_identities(data)
    assert 'RACEMIC_NAME_WITH_SPECIFIED_STEREO' in result[0]['flags']
    assert 'UNRESOLVED_STEREO' in result[1]['flags']
    assert 'INVALID_SMILES' in result[2]['flags']
    assert result[0]['raw_smiles'] == data.iloc[0].SMILES
    assert 'CAS_MULTIPLE_ISOMERIC_STRUCTURES' in result[0]['flags']


def test_condition_lookup_never_joins_pea_across_sets_or_guesses_retest():
    stimuli = pd.DataFrame([
        {'odor_group': '1', 'molcode': 'PEA'},
        {'odor_group': '9', 'molcode': 'PEA'},
        {'odor_group': '0', 'molcode': 'anchor'},
    ])
    assert condition_candidates('main', 'home', '1', 'PEA', stimuli) == ([2], 'SET_MATCH_PROPOSED')
    assert condition_candidates('main', 'lab', '9', 'PEA', stimuli) == ([3], 'SET_MATCH_PROPOSED')
    assert condition_candidates('main', 'home', '1', 'anchor', stimuli) == ([4], 'ANCHOR_MATCH_PROPOSED')
    assert condition_candidates('retest', 'test', 'retest', 'PEA', stimuli) == ([2, 3], 'RETEST_CONTEXT_REVIEW')
    assert condition_candidates('main', 'pat', '9', 'PEA', stimuli)[1] == 'PATIENT_CONTEXT_REVIEW'
    assert condition_candidates('main', 'home', '2', 'PEA', stimuli) == ([], 'NO_CONTEXT_MATCH')
    duplicated = pd.concat([stimuli, stimuli.iloc[:1]], ignore_index=True)
    assert condition_candidates('main', 'home', '1', 'PEA', duplicated)[1] == 'AMBIGUOUS_CONTEXT'


def test_audit_roundtrip_is_private_immutable_and_fails_closed(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    row = {k: '1' for k in IDENTIFIERS}
    row.update(study='main', sampling_group='home', code='private-participant', molcode='PEA')
    row.update({d: '0' for d in DESCRIPTORS})
    row['sweet'] = None
    pd.DataFrame([row]).to_csv(raw / 'data.csv', sep=';', index=False)
    pd.DataFrame([{'odor_group': '1', 'molcode': 'PEA', 'SMILES': 'CCO', 'cas': '64-17-5'}]).to_csv(raw / 'odors.csv', sep=';', index=False)
    (raw / 'zenodo_metadata.json').write_text('{}')
    manifest = {'source_config': {'record_id': 15657278, 'license_id': 'cc-by-4.0'},
        'training_eligible': False, 'metadata_sha256': file_hash(raw / 'zenodo_metadata.json'),
        'files': [{'filename': n, 'sha256': file_hash(raw / n)} for n in ['data.csv', 'odors.csv']]}
    (raw / 'source-manifest.json').write_text(json.dumps(manifest))
    result = run_audit(raw, tmp_path / 'audit')
    assert result['context_status_rows'] == {'SET_MATCH_PROPOSED': 1}
    assert result['training_eligible'] is False
    conditions = json.loads((tmp_path / 'audit/condition_review.json').read_text())
    assert conditions[0]['descriptor_counts']['sweet'] == {'recorded_zero': 0, 'recorded_one': 0, 'missing': 1}
    mappings = json.loads((tmp_path / 'audit/descriptor_review.json').read_text())
    assert len(mappings) == 16
    assert all(not r['training_eligible'] for r in mappings)
    saved = json.loads((tmp_path / 'audit/manifest.json').read_text())
    for name, digest in saved['files'].items():
        assert file_hash(tmp_path / 'audit' / name) == digest
        assert 'private-participant' not in (tmp_path / 'audit' / name).read_text()
    with pytest.raises(FileExistsError):
        run_audit(raw, tmp_path / 'audit')
    (raw / 'data.csv').write_text('corrupt')
    with pytest.raises(ValueError, match='checksum'):
        run_audit(raw, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
