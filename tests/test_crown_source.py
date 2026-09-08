import hashlib
import json

import pytest
import pandas as pd

from ingest_crown_source import mirror
from stage_crown_review import normalize_for_review, stage_source, DESCRIPTORS, IDENTIFIERS


class Response:
    def __init__(self, data=None, body=b''):
        self.data, self.body = data, body
    def raise_for_status(self):
        pass
    def json(self):
        return self.data
    def iter_content(self, size):
        yield self.body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class Session:
    def __init__(self, metadata, body):
        self.metadata, self.body, self.calls = metadata, body, []
    def get(self, url, **kwargs):
        self.calls.append(url)
        return Response(self.metadata, self.body)


def fixture():
    body = b'fixture,data\n1,2\n'
    pin = {'md5': hashlib.md5(body).hexdigest(), 'bytes': len(body)}
    config = {'record_id': 123, 'license_id': 'cc-by-4.0', 'training_eligible': False, 'files': {'data.csv': pin}}
    meta = {'id': 123, 'metadata': {'license': {'id': 'cc-by-4.0'}},
            'files': [{'key': 'data.csv', 'checksum': 'md5:' + pin['md5'], 'size': pin['bytes']}]}
    return config, meta, body


def test_mirror_is_pinned_immutable_and_audit_only(tmp_path):
    config, metadata, body = fixture()
    session = Session(metadata, body)
    result = mirror(config, tmp_path / 'raw', session)
    assert (tmp_path / 'raw/data.csv').read_bytes() == body
    assert result['files'][0]['sha256'] == hashlib.sha256(body).hexdigest()
    assert result['training_eligible'] is False
    assert len(session.calls) == 2
    with pytest.raises(FileExistsError):
        mirror(config, tmp_path / 'raw', session)
    assert len(session.calls) == 2


@pytest.mark.parametrize('error', ['license', 'version', 'corrupt', 'oversized', 'path'])
def test_source_failures_never_publish(tmp_path, error):
    config, meta, body = fixture()
    if error == 'license':
        meta['metadata']['license']['id'] = 'restricted'
    elif error == 'version':
        meta['files'][0]['checksum'] = 'md5:wrong'
    elif error == 'corrupt':
        body = b'x' * len(body)
    elif error == 'oversized':
        body += b'x'
    else:
        config['files']['../data.csv'] = config['files'].pop('data.csv')
    session = Session(meta, body)
    with pytest.raises(ValueError):
        mirror(config, tmp_path / 'raw', session)
    assert not (tmp_path / 'raw').exists()
    assert not list(tmp_path.glob('.crown-download-*'))
    if error in ('license', 'version', 'path'):
        assert len(session.calls) == 1


def ratings():
    data = {k: ['1', '1', '1'] for k in IDENTIFIERS}
    data.update(study=['main', 'retest', 'retest'], sampling_group=['home', 'test', 'retest'],
                code=['p1', 'p2', 'p2'], molcode=['odor1'] * 3, inclusion=['0', '1', '1'])
    data.update({term: ['0', '1', None] for term in DESCRIPTORS})
    data['age'] = ['private'] * 3
    data['phq4_total'] = ['private'] * 3
    return pd.DataFrame(data)


def test_stage_preserves_raw_codes_blanks_cohorts_and_excludes_sensitive_columns():
    frame = normalize_for_review(ratings())
    assert len(frame) == 48
    assert frame.source_response_status.value_counts().to_dict() == {'RECORDED_ZERO': 16, 'RECORDED_ONE': 16, 'MISSING': 16}
    assert frame.presence_state.eq('UNASSESSED').all() and frame.intensity.isna().all()
    assert not {'age', 'phq4_total', 'code'} & set(frame)
    assert frame.loc[frame.source_descriptor.eq('musky'), 'proposed_descriptor'].isna().all()
    assert frame.assessor_id.nunique() == 2
    assert set(frame.sampling_group) == {'home', 'test', 'retest'}
    assert frame.loc[frame.sampling_group.eq('home'), 'cohort_disposition'].eq('EXCLUDED_BY_SOURCE').all()


@pytest.mark.parametrize('failure', ['unknown_code', 'duplicate', 'unknown_cohort'])
def test_bad_source_values_cannot_silently_become_negative(failure):
    data = ratings()
    if failure == 'unknown_code':
        data.loc[0, 'sweet'] = '2'
    elif failure == 'duplicate':
        data = pd.concat([data, data.iloc[:1]], ignore_index=True)
    else:
        data.loc[0, 'sampling_group'] = 'invented'
    with pytest.raises(ValueError):
        normalize_for_review(data)


def test_staging_roundtrip_preserves_sources_and_never_releases_targets(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    ratings().to_csv(raw / 'data.csv', sep=';', index=False)
    pd.DataFrame([{'molcode': 'odor1', 'SMILES': 'CCO', 'cas': '64-17-5',
                   'odor_group': '1', 'concentration_final': '1/100'}]).to_csv(raw / 'odors.csv', sep=';', index=False)
    (raw / 'zenodo_metadata.json').write_text('{}')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'source_config': {'record_id': 15657278, 'license_id': 'cc-by-4.0'},
                'training_eligible': False, 'metadata_sha256': digest(raw / 'zenodo_metadata.json'),
                'files': [{'filename': name, 'sha256': digest(raw / name)} for name in ('data.csv', 'odors.csv')]}
    (raw / 'source-manifest.json').write_text(json.dumps(manifest))
    before = digest(raw / 'data.csv')
    report = stage_source(raw, tmp_path / 'staged')
    assert report['response_slots'] == 48 and report['training_eligible'] is False
    saved = pd.read_parquet(tmp_path / 'staged/source_responses.parquet')
    assert saved.presence_state.eq('UNASSESSED').all()
    assert not {'age', 'phq4_total', 'code'} & set(saved)
    assert digest(raw / 'data.csv') == before
    m = json.loads((tmp_path / 'staged/manifest.json').read_text())
    for name, expected in m['files'].items():
        assert digest(tmp_path / 'staged' / name) == expected
    with pytest.raises(FileExistsError):
        stage_source(raw, tmp_path / 'staged')
    (raw / 'data.csv').write_text('corrupt fixture')
    with pytest.raises(ValueError, match='checksum'):
        stage_source(raw, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
