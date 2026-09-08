#!/usr/bin/env python3
"""Mirror the explicitly licensed, pinned CROWN source into private storage.

Only downloads a reviewed file allowlist. Never runs publisher notebooks or
imports demographic/health fields into the application or training pipeline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def mirror(config, output, session=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Immutable raw source already exists')
    if config.get('license_id') != 'cc-by-4.0' or config.get('training_eligible') is not False:
        raise ValueError('Expected reviewed CC-BY-4.0 audit source')
    record_id = config['record_id']
    if type(record_id) is not int or record_id <= 0:
        raise ValueError('Invalid source record ID')
    owned = session is None
    session = session or requests.Session()
    if owned:
        session.headers.update({'User-Agent': 'Scent-Molecule-Studio/source-audit (local research)'})
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'])))
    stage = None
    try:
        url = f'https://zenodo.org/api/records/{record_id}'
        response = session.get(url, timeout=(3.05, 20))
        response.raise_for_status()
        metadata = response.json()
        if (metadata.get('id') != record_id or metadata.get('metadata', {}).get('license', {}).get('id') != config['license_id']):
            raise ValueError('Published license/record mismatch')
        published = {f['key']: f for f in metadata['files']}
        for filename, pin in config['files'].items():
            if Path(filename).name != filename or filename not in published:
                raise ValueError('Invalid allowlisted filename')
            if published[filename]['checksum'] != 'md5:' + pin['md5'] or published[filename]['size'] != pin['bytes']:
                raise ValueError('Published source version changed')
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.crown-download-', dir=output.parent))
        records = []
        for filename, pin in config['files'].items():
            url = f'https://zenodo.org/api/records/{record_id}/files/{filename}/content'
            md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
            with session.get(url, stream=True, timeout=(3.05, 20)) as response:
                response.raise_for_status()
                with (stage / filename).open('xb') as handle:
                    for block in response.iter_content(65536):
                        size += len(block)
                        if size > min(pin['bytes'], 10 * 1024 * 1024):
                            raise ValueError('Source exceeds pinned size limit')
                        md5.update(block); sha.update(block); handle.write(block)
            if md5.hexdigest() != pin['md5'] or size != pin['bytes']:
                raise ValueError('Source checksum/size mismatch')
            records.append({'filename': filename, 'url': url, 'md5': md5.hexdigest(), 'sha256': sha.hexdigest(), 'bytes': size})
        (stage / 'zenodo_metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True))
        manifest = {'source_config': config, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                    'files': records, 'metadata_sha256': hashlib.sha256((stage / 'zenodo_metadata.json').read_bytes()).hexdigest(),
                    'training_eligible': False}
        (stage / 'source-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if output.exists():
            raise FileExistsError('Immutable raw source already exists')
        stage.rename(output)
        return manifest
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        if owned:
            session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'data/bierling_2025_source_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = mirror(json.loads(args.config.read_text()), args.output)
    print(json.dumps({'output': str(args.output), 'files': result['files'], 'training_eligible': False}, indent=2))
