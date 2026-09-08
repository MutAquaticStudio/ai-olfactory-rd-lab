"""Pretraining audit for unassessed Keller evidence, never a target exporter."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from .evidence import checked_file, content_hash, file_hash, load_evidence_snapshot, evidence_coverage
from .repository import utc_now

COLUMNS = ('source_cas', 'raw_source_identifier', 'context', 'descriptor',
           'source_applicability_state', 'raw_value', 'raw_numeric_value',
           'presence_state', 'intensity', 'evidence_review', 'inchikey', 'stereo_state',
           'reference_identity_decision', 'assessor_id', 'trial_id', 'session_id',
           'replicate_number', 'source_replicate_marker', 'detection_response')
PRIORITY = ('musk', 'jasmine', 'floral', 'woody', 'citrus')


def summarize_panel(frames, labels):
    counts = {label: Counter() for label in labels}
    identities = {label: defaultdict(set) for label in labels}
    groups = {}
    for frame in frames:
        if (not frame.presence_state.eq('UNASSESSED').all() or not frame.intensity.isna().all()
                or not frame.evidence_review.eq('UNREVIEWED').all()):
            raise ValueError('Expected audit-only unassessed evidence; use reviewed-evidence audit for released targets')
        selected = frame[frame.descriptor.notna()]
        if not set(selected.descriptor) <= set(labels):
            raise ValueError('Descriptor outside fixed label contract')
        for row in selected.itertuples(index=False):
            label = row.descriptor
            source = (row.source_cas, row.raw_source_identifier)
            # Canonical JSON only: do not merge concentrations, solvents or studies.
            context = json.dumps(json.loads(row.context), sort_keys=True, separators=(',', ':'))
            key = (*source, context, label)
            group = groups.setdefault(key, {'counts': Counter(), 'sets': defaultdict(set)})
            group['counts']['response_slots'] += 1
            group['sets']['assessors'].add(row.assessor_id)
            group['sets']['trials'].add(row.trial_id)
            if pd.notna(row.session_id):
                group['sets']['sessions'].add(row.session_id)
            if pd.notna(row.replicate_number):
                group['sets']['replicates'].add(row.replicate_number)
            if row.source_replicate_marker == 'true':
                group['sets']['source_marked_repeat_trials'].add(row.trial_id)
            if row.detection_response == 'I smell something':
                group['sets']['detected_assessors'].add(row.assessor_id)
            counts[label]['panel_response_slots'] += 1
            blank = pd.isna(row.raw_value)
            zero = pd.notna(row.raw_numeric_value) and row.raw_numeric_value == 0
            counts[label]['blank_slots'] += int(blank)
            counts[label]['explicit_zero_slots'] += int(zero)
            group['counts']['blank_slots'] += int(blank)
            group['counts']['explicit_zero_slots'] += int(zero)
            positive = row.source_applicability_state == 'DESCRIPTOR_APPLIED_AT_TRIAL_ONLY'
            if positive:
                counts[label]['positive_trial_observations'] += 1
                identities[label]['positive_source_identities'].add(source)
                if pd.notna(row.inchikey):
                    identities[label]['positive_source_connectivity_groups'].add(row.inchikey.split('-')[0])
                if row.stereo_state in ('UNKNOWN', 'UNRESOLVED'):
                    identities[label]['positive_source_identities_with_unresolved_stereo'].add(source)
                if row.reference_identity_decision == 'ACCEPT_DATABASE_REFERENCE_ONLY':
                    identities[label]['positive_identities_with_reference_only_acceptance'].add(source)
                if row.reference_identity_decision.startswith('HOLD_') or row.reference_identity_decision == 'CAS_CORRECTED_IDENTITY_REVIEW_REQUIRED':
                    identities[label]['positive_identities_on_hold'].add(source)
                group['sets']['positive_assessors'].add(row.assessor_id)
                group['counts']['positive_trial_observations'] += 1
            if row.source_applicability_state in ('CONFLICT_REVIEW_REQUIRED', 'INVALID_REVIEW_REQUIRED'):
                counts[label]['conflict_or_invalid_slots'] += 1
                group['counts']['conflict_or_invalid_slots'] += 1
    summary = {label: {**dict(counts[label]), **{k: len(v) for k, v in identities[label].items()}}
               for label in labels}
    conditions = []
    for (cas, identifier, context, label), group in sorted(groups.items()):
        sets = group['sets']
        conditions.append({
            'source_cas': cas, 'source_identifier': identifier, 'context': context, 'descriptor': label,
            **dict(group['counts']), 'distinct_assessors': len(sets['assessors']),
            'detected_assessors': len(sets['detected_assessors']), 'positive_assessors': len(sets['positive_assessors']),
            'source_trials': len(sets['trials']), 'source_marked_repeat_trials': len(sets['source_marked_repeat_trials']),
            'documented_sessions': len(sets['sessions']), 'documented_replicates': len(sets['replicates']),
            'agreement_status': 'NOT_EVALUABLE', 'krippendorff_alpha': None, 'intensity_icc': None,
            'reason': 'No assessed descriptor negatives or reviewed intensity targets; source repeat flags do not establish independent sessions',
        })
    return summary, conditions


def coverage_rows(labels, panel, catalog):
    fields = ('panel_response_slots', 'positive_trial_observations', 'positive_source_identities',
              'positive_source_connectivity_groups', 'positive_source_identities_with_unresolved_stereo',
              'positive_identities_with_reference_only_acceptance', 'positive_identities_on_hold',
              'blank_slots', 'explicit_zero_slots', 'conflict_or_invalid_slots')
    return [{
        'descriptor': label, **{field: int(panel.get(label, {}).get(field, 0)) for field in fields},
        'weak_catalog_positive_molecules': int(catalog.get(label, {}).get('catalog_positive', 0)),
        'eligible_positive': 0, 'eligible_negative': 0, 'eligible_intensity': 0,
        'maturity': 'INSUFFICIENT', 'calibration_status': 'NOT_EVALUABLE',
        'global_positive_gap_to_50': 50, 'global_negative_gap_to_50': 50,
    } for label in labels]


def build_readiness_report(panel_manifest, catalog_manifest, output):
    panel_manifest, catalog_manifest, output = map(Path, (panel_manifest, catalog_manifest, output))
    if output.exists():
        raise FileExistsError('Immutable readiness report already exists')
    manifest = json.loads(panel_manifest.read_text())
    if (content_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']
            or manifest.get('status') != 'AUDIT_ONLY' or manifest.get('training_eligible') is not False
            or manifest.get('public_panel_schema_version') != 4):
        raise ValueError('Expected checksummed harmonized audit snapshot')
    for name, sha in manifest['files'].items():
        checked_file(panel_manifest.parent, name, sha)
    cm, molecules, observations = load_evidence_snapshot(catalog_manifest)
    if cm['label_names'] != manifest['label_names'] or cm.get('training_eligible') is not False:
        raise ValueError('Catalog label/release mismatch')
    if not observations.source_kind.eq('CATALOG').all():
        raise ValueError('Keep non-catalog evidence in its own source-domain audit')
    catalog = evidence_coverage(cm, molecules, observations)
    if any(r['eligible_positive'] or r['eligible_negative'] for r in catalog['labels']):
        raise ValueError('Reviewed targets require the general reviewed-evidence audit')
    parquet = pq.ParquetFile(panel_manifest.parent / 'ratings.parquet')
    if parquet.metadata.num_rows != manifest['row_count']:
        raise ValueError('Panel row count mismatch')
    summary, conditions = summarize_panel(
        (batch.to_pandas() for batch in parquet.iter_batches(batch_size=20000, columns=list(COLUMNS))), manifest['label_names'])
    rows = coverage_rows(manifest['label_names'], summary, {r['descriptor']: r for r in catalog['labels']})
    queue = sorted(rows, key=lambda r: (PRIORITY.index(r['descriptor']) if r['descriptor'] in PRIORITY else len(PRIORITY), r['descriptor']))
    gate = {
        'status': 'BLOCKED_DATA_EVIDENCE', 'training_eligible': False, 'calibration_eligible': False,
        'dataset_versions': [manifest['dataset_version'], cm['dataset_version']],
        'label_count': len(rows), 'supported_labels': 0, 'eligible_positive': 0, 'eligible_negative': 0,
        'panel_positive_trial_observations': sum(r['positive_trial_observations'] for r in rows),
        'source_identity_condition_descriptor_groups': len(conditions),
        'agreement_status': 'NOT_EVALUABLE', 'partition_support': 'NOT_EVALUABLE_NO_RELEASED_TRAINING_SNAPSHOT',
        'threshold_notes': ['50/50 global support is not a guarantee of calibration eligibility',
                            'Per-label Platt requires 50 positive and 50 assessed negative within the calibration partition',
                            'Do not invent negatives, infer consensus by OR over assessors, or reuse source repeat flags as session IDs'],
        'next_actions': ['Collect or license protocol-explicit assessed presence/absence evidence for priority descriptors',
                         'Resolve held sample identities with source/stereo/material documentation',
                         'Review measurements and define condition-specific consensus before releasing targets',
                         'Create a new immutable training snapshot and 60/10/15/15 chemical-group split only after the data gate passes'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.readiness-', dir=output.parent))
    try:
        pd.DataFrame(rows).to_csv(stage / 'coverage_113.csv', index=False)
        pd.DataFrame(conditions).to_csv(stage / 'condition_review.csv', index=False)
        pd.DataFrame(queue).to_csv(stage / 'collection_priorities.csv', index=False)
        (stage / 'coverage_113.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
        (stage / 'data_gate.json').write_text(json.dumps(gate, indent=2, allow_nan=False))
        lines = ['# Kiểm tra trước retrain — Keller + catalog', '',
                 '**Kết luận: chưa đủ điều kiện retrain/calibration. Không thay weights.**', '',
                 f"Snapshot panel: `{manifest['dataset_version']}`; catalog: `{cm['dataset_version']}`.", '',
                 'Các cột panel và catalog giữ riêng domain; không cộng chúng thành support đã được duyệt.',
                 'Source identity/connectivity là định danh được ghi trong nguồn, chưa khẳng định đúng mẫu vật lý.', '',
                 '| Nhãn ưu tiên | Dương catalog (phân tử) | Dương panel (lượt ghi nhận) | Định danh nguồn có ghi nhận dương | Âm đủ điều kiện |',
                 '|---|---:|---:|---:|---:|']
        for r in queue[:len(PRIORITY)]:
            lines.append(f"| {r['descriptor']} | {r['weak_catalog_positive_molecules']} | {r['positive_trial_observations']} | {r['positive_source_identities']} | 0 |")
        lines += ['', '## Cách đọc báo cáo', '',
                  '- `coverage_113.csv`: đủ 113 nhãn; không có cột nào được coi là negative chỉ vì nguồn không đề cập.',
                  '- `condition_review.csv`: giữ riêng cấu trúc nguồn/nồng độ/dung môi/descriptor; chỉ xuất số đếm, không xuất ID assessor.',
                  '- Số assessor và lượt lặp không làm tăng số phân tử độc lập.',
                  '- Alpha/ICC là `NOT_EVALUABLE`, không hiển thị thành 0: chưa có target presence/absence hoặc intensity đã duyệt và chưa xác nhận phiên lặp độc lập.',
                  '- Mọi nhãn vẫn `INSUFFICIENT` theo evidence đủ điều kiện; positive thô không mở gate.',
                  '- Mốc 50/50 toàn tập chỉ là mức thiếu hụt ban đầu, không bảo đảm 50/50 trong calibration sau split.', '',
                  '## Việc cần làm tiếp', '',
                  '1. Bổ sung evidence đánh giá rõ có/không cho musk, jasmine, floral, woody, citrus; không thay omission thành ABSENT.',
                  '2. Bổ sung tài liệu định danh mẫu/stereo cho các mục còn hold; không lấy bản ghi PubChem thay chứng nhận vật liệu.',
                  '3. Review từng điều kiện và khóa consensus; giữ nồng độ/nguồn khác nhau riêng biệt.',
                  '4. Khi đủ evidence, tạo snapshot training và split mới; chỉ sau đó chạy baseline/retrain.', '',
                  'Không tạo negative, không tính đồng thuận giả, không liên hệ nhà cung cấp hoặc gửi dữ liệu ra ngoài trong bước này.']
        (stage / 'REPORT_VI.md').write_text('\n'.join(lines) + '\n')
        result = {'schema_version': 1, 'created_at': utc_now(), 'training_eligible': False,
                  'inputs': {str(panel_manifest): file_hash(panel_manifest), str(catalog_manifest): file_hash(catalog_manifest)},
                  'script_sha256': file_hash(Path(__file__)),
                  'files': {p.name: file_hash(p) for p in stage.iterdir()}}
        (stage / 'audit_manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True))
        if output.exists():
            raise FileExistsError('Immutable readiness report already exists')
        stage.rename(output)
        return {'output': str(output), 'gate': gate}
    finally:
        if stage.exists():
            shutil.rmtree(stage)
