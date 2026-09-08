import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finalize_crown_review import apply_source_policy, agreement_audit
from olfactory.training.reliability import krippendorff_alpha_nominal


POLICY = json.loads((Path(__file__).parents[1] / 'data/crown_descriptor_policy_v1.json').read_text())


def test_trial_encoding_is_not_a_training_target_or_intensity():
    frame = pd.DataFrame({'source_descriptor': ['sweet', 'sweet', 'sweet', 'musky'],
                          'raw_binary_code': ['0', '1', None, '1']})
    out = apply_source_policy(frame, POLICY)
    assert out.source_presence_state.tolist() == ['ABSENT_AT_TRIAL', 'PRESENT_AT_TRIAL', 'UNASSESSED', 'PRESENT_AT_TRIAL']
    assert out.presence_state.eq('UNASSESSED').all()
    assert out.intensity.isna().all()
    assert pd.isna(out.iloc[-1].descriptor)
    assert out.iloc[1].descriptor == 'sweet'
    assert out.raw_binary_code.tolist() == frame.raw_binary_code.tolist()


def test_policy_cannot_silently_accept_unknown_encoding_or_terms():
    with pytest.raises(ValueError):
        apply_source_policy(pd.DataFrame({'source_descriptor': ['sweet'], 'raw_binary_code': ['yes']}), POLICY)
    with pytest.raises(ValueError):
        apply_source_policy(pd.DataFrame({'source_descriptor': ['unknown'], 'raw_binary_code': ['1']}), POLICY)
    policy = json.loads(json.dumps(POLICY))
    policy['mappings'][0]['target'] = 'fake'
    with pytest.raises(ValueError):
        apply_source_policy(pd.DataFrame({'source_descriptor': ['sweet'], 'raw_binary_code': ['1']}), policy)


def test_agreement_known_results_and_missing_data():
    assert krippendorff_alpha_nominal(np.array([[0, 0], [1, 1]])) == 1
    assert krippendorff_alpha_nominal(np.array([[0, 1], [0, 1]])) == pytest.approx(-0.5)
    assert np.isnan(krippendorff_alpha_nominal(np.zeros((2, 2))))
    records = []
    for mol, value in [('a', '0'), ('b', '1')]:
        for session in ['test', 'retest']:
            for assessor in ['p1', 'p2']:
                records.append(dict(study='retest', sampling_group=session, odor_set='retest',
                    molcode=mol, assessor_id=assessor, inclusion='1', source_descriptor='sweet', raw_binary_code=value))
    results = agreement_audit(pd.DataFrame(records))
    assert results[0]['alpha'] == 1
    assert results[0]['assessors_with_two_sessions_minimum'] == 2
    assert results[0]['passes_repeated_panel_gate'] is False  # Need eight, not two.
