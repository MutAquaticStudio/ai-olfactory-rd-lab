import numpy as np
import pandas as pd
import pytest
import torch

from train_crown_exploratory import condition_targets, masked_bce


def test_condition_rates_preserve_missing_and_never_merge_dilutions():
    rows = []
    for condition, values in [('set1', ['PRESENT_AT_TRIAL', 'ABSENT_AT_TRIAL']),
                              ('set2', ['ABSENT_AT_TRIAL', 'ABSENT_AT_TRIAL'])]:
        for value in values:
            rows.append(dict(condition_id=condition, descriptor='sweet', source_presence_state=value,
                             isomeric_smiles='CCO'))
    rows.append(dict(condition_id='set1', descriptor='sweet', source_presence_state='UNASSESSED', isomeric_smiles='CCO'))
    table, targets, counts = condition_targets(pd.DataFrame(rows), ['sweet', 'musk'])
    assert table.condition_id.tolist() == ['set1', 'set2']
    np.testing.assert_allclose(targets[:, 0], [0.5, 0.0])
    assert np.isnan(targets[:, 1]).all()
    np.testing.assert_array_equal(counts[:, 0], [2, 2])


def test_masked_bce_has_no_gradient_for_unassessed_output():
    logits = torch.zeros((1, 3), requires_grad=True)
    loss = masked_bce(logits, torch.tensor([[0.0, 1.0, float('nan')]]))
    loss.backward()
    assert loss.item() == pytest.approx(np.log(2))
    np.testing.assert_allclose(logits.grad.numpy(), [[0.25, -0.25, 0.]])


def test_condition_cannot_have_two_identities_or_unknown_states():
    rows = pd.DataFrame([dict(condition_id='x', descriptor='sweet', source_presence_state='PRESENT_AT_TRIAL', isomeric_smiles='CCO'),
                         dict(condition_id='x', descriptor='sweet', source_presence_state='ABSENT_AT_TRIAL', isomeric_smiles='CC')])
    with pytest.raises(ValueError):
        condition_targets(rows, ['sweet'])
    rows.loc[1, 'isomeric_smiles'] = 'CCO'
    rows.loc[1, 'source_presence_state'] = 'UNKNOWN_BAD_STATE'
    with pytest.raises(ValueError):
        condition_targets(rows, ['sweet'])
