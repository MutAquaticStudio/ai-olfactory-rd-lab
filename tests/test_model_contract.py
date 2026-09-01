import pytest
import torch
from rdkit import Chem

from olfactory.features import create_morgan_tensor, geometric_mean
from olfactory.models import ODOR_LABEL_COUNT, OdorPredictor


def test_chiral_morgan_contract_and_output_shape():
    first = create_morgan_tensor(Chem.MolFromSmiles("F[C@H](Cl)Br"))
    second = create_morgan_tensor(Chem.MolFromSmiles("F[C@@H](Cl)Br"))
    assert first.shape == (2048,)
    assert first.dtype == torch.float32
    assert not torch.equal(first, second)

    output = OdorPredictor()(first.unsqueeze(0))
    assert output.shape == (1, ODOR_LABEL_COUNT)


def test_target_fit_keeps_geometric_mean_formula():
    values = torch.tensor([0.2, 0.8], dtype=torch.float32)
    assert geometric_mean(values) == pytest.approx(0.4)
