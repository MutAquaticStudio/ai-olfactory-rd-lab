from pathlib import Path

import pytest
import torch

from olfactory.taxonomy import load_mapping, project_probabilities


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def label_names():
    dataset = torch.load(
        ROOT / "odor_morgan_tensor_dataset.pt",
        map_location="cpu",
        weights_only=False,
    )
    return tuple(str(label) for label in dataset.label_names)


def test_mapping_contains_exactly_the_113_model_labels(label_names):
    mapping = load_mapping(label_names)
    assert len(mapping["labels"]) == 113
    assert set(mapping["labels"]) == set(label_names)


def test_radar_always_has_the_11_ordered_grand_families(label_names):
    profile = project_probabilities([0.0] * 113, label_names)
    assert [name for name, _ in profile.facets] == [
        "Animalic",
        "Citrus",
        "Floral",
        "Fruity",
        "Green",
        "Herbal",
        "Industrial",
        "Mineral",
        "Soulful",
        "Sweet/Balsamic",
        "Woody",
    ]


def test_weighted_max_is_not_a_sum_or_average(label_names):
    probabilities = [0.0] * 113
    probabilities[label_names.index("burnt")] = 0.8
    probabilities[label_names.index("woody")] = 0.2
    profile = project_probabilities(probabilities, label_names)
    facets = dict(profile.facets)
    assert facets["Industrial"] == pytest.approx(0.8 * 0.6)
    assert facets["Woody"] == pytest.approx(max(0.8 * 0.4, 0.2))
