"""Versioned, exact projection of the 113 model labels into Osmo categories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY_PATH = PACKAGE_ROOT / "data" / "osmo_taxonomy_v1_2.json"
DEFAULT_MAPPING_PATH = PACKAGE_ROOT / "data" / "odor_taxonomy_mapping_v1_2.json"

CHEMESTHESIS_CATEGORIES = frozenset(
    {
        "Cooling",
        "Cold/Crisp/Fresh",
        "Heaty/Hot",
        "Harsh",
        "Scratchy",
        "Strong/Pungent/Sharp",
    }
)


@dataclass(frozen=True)
class TaxonomyProfile:
    facets: Tuple[Tuple[str, float], ...]
    textures: Tuple[Tuple[str, float], ...]
    sensations: Tuple[Tuple[str, float], ...]
    projection_name: str
    taxonomy_version: str


@lru_cache(maxsize=4)
def _read_json(path: str) -> Mapping[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_mapping(
    label_names: Sequence[str],
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> Mapping[str, object]:
    """Validate exact label coverage and every category before projection."""
    mapping = _read_json(str(mapping_path.resolve()))
    taxonomy = _read_json(str(taxonomy_path.resolve()))
    entries = mapping.get("labels")
    if not isinstance(entries, dict):
        raise ValueError("Taxonomy mapping has no labels object")

    expected = {str(label) for label in label_names}
    actual = set(entries)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Taxonomy mapping must match model labels exactly; "
            f"missing={missing}, extra={extra}"
        )
    if len(label_names) != len(expected):
        raise ValueError("Model label names must be unique")

    valid_categories = {
        "facets": set(taxonomy["GRAND_FAMILIES"]),
        "textures": set(taxonomy["TEXTURES"]),
        "sensations": set(taxonomy["SENSATIONS"]),
    }
    for label, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Mapping entry for {label} must be an object")
        if not any(entry.get(kind) for kind in valid_categories):
            raise ValueError(f"Mapping entry for {label} is empty")
        for kind, allowed in valid_categories.items():
            categories = entry.get(kind, {})
            if not isinstance(categories, dict):
                raise ValueError(f"{label}.{kind} must be an object")
            for category, weight in categories.items():
                if category not in allowed:
                    raise ValueError(f"Unknown {kind} category: {category}")
                if not 0.0 < float(weight) <= 1.0:
                    raise ValueError(f"Invalid weight for {label}.{category}")
    return mapping


def _weighted_max(
    probabilities: Sequence[float],
    label_names: Sequence[str],
    entries: Mapping[str, object],
    kind: str,
    category_order: Sequence[str],
) -> Tuple[Tuple[str, float], ...]:
    scores: Dict[str, float] = {category: 0.0 for category in category_order}
    for label, probability in zip(label_names, probabilities):
        entry = entries[str(label)]
        categories = entry.get(kind, {})
        for category, weight in categories.items():
            scores[category] = max(scores[category], float(probability) * float(weight))
    if kind == "facets":
        return tuple((category, scores[category]) for category in category_order)
    return tuple(
        sorted(
            ((category, score) for category, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
    )


def project_probabilities(
    probabilities: Sequence[float],
    label_names: Sequence[str],
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> TaxonomyProfile:
    """Project model output using max(probability × explicit mapping weight)."""
    if len(probabilities) != len(label_names):
        raise ValueError("Probability and label counts do not match")
    mapping = load_mapping(
        label_names,
        mapping_path=mapping_path,
        taxonomy_path=taxonomy_path,
    )
    taxonomy = _read_json(str(taxonomy_path.resolve()))
    entries = mapping["labels"]
    return TaxonomyProfile(
        facets=_weighted_max(
            probabilities,
            label_names,
            entries,
            "facets",
            taxonomy["GRAND_FAMILIES"],
        ),
        textures=_weighted_max(
            probabilities,
            label_names,
            entries,
            "textures",
            taxonomy["TEXTURES"],
        ),
        sensations=_weighted_max(
            probabilities,
            label_names,
            entries,
            "sensations",
            taxonomy["SENSATIONS"],
        ),
        projection_name=str(mapping["projection_name"]),
        taxonomy_version=str(mapping["taxonomy_version"]),
    )


def grand_family_colors(
    taxonomy_path: Path = DEFAULT_TAXONOMY_PATH,
) -> Mapping[str, str]:
    taxonomy = _read_json(str(taxonomy_path.resolve()))
    return dict(taxonomy["GRAND_FAMILY_COLORS"])
