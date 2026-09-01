"""Stereochemical identity resolution shared by analysis and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)

from .depictions import molecule_svg
from .features import canonical_isomeric_smiles


@dataclass(frozen=True)
class StereoOption:
    isomeric_smiles: str
    cip_summary: str
    structure_2d_svg: str


@dataclass(frozen=True)
class StereoResolution:
    unresolved_elements: int
    options: Tuple[StereoOption, ...]
    exceeds_limit: bool

    @property
    def complete(self) -> bool:
        return self.unresolved_elements == 0


def unresolved_stereo_elements(molecule: Chem.Mol) -> Tuple[object, ...]:
    """Return potential stereo elements whose configuration is not specified."""
    return tuple(
        item
        for item in Chem.FindPotentialStereo(molecule)
        if item.specified == Chem.StereoSpecified.Unspecified
    )


def has_unresolved_stereo(molecule: Chem.Mol) -> bool:
    return bool(unresolved_stereo_elements(molecule))


def cip_summary(molecule: Chem.Mol) -> str:
    """Create a compact, atom-indexed summary for resolved R/S and E/Z stereo."""
    assigned = Chem.Mol(molecule)
    Chem.AssignStereochemistry(assigned, cleanIt=True, force=True)
    labels = []
    for atom_index, descriptor in Chem.FindMolChiralCenters(
        assigned,
        includeUnassigned=False,
        includeCIP=True,
    ):
        atom = assigned.GetAtomWithIdx(atom_index)
        labels.append(f"{atom.GetSymbol()}{atom_index + 1} {descriptor}")

    for bond in assigned.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            continue
        descriptor = "E" if stereo == Chem.BondStereo.STEREOE else "Z"
        labels.append(
            f"{bond.GetBeginAtomIdx() + 1}-{bond.GetEndAtomIdx() + 1} {descriptor}"
        )
    return " · ".join(labels) or "No assigned stereogenic elements"


def enumerate_stereo_options(
    molecule: Chem.Mol,
    *,
    limit: int,
    include_depictions: bool = True,
) -> StereoResolution:
    """Enumerate only unresolved stereo and report when the configured cap is exceeded."""
    unresolved = unresolved_stereo_elements(molecule)
    if not unresolved:
        return StereoResolution(0, (), False)

    options = StereoEnumerationOptions(
        onlyUnassigned=True,
        unique=True,
        maxIsomers=limit + 1,
        tryEmbedding=False,
    )
    enumerated = list(EnumerateStereoisomers(molecule, options=options))
    if len(enumerated) > limit:
        return StereoResolution(len(unresolved), (), True)

    unique = {}
    for isomer in enumerated:
        smiles = canonical_isomeric_smiles(isomer)
        unique[smiles] = StereoOption(
            isomeric_smiles=smiles,
            cip_summary=cip_summary(isomer),
            structure_2d_svg=molecule_svg(isomer) if include_depictions else "",
        )
    return StereoResolution(
        unresolved_elements=len(unresolved),
        options=tuple(unique[key] for key in sorted(unique)),
        exceeds_limit=False,
    )


def resolved_molecules(
    molecule: Chem.Mol,
    *,
    limit: int,
) -> Optional[Tuple[Chem.Mol, ...]]:
    """Return resolved variants, or ``None`` when enumeration exceeds the limit."""
    if not has_unresolved_stereo(molecule):
        return (Chem.Mol(molecule),)
    options = StereoEnumerationOptions(
        onlyUnassigned=True,
        unique=True,
        maxIsomers=limit + 1,
        tryEmbedding=False,
    )
    variants = list(EnumerateStereoisomers(molecule, options=options))
    if len(variants) > limit:
        return None
    return tuple(
        sorted(
            (Chem.Mol(variant) for variant in variants),
            key=canonical_isomeric_smiles,
        )
    )
