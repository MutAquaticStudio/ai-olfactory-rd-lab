from rdkit import Chem

from olfactory.stereo import (
    cip_summary,
    enumerate_stereo_options,
    has_unresolved_stereo,
    resolved_molecules,
)


HEDIONE = "CCCCC1C(CC(=O)C1)CC(=O)OC"


def test_hedione_has_exactly_four_resolved_stereo_options():
    resolution = enumerate_stereo_options(Chem.MolFromSmiles(HEDIONE), limit=16)

    assert resolution.unresolved_elements == 2
    assert not resolution.exceeds_limit
    assert len(resolution.options) == 4
    assert all("@" in option.isomeric_smiles for option in resolution.options)
    assert all("R" in option.cip_summary or "S" in option.cip_summary for option in resolution.options)
    assert all(option.structure_2d_svg.lstrip().startswith("<?xml") for option in resolution.options)


def test_achiral_and_fully_resolved_smiles_need_no_resolution():
    assert not has_unresolved_stereo(Chem.MolFromSmiles("CCO"))
    assert not has_unresolved_stereo(Chem.MolFromSmiles("C[C@H](O)C(=O)O"))


def test_rs_and_ez_descriptors_are_preserved_in_summary():
    molecule = Chem.MolFromSmiles("C[C@H](O)/C=C/F")
    summary = cip_summary(molecule)

    assert any(descriptor in summary for descriptor in (" R", " S"))
    assert any(descriptor in summary for descriptor in (" E", " Z"))


def test_enumeration_over_limit_requires_manual_stereo_input():
    molecule = Chem.MolFromSmiles("CC(F)C(Cl)C(Br)C(I)C(O)C(N)C(S)C")
    resolution = enumerate_stereo_options(molecule, limit=16, include_depictions=False)

    assert resolution.exceeds_limit
    assert resolution.options == ()


def test_candidate_resolution_returns_none_over_four_variants():
    molecule = Chem.MolFromSmiles("CC(F)C(Cl)C(Br)C")

    assert resolved_molecules(molecule, limit=4) is None
