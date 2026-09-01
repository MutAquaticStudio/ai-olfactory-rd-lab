from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import olfactory.chemistry as chemistry_module
from olfactory.chemistry import (
    ChemicalDecision,
    build_conformer_ensemble,
    clear_conformer_cache,
    screen_molecule,
)
from olfactory.stereo import enumerate_stereo_options, resolved_molecules


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("smiles", "reason"),
    [
        ("CC.CC", "MULTIPLE_FRAGMENTS"),
        ("[NH4+]", "NONZERO_FORMAL_CHARGE"),
        ("[CH3]", "RADICAL"),
        ("COOC", "PEROXIDE"),
        ("CN=[N+]=[N-]", "AZIDE"),
        ("O=[N+]([O-])c1ccccc1", "AROMATIC_NITRO"),
        ("NN", "HYDRAZINE"),
        ("CC(=O)Cl", "ACYL_HALIDE"),
        ("CCN=C=O", "ISOCYANATE"),
        ("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC", "MW_OUT_OF_RANGE"),
    ],
)
def test_hard_rejections(smiles, reason):
    result = screen_molecule(Chem.MolFromSmiles(smiles))
    assert result.decision is ChemicalDecision.REJECT
    assert reason in result.reason_codes


def test_invalid_molecule_is_rejected():
    result = screen_molecule(None)
    assert result.decision is ChemicalDecision.REJECT
    assert result.reason_codes == ("PARSE_OR_SANITIZE_ERROR",)


def test_dataset_retention_and_macrocycle_exception():
    frame = pd.read_csv(ROOT / "clean_dataset.csv")
    results = [screen_molecule(Chem.MolFromSmiles(str(value))) for value in frame["SMILES"]]
    retained = sum(result.decision is not ChemicalDecision.REJECT for result in results)
    assert retained / len(results) >= 0.97

    macrocycles = [result for result in results if result.is_macrocycle]
    assert len(macrocycles) == 9
    assert all(result.decision is ChemicalDecision.PASS for result in macrocycles)


def test_all_nine_reference_macrocycles_embed_and_converge_with_etkdgv3():
    frame = pd.read_csv(ROOT / "clean_dataset.csv")
    macrocycle_smiles = []
    for value in frame["SMILES"]:
        molecule = Chem.MolFromSmiles(str(value))
        if screen_molecule(molecule).is_macrocycle:
            variants = resolved_molecules(molecule, limit=16)
            assert variants
            macrocycle_smiles.append(
                Chem.MolToSmiles(variants[0], canonical=True, isomericSmiles=True)
            )
    results = [build_conformer_ensemble(smiles) for smiles in macrocycle_smiles]
    assert len(results) == 9
    assert all(result.available for result in results)
    assert all(result.requested_count == 100 for result in results)
    assert all(result.converged_count >= 1 for result in results)
    assert all(
        record.relative_energy >= 0
        for result in results
        for record in result.conformers
    )
    assert all(result.conformers[0].relative_energy == pytest.approx(0.0) for result in results)
    for result in results:
        for record in result.conformers:
            molecule = Chem.MolFromMolBlock(record.molblock, removeHs=False)
            assert molecule is not None
            assert not chemistry_module._heavy_atom_clash(
                molecule,
                molecule.GetConformer().GetId(),
            )


def test_invalid_3d_input_returns_a_nonfatal_fallback_result():
    result = build_conformer_ensemble("not-a-smiles")
    assert not result.available
    assert result.error


def test_unresolved_stereo_is_not_embedded():
    result = build_conformer_ensemble("CCCCC1C(CC(=O)C1)CC(=O)OC")

    assert not result.available
    assert result.error == "STEREO_UNRESOLVED"


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H](O)C(=O)O",
        "C[C@@H](O)C(=O)O",
        "F/C=C/F",
        "F/C=C\\F",
        "C[C@H](O)/C=C/F",
    ],
)
def test_explicit_rs_and_ez_survive_conformer_round_trip(smiles):
    expected = Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles),
        canonical=True,
        isomericSmiles=True,
    )
    result = build_conformer_ensemble(expected)

    assert result.available, result.error
    for conformer in result.conformers:
        parsed = Chem.RemoveHs(
            Chem.MolFromMolBlock(conformer.molblock, removeHs=False)
        )
        assert Chem.MolToSmiles(
            parsed,
            canonical=True,
            isomericSmiles=True,
        ) == expected


def test_hedione_ensemble_returns_sorted_distinct_low_energy_representatives():
    option = enumerate_stereo_options(
        Chem.MolFromSmiles("CCCCC1C(CC(=O)C1)CC(=O)OC"),
        limit=16,
        include_depictions=False,
    ).options[0]
    result = build_conformer_ensemble(option.isomeric_smiles)

    assert result.available
    assert len(result.conformers) <= 5
    energies = [item.relative_energy for item in result.conformers]
    assert energies == sorted(energies)
    assert energies[0] == pytest.approx(0.0)
    molecules = [
        Chem.RemoveHs(Chem.MolFromMolBlock(item.molblock, removeHs=False))
        for item in result.conformers
    ]
    for index, molecule in enumerate(molecules):
        for other in molecules[index + 1 :]:
            assert rdMolAlign.GetBestRMS(Chem.Mol(molecule), Chem.Mol(other)) >= 0.75


def test_same_seed_rebuilds_same_order_and_relative_energies():
    clear_conformer_cache()
    first = build_conformer_ensemble("CCCCC")
    clear_conformer_cache()
    second = build_conformer_ensemble("CCCCC")

    assert first.method == second.method
    assert first.embedded_count == second.embedded_count
    assert [item.relative_energy for item in first.conformers] == pytest.approx(
        [item.relative_energy for item in second.conformers],
        abs=1e-6,
    )


def test_optimizer_retries_nonconverged_force_field(monkeypatch):
    class ForceField:
        def __init__(self):
            self.calls = []

        def Minimize(self, maxIts):
            self.calls.append(maxIts)
            return 1 if len(self.calls) == 1 else 0

        def CalcEnergy(self):
            return 12.5

    force_field = ForceField()
    monkeypatch.setattr(
        chemistry_module,
        "_force_field",
        lambda *_args, **_kwargs: force_field,
    )

    status, energy = chemistry_module._optimize_conformer(
        Chem.AddHs(Chem.MolFromSmiles("CCO")),
        0,
        "MMFF94s",
    )

    assert status == 0
    assert energy == pytest.approx(12.5)
    assert force_field.calls == [1000, 4000]


def test_nonconverged_conformer_is_excluded(monkeypatch):
    molecule = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=1))
    monkeypatch.setattr(
        chemistry_module,
        "_optimize_conformer",
        lambda *_args, **_kwargs: (1, 1.0),
    )

    _, energies = chemistry_module._validated_energies(
        molecule,
        conformer_ids,
        "MMFF94s",
        "CCO",
    )

    assert energies == []


def test_uff_fallback_is_used_only_when_mmff_has_no_valid_conformer(monkeypatch):
    calls = []

    def validation(molecule, conformer_ids, method, _expected):
        calls.append(method)
        if method == "MMFF94s":
            return Chem.Mol(molecule), []
        return Chem.Mol(molecule), [(3.0, conformer_ids[0])]

    monkeypatch.setattr(chemistry_module, "_validated_energies", validation)
    clear_conformer_cache()
    result = build_conformer_ensemble("CCO")

    assert calls == ["MMFF94s", "UFF"]
    assert result.method == "UFF"
    assert result.conformers[0].relative_energy == pytest.approx(0.0)
    clear_conformer_cache()
