"""Chemical screening and deterministic high-fidelity conformer ensembles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    FilterCatalog,
    rdMolAlign,
    rdMolDescriptors,
)
from rdkit.Contrib.SA_Score import sascorer

from .features import canonical_isomeric_smiles
from .stereo import has_unresolved_stereo


class ChemicalDecision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ChemicalScreenResult:
    decision: ChemicalDecision
    reason_codes: Tuple[str, ...]
    descriptors: Dict[str, float]
    is_macrocycle: bool
    macrocycle_ring_size: Optional[int]
    macrocycle_carbon_fraction: float
    macrocycle_heteroatoms: int
    alerts: Tuple[str, ...]


@dataclass(frozen=True)
class ConformerRecord:
    molblock: str
    relative_energy: float


@dataclass(frozen=True)
class ConformerEnsembleResult:
    conformers: Tuple[ConformerRecord, ...]
    method: Optional[str]
    requested_count: int
    embedded_count: int
    converged_count: int
    is_macrocycle: bool
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.conformers)


NORMAL_CONFORMER_COUNT = 50
MACROCYCLE_CONFORMER_COUNT = 100
MAX_ENSEMBLE_SIZE = 5
NORMAL_CLUSTER_RMSD = 0.75
MACROCYCLE_CLUSTER_RMSD = 1.0
MIN_HEAVY_ATOM_DISTANCE = 1.0
CONFORMER_CACHE_SIZE = 128
CONFORMER_RANDOM_SEED = 0xF00D

_CONFORMER_LOCK = threading.Lock()


ALLOWED_ELEMENTS = frozenset({"H", "C", "N", "O", "S", "F", "Cl", "Br", "I"})

_HARD_ALERT_SMARTS = {
    "AROMATIC_NITRO": "[c:1]-[N+:2](=[O:3])-[O-:4]",
    "PEROXIDE": "[O;X2:1]-[O;X2:2]",
    "AZIDE": "[$([N:1]=[N+:2]=[N-:3]),$([N-:1]=[N+:2]=[N:3])]",
    "DIAZONIUM": "[N+:1]#[N:2]",
    "AZO": "[N;X2:1]=[N;X2:2]",
    "HYDRAZINE": "[N;X3:1]-[N;X3:2]",
    "ACYL_HALIDE": "[C;X3:1](=[O;X1:2])[F,Cl,Br,I:3]",
    "ISOCYANATE": "[N;X2:1]=[C;X2:2]=[O;X1:3]",
}
_HARD_ALERT_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in _HARD_ALERT_SMARTS.items()
}


def _build_review_catalog() -> FilterCatalog.FilterCatalog:
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
    return FilterCatalog.FilterCatalog(params)


_REVIEW_CATALOG = _build_review_catalog()


def _empty_result(code: str) -> ChemicalScreenResult:
    return ChemicalScreenResult(
        decision=ChemicalDecision.REJECT,
        reason_codes=(code,),
        descriptors={},
        is_macrocycle=False,
        macrocycle_ring_size=None,
        macrocycle_carbon_fraction=0.0,
        macrocycle_heteroatoms=0,
        alerts=(),
    )


def macrocycle_metadata(molecule: Chem.Mol) -> Tuple[bool, Optional[int], float, int]:
    """Identify the protected 15–17 member non-aromatic musk profile."""
    best: Optional[Tuple[int, float, int]] = None
    for ring in molecule.GetRingInfo().AtomRings():
        ring_size = len(ring)
        if not 15 <= ring_size <= 17:
            continue
        atoms = [molecule.GetAtomWithIdx(index) for index in ring]
        if any(atom.GetIsAromatic() for atom in atoms):
            continue
        carbon_fraction = sum(atom.GetSymbol() == "C" for atom in atoms) / ring_size
        heteroatoms = sum(atom.GetSymbol() in {"N", "O", "S"} for atom in atoms)
        unsupported = any(atom.GetSymbol() not in {"C", "N", "O", "S"} for atom in atoms)
        if carbon_fraction < 0.8 or heteroatoms > 2 or unsupported:
            continue
        candidate = (ring_size, carbon_fraction, heteroatoms)
        if best is None or candidate[1] > best[1]:
            best = candidate
    if best is None:
        return False, None, 0.0, 0
    return True, best[0], best[1], best[2]


def _descriptors(molecule: Chem.Mol) -> Dict[str, float]:
    return {
        "exact_mw": float(Descriptors.ExactMolWt(molecule)),
        "log_p": float(Descriptors.MolLogP(molecule)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "heavy_atoms": float(molecule.GetNumHeavyAtoms()),
        "formal_charge": float(Chem.GetFormalCharge(molecule)),
        "radical_electrons": float(
            sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
        ),
        "fragment_count": float(len(Chem.GetMolFrags(molecule))),
        "sa_score": float(sascorer.calculateScore(molecule)),
    }


def screen_molecule(molecule: Optional[Chem.Mol]) -> ChemicalScreenResult:
    """Apply hard exclusions, profile bounds and a separate review queue."""
    if molecule is None:
        return _empty_result("PARSE_OR_SANITIZE_ERROR")

    try:
        checked = Chem.Mol(molecule)
        Chem.SanitizeMol(checked)
        descriptors = _descriptors(checked)
    except Exception:
        return _empty_result("PARSE_OR_SANITIZE_ERROR")

    is_macrocycle, ring_size, carbon_fraction, heteroatoms = macrocycle_metadata(checked)
    reject_codes: List[str] = []
    review_codes: List[str] = []
    alerts: List[str] = []

    if descriptors["fragment_count"] != 1:
        reject_codes.append("MULTIPLE_FRAGMENTS")
    if descriptors["formal_charge"] != 0:
        reject_codes.append("NONZERO_FORMAL_CHARGE")
    if descriptors["radical_electrons"] != 0:
        reject_codes.append("RADICAL")

    elements = {atom.GetSymbol() for atom in checked.GetAtoms()}
    if not elements.issubset(ALLOWED_ELEMENTS):
        reject_codes.append("UNSUPPORTED_ELEMENT")

    for code, pattern in _HARD_ALERT_PATTERNS.items():
        if pattern is not None and checked.HasSubstructMatch(pattern):
            reject_codes.append(code)
            alerts.append(code)

    if descriptors["sa_score"] > 7.0:
        reject_codes.append("SA_ABOVE_7")

    if is_macrocycle:
        profile_bounds = (
            (descriptors["exact_mw"] <= 330.0, "MW_OUT_OF_RANGE"),
            (-1.0 <= descriptors["log_p"] <= 7.0, "LOGP_OUT_OF_RANGE"),
            (descriptors["tpsa"] <= 90.0, "TPSA_OUT_OF_RANGE"),
            (descriptors["heavy_atoms"] <= 30.0, "HEAVY_ATOMS_OUT_OF_RANGE"),
            (descriptors["exact_mw"] >= 30.0, "MW_OUT_OF_RANGE"),
        )
    else:
        profile_bounds = (
            (30.0 <= descriptors["exact_mw"] <= 300.0, "MW_OUT_OF_RANGE"),
            (-1.0 <= descriptors["log_p"] <= 6.5, "LOGP_OUT_OF_RANGE"),
            (descriptors["tpsa"] <= 80.0, "TPSA_OUT_OF_RANGE"),
            (descriptors["rotatable_bonds"] <= 15.0, "ROTATABLE_BONDS_OUT_OF_RANGE"),
            (descriptors["heavy_atoms"] <= 25.0, "HEAVY_ATOMS_OUT_OF_RANGE"),
        )
    reject_codes.extend(code for allowed, code in profile_bounds if not allowed)

    if reject_codes:
        return ChemicalScreenResult(
            decision=ChemicalDecision.REJECT,
            reason_codes=tuple(dict.fromkeys(reject_codes)),
            descriptors=descriptors,
            is_macrocycle=is_macrocycle,
            macrocycle_ring_size=ring_size,
            macrocycle_carbon_fraction=carbon_fraction,
            macrocycle_heteroatoms=heteroatoms,
            alerts=tuple(dict.fromkeys(alerts)),
        )

    if 5.0 <= descriptors["sa_score"] <= 7.0:
        review_codes.append("SA_REVIEW_RANGE")
    if elements.intersection({"F", "Cl", "Br", "I"}):
        review_codes.append("HALOGEN_PRESENT")

    catalog_descriptions = [
        str(match.GetDescription()) for match in _REVIEW_CATALOG.GetMatches(checked)
    ]
    # An isolated double bond is intrinsic to several known macrocyclic musk
    # references. It is not sufficient on its own to force that protected
    # profile into REVIEW; all other BRENK/NIH alerts still apply.
    if is_macrocycle:
        catalog_descriptions = [
            description
            for description in catalog_descriptions
            if description != "isolated_alkene"
        ]
    alerts.extend(catalog_descriptions)
    if catalog_descriptions:
        review_codes.append("BRENK_OR_NIH_ALERT")

    if not is_macrocycle:
        near_bounds = (
            (descriptors["exact_mw"] > 280.0, "MW_NEAR_LIMIT"),
            (descriptors["log_p"] > 5.5, "LOGP_NEAR_LIMIT"),
            (descriptors["tpsa"] > 70.0, "TPSA_NEAR_LIMIT"),
            (descriptors["rotatable_bonds"] > 12.0, "ROTATABLE_BONDS_NEAR_LIMIT"),
        )
        review_codes.extend(code for matched, code in near_bounds if matched)

    decision = ChemicalDecision.REVIEW if review_codes else ChemicalDecision.PASS
    return ChemicalScreenResult(
        decision=decision,
        reason_codes=tuple(dict.fromkeys(review_codes)) or ("PROFILE_ACCEPTED",),
        descriptors=descriptors,
        is_macrocycle=is_macrocycle,
        macrocycle_ring_size=ring_size,
        macrocycle_carbon_fraction=carbon_fraction,
        macrocycle_heteroatoms=heteroatoms,
        alerts=tuple(dict.fromkeys(alerts)),
    )


def _force_field(
    molecule: Chem.Mol,
    conformer_id: int,
    method: str,
):
    if method == "MMFF94s":
        properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94s")
        if properties is None:
            raise ValueError("MMFF94s parameters unavailable")
        force_field = AllChem.MMFFGetMoleculeForceField(
            molecule,
            properties,
            confId=conformer_id,
        )
    else:
        force_field = AllChem.UFFGetMoleculeForceField(molecule, confId=conformer_id)
    if force_field is None:
        raise ValueError(f"{method} force field unavailable")
    force_field.Initialize()
    return force_field


def _optimize_conformer(
    molecule: Chem.Mol,
    conformer_id: int,
    method: str,
) -> Tuple[int, float]:
    """Optimize one conformer and expose the RDKit convergence status."""
    force_field = _force_field(molecule, conformer_id, method)
    status = int(force_field.Minimize(maxIts=1000))
    if status != 0:
        status = int(force_field.Minimize(maxIts=4000))
    energy = float(force_field.CalcEnergy())
    return status, energy


def _heavy_atom_clash(molecule: Chem.Mol, conformer_id: int) -> bool:
    conformer = molecule.GetConformer(conformer_id)
    heavy_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    for offset, first in enumerate(heavy_indices):
        first_point = conformer.GetAtomPosition(first)
        for second in heavy_indices[offset + 1 :]:
            if molecule.GetBondBetweenAtoms(first, second) is not None:
                continue
            second_point = conformer.GetAtomPosition(second)
            distance = first_point.Distance(second_point)
            if not math.isfinite(distance) or distance < MIN_HEAVY_ATOM_DISTANCE:
                return True
    return False


def _coordinates_are_finite(molecule: Chem.Mol, conformer_id: int) -> bool:
    conformer = molecule.GetConformer(conformer_id)
    for atom_index in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(atom_index)
        if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
            return False
    return True


def _round_trip_matches(
    molecule: Chem.Mol,
    conformer_id: int,
    expected_isomeric_smiles: str,
) -> bool:
    molblock = Chem.MolToMolBlock(molecule, confId=conformer_id)
    round_trip = Chem.MolFromMolBlock(
        molblock,
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if round_trip is None:
        return False
    try:
        heavy = Chem.RemoveHs(round_trip)
        Chem.AssignStereochemistry(heavy, cleanIt=True, force=True)
        return canonical_isomeric_smiles(heavy) == expected_isomeric_smiles
    except Exception:
        return False


def _validated_energies(
    embedded: Chem.Mol,
    conformer_ids: Sequence[int],
    method: str,
    expected_isomeric_smiles: str,
) -> Tuple[Chem.Mol, List[Tuple[float, int]]]:
    """Optimize a whole ensemble with one force field and keep valid minima only."""
    working = Chem.Mol(embedded)
    energies: List[Tuple[float, int]] = []
    for conformer_id in conformer_ids:
        try:
            status, energy = _optimize_conformer(working, conformer_id, method)
        except Exception:
            continue
        if status != 0 or not math.isfinite(energy):
            continue
        if not _coordinates_are_finite(working, conformer_id):
            continue
        if _heavy_atom_clash(working, conformer_id):
            continue
        if not _round_trip_matches(
            working,
            conformer_id,
            expected_isomeric_smiles,
        ):
            continue
        energies.append((energy, conformer_id))
    return working, energies


def _single_conformer_molecule(molecule: Chem.Mol, conformer_id: int) -> Chem.Mol:
    single = Chem.Mol(molecule)
    for current in [conformer.GetId() for conformer in single.GetConformers()]:
        if current != conformer_id:
            single.RemoveConformer(current)
    return single


def _heavy_atom_rmsd(
    molecule: Chem.Mol,
    first_conformer_id: int,
    second_conformer_id: int,
) -> float:
    first = Chem.RemoveHs(_single_conformer_molecule(molecule, first_conformer_id))
    second = Chem.RemoveHs(_single_conformer_molecule(molecule, second_conformer_id))
    return float(rdMolAlign.GetBestRMS(first, second))


def _cluster_representatives(
    molecule: Chem.Mol,
    energies: Sequence[Tuple[float, int]],
    threshold: float,
) -> List[Tuple[float, int]]:
    """Greedily retain the lowest-energy member of distinct RMSD clusters."""
    representatives: List[Tuple[float, int]] = []
    for energy, conformer_id in sorted(energies, key=lambda item: (item[0], item[1])):
        if all(
            _heavy_atom_rmsd(molecule, conformer_id, existing_id) >= threshold
            for _, existing_id in representatives
        ):
            representatives.append((energy, conformer_id))
        if len(representatives) >= MAX_ENSEMBLE_SIZE:
            break
    return representatives


def _empty_ensemble(
    *,
    requested_count: int,
    embedded_count: int = 0,
    converged_count: int = 0,
    is_macrocycle: bool = False,
    error: str,
) -> ConformerEnsembleResult:
    return ConformerEnsembleResult(
        conformers=(),
        method=None,
        requested_count=requested_count,
        embedded_count=embedded_count,
        converged_count=converged_count,
        is_macrocycle=is_macrocycle,
        error=error,
    )


@lru_cache(maxsize=CONFORMER_CACHE_SIZE)
def _build_conformer_ensemble_cached(
    isomeric_smiles: str,
) -> ConformerEnsembleResult:
    """Build and validate a deterministic, force-field-consistent ensemble."""
    requested_count = NORMAL_CONFORMER_COUNT
    try:
        molecule = Chem.MolFromSmiles(isomeric_smiles)
        if molecule is None:
            raise ValueError("SMILES could not be parsed")
        if has_unresolved_stereo(molecule):
            raise ValueError("STEREO_UNRESOLVED")
        expected_smiles = canonical_isomeric_smiles(molecule)
        is_macrocycle, _, _, _ = macrocycle_metadata(molecule)
        requested_count = (
            MACROCYCLE_CONFORMER_COUNT if is_macrocycle else NORMAL_CONFORMER_COUNT
        )
        working = Chem.AddHs(molecule)
        params = AllChem.ETKDGv3()
        params.randomSeed = CONFORMER_RANDOM_SEED
        params.enforceChirality = True
        params.pruneRmsThresh = 0.5
        params.useMacrocycleTorsions = True
        if hasattr(params, "useMacrocycle14config"):
            params.useMacrocycle14config = True
        conformer_ids = list(
            AllChem.EmbedMultipleConfs(
                working,
                numConfs=requested_count,
                params=params,
            )
        )
        if not conformer_ids:
            raise ValueError("ETKDGv3 did not produce a conformer")

        optimized, energies = _validated_energies(
            working,
            conformer_ids,
            "MMFF94s",
            expected_smiles,
        )
        method = "MMFF94s"
        if not energies:
            optimized, energies = _validated_energies(
                working,
                conformer_ids,
                "UFF",
                expected_smiles,
            )
            method = "UFF"
        if not energies:
            return _empty_ensemble(
                requested_count=requested_count,
                embedded_count=len(conformer_ids),
                is_macrocycle=is_macrocycle,
                error="NO_CONVERGED_VALID_CONFORMER",
            )

        threshold = (
            MACROCYCLE_CLUSTER_RMSD if is_macrocycle else NORMAL_CLUSTER_RMSD
        )
        representatives = _cluster_representatives(optimized, energies, threshold)
        minimum_energy = representatives[0][0]
        records = tuple(
            ConformerRecord(
                molblock=Chem.MolToMolBlock(optimized, confId=conformer_id),
                relative_energy=max(0.0, float(energy - minimum_energy)),
            )
            for energy, conformer_id in representatives
        )
        return ConformerEnsembleResult(
            conformers=records,
            method=method,
            requested_count=requested_count,
            embedded_count=len(conformer_ids),
            converged_count=len(energies),
            is_macrocycle=is_macrocycle,
        )
    except Exception as error:
        return _empty_ensemble(
            requested_count=requested_count,
            error=str(error),
        )


def build_conformer_ensemble(isomeric_smiles: str) -> ConformerEnsembleResult:
    """Serialize conformer work so concurrent requests cannot saturate the CPU."""
    with _CONFORMER_LOCK:
        return _build_conformer_ensemble_cached(isomeric_smiles)


def clear_conformer_cache() -> None:
    """Expose deterministic cache reset for tests and controlled reloads."""
    with _CONFORMER_LOCK:
        _build_conformer_ensemble_cached.cache_clear()
