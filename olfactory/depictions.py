"""UI-neutral molecular depictions and display descriptors."""

from __future__ import annotations

from typing import Dict

from rdkit import Chem
from rdkit.Chem import rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


def molecule_svg(molecule: Chem.Mol, width: int = 720, height: int = 520) -> str:
    """Return a deterministic RDKit SVG with stereochemical wedge/dash bonds."""
    prepared = Chem.Mol(molecule)
    rdDepictor.Compute2DCoords(prepared)
    prepared = rdMolDraw2D.PrepareMolForDrawing(
        prepared,
        kekulize=True,
        addChiralHs=True,
        wedgeBonds=True,
    )
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.08
    drawer.DrawMolecule(prepared)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def formula(molecule: Chem.Mol) -> str:
    return str(rdMolDescriptors.CalcMolFormula(molecule))


def volatility_tier(exact_mw: float) -> str:
    """Return the existing MW-based display estimate, not experimental evidence."""
    if exact_mw < 150.0:
        return "Top"
    if exact_mw <= 220.0:
        return "Middle"
    return "Base"


def display_descriptors(molecule: Chem.Mol, values: Dict[str, float]) -> Dict[str, object]:
    exact_mw = float(values.get("exact_mw", 0.0))
    return {
        "formula": formula(molecule),
        "exact_mw": exact_mw,
        "log_p": float(values.get("log_p", 0.0)),
        "tpsa": float(values.get("tpsa", 0.0)),
        "rotatable_bonds": int(values.get("rotatable_bonds", 0.0)),
        "heavy_atoms": int(values.get("heavy_atoms", 0.0)),
        "sa_score": float(values.get("sa_score", 0.0)),
        "estimated_volatility_tier": volatility_tier(exact_mw),
        "volatility_basis": "MW-based estimate",
    }
