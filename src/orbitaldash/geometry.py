"""Geometry primitives and utilities.

This module defines a minimal `Molecule` container and helpers for normalizing
user-provided geometry inputs to a consistent internal representation.

Conventions:
- Coordinates are in Angstrom for all internal computations.
- Element metadata is provided via `ELEMENT_COLORS` and `COVALENT_RADII`.

Color/radius sources:
- Colors are chosen to match the widely used Jmol/CPK palette where possible.
- Covalent radii (Å) are approximations primarily based on Cordero et al.
    (2011, Dalton Trans., 40, 1919–1925) and common visualization defaults.
    They are meant for bond inference and visual sizing, not for high-precision.

Notes:
    The supported symbol/Z maps are intentionally small to keep this package
    lightweight. Extend the dictionaries if you need additional elements.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, overload, Literal


def _estimate_bond_order(_zi: int, _zj: int, distance: float, ri: float, rj: float, max_order: int = 3) -> int:
    """Estimate a bond order (1/2/3) from interatomic distance.

    This is a heuristic intended for visualization only.

    The classifier uses the ratio r = d / (ri + rj). Roughly:
      - r < 0.85 -> triple
      - r < 0.95 -> double
      - else -> single

    Args:
        zi, zj: Atomic numbers.
        distance: Interatomic distance in Angstrom.
        ri, rj: Covalent radii used for baseline (Angstrom).
        max_order: Maximum bond order to return.

    Returns:
        int: Estimated bond order in [1, max_order].
    """
    denom = float(ri + rj)
    if denom <= 1e-12:
        return 1
    ratio = float(distance) / denom
    order = 1
    if ratio < 0.85:
        order = 3
    elif ratio < 0.95:
        order = 2
    if max_order < 1:
        return 1
    return int(max(1, min(int(max_order), order)))


# Basic element data (subset extended with common elements)
# Colors aim to follow Jmol/CPK defaults for familiarity.
ELEMENT_COLORS = {
    1: '#FFFFFF',   # H
    2: '#D9FFFF',   # He
    3: '#CC80FF',   # Li
    4: '#C2FF00',   # Be
    5: '#FFB5B5',   # B
    6: '#909090',   # C
    7: '#3050F8',   # N
    8: '#FF0D0D',   # O
    9: '#90E050',   # F
    10: '#B3E3F5',  # Ne
    11: '#AB5CF2',  # Na
    12: '#8AFF00',  # Mg
    13: '#BFA6A6',  # Al
    14: '#F0C8A0',  # Si
    15: '#FF8000',  # P
    16: '#FFFF30',  # S
    17: '#1FF01F',  # Cl
    18: '#80D1E3',  # Ar
    19: '#8F40D4',  # K
    20: '#3DFF00',  # Ca
    35: '#A62929',  # Br
    53: '#940094',  # I
}

COVALENT_RADII = {  # rough covalent radii in Angstrom (Cordero et al.)
    1: 0.31,  # H
    # He (formal covalent radius is not meaningful; small placeholder)
    2: 0.28,
    3: 1.28,  # Li
    4: 0.96,  # Be
    5: 0.84,  # B
    6: 0.76,  # C
    7: 0.71,  # N
    8: 0.66,  # O
    9: 0.57,  # F
    10: 0.58,  # Ne
    11: 1.66,  # Na
    12: 1.41,  # Mg
    13: 1.21,  # Al
    14: 1.11,  # Si
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    18: 1.06,  # Ar
    19: 2.03,  # K
    20: 1.76,  # Ca
    35: 1.20,  # Br
    53: 1.39,  # I
}

# Basic symbol <-> atomic number mapping (extended as needed)
_SYMBOL_TO_Z = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Br': 35,
    'I': 53,
}
_Z_TO_SYMBOL = {v: k for k, v in _SYMBOL_TO_Z.items()}


@dataclass
class Molecule:
    """Simple molecule container.

    Attributes:
        symbols: List of element symbols (e.g., ["O", "H", "H"]).
        coords: Cartesian coordinates in Angstrom with shape (N, 3).

    Notes:
        This class keeps only what is necessary for visualization and simple
        bond inference. For electronic structure data, see orbitals module.
    """
    symbols: List[str]
    coords: np.ndarray  # shape (N, 3), Angstrom

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Atomic numbers derived from element symbols.

        Returns:
            np.ndarray: Array of shape (N,) with integer atomic numbers.

        Raises:
            ValueError: If an unknown element symbol is encountered.
        """
        Z = []
        for s in self.symbols:
            if s not in _SYMBOL_TO_Z:
                raise ValueError(f"Unknown element symbol: {s}")
            Z.append(_SYMBOL_TO_Z[s])
        return np.array(Z, dtype=int)

    @overload
    def infer_bonds(self, scale: float = 1.2, *,
                    with_order: Literal[False] = False) -> List[Tuple[int, int]]: ...

    @overload
    def infer_bonds(self, scale: float = 1.2, *,
                    with_order: Literal[True], max_order: int = 3) -> List[Tuple[int, int, int]]: ...

    def infer_bonds(self, scale: float = 1.2, *, with_order: bool = False, max_order: int = 3):
        """Infer covalent bonds using radii and a distance cutoff.

        The cutoff for two atoms i and j is computed as:
            cutoff = scale * (r_i + r_j)
        where radii are looked up in `COVALENT_RADII` (fallback 0.7 Å).

        Args:
            scale: Multiplicative factor applied to the sum of covalent radii.
            with_order: If True, return (i, j, order) tuples where order is a
                rough estimate of bond multiplicity (1/2/3) based on distance.
            max_order: Maximum bond order to return when ``with_order`` is True.

        Returns:
            If ``with_order`` is False (default):
                List[Tuple[int, int]] of pairs (i, j) with i < j.
            If ``with_order`` is True:
                List[Tuple[int, int, int]] of (i, j, order).
        """
        Z = self.atomic_numbers
        bonds: list = []
        for i in range(len(Z)):
            for j in range(i+1, len(Z)):
                ri = COVALENT_RADII.get(int(Z[i]), 0.7)
                rj = COVALENT_RADII.get(int(Z[j]), 0.7)
                cutoff = scale * (ri + rj)
                d = float(np.linalg.norm(self.coords[i] - self.coords[j]))
                if d <= cutoff:
                    if with_order:
                        order = _estimate_bond_order(int(Z[i]), int(
                            Z[j]), d, ri, rj, max_order=int(max_order))
                        bonds.append((i, j, order))
                    else:
                        bonds.append((i, j))
        return bonds

    def to_dict(self):
        """Serialize to a plain dictionary.

        Returns:
            dict: Dictionary with keys ``symbols`` and ``coords``.
        """
        return {"symbols": self.symbols, "coords": self.coords}


def standardize_geometry(geometry: Union[dict, Tuple[Sequence[int], np.ndarray], Tuple[Sequence[str], np.ndarray], Molecule]) -> Molecule:
    """Normalize geometry inputs to a :class:`Molecule`.

    Accepted forms:

    - Molecule instance (returned as-is)
    - dict with keys ``symbols`` (list[str]) and ``coords`` (array (N, 3))
    - tuple of (Z, coords) where Z are atomic numbers (Sequence[int])
    - tuple of (symbols, coords) where symbols are element strings

    Args:
        geometry: Geometry in one of the accepted formats.

    Returns:
        Molecule: Standardized molecule with Angstrom coordinates.

    Raises:
        ValueError: If the input dict is missing required keys or symbols are
            unknown.
    """
    if isinstance(geometry, Molecule):
        return geometry
    if isinstance(geometry, dict):
        symbols = geometry.get('symbols')
        if symbols is None:
            raise ValueError("geometry dict must contain 'symbols'")
        coords = np.asarray(geometry.get('coords'), dtype=float)
        return Molecule(symbols=list(symbols), coords=coords)
    symbols_or_Z, coords = geometry  # type: ignore
    coords = np.asarray(coords, dtype=float)
    # atomic numbers
    if len(symbols_or_Z) and isinstance(symbols_or_Z[0], int):
        # type: ignore[arg-type]
        symbols = [_Z_TO_SYMBOL[int(z)] for z in symbols_or_Z]
    else:
        symbols = [str(s) for s in symbols_or_Z]
    return Molecule(symbols=symbols, coords=coords)
