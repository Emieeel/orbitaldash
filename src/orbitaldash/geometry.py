import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

# Basic element data (subset)
ELEMENT_COLORS = {
    1: '#FFFFFF',  # H
    6: '#909090',  # C
    7: '#3050F8',  # N
    8: '#FF0D0D',  # O
}

COVALENT_RADII = {  # rough covalent radii in Angstrom
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
}

# Basic symbol <-> atomic number mapping (extend as needed)
_SYMBOL_TO_Z = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
}
_Z_TO_SYMBOL = {v: k for k, v in _SYMBOL_TO_Z.items()}


@dataclass
class Molecule:
    symbols: List[str]
    coords: np.ndarray  # shape (N, 3), Angstrom

    @property
    def atomic_numbers(self) -> np.ndarray:
        Z = []
        for s in self.symbols:
            if s not in _SYMBOL_TO_Z:
                raise ValueError(f"Unknown element symbol: {s}")
            Z.append(_SYMBOL_TO_Z[s])
        return np.array(Z, dtype=int)

    def infer_bonds(self, scale: float = 1.2) -> List[Tuple[int, int]]:
        Z = self.atomic_numbers
        bonds = []
        for i in range(len(Z)):
            for j in range(i+1, len(Z)):
                ri = COVALENT_RADII.get(int(Z[i]), 0.7)
                rj = COVALENT_RADII.get(int(Z[j]), 0.7)
                cutoff = scale * (ri + rj)
                if np.linalg.norm(self.coords[i] - self.coords[j]) <= cutoff:
                    bonds.append((i, j))
        return bonds

    def to_dict(self):
        return {"symbols": self.symbols, "coords": self.coords}


def standardize_geometry(geometry: Union[dict, Tuple[Sequence[int], np.ndarray], Tuple[Sequence[str], np.ndarray], Molecule]) -> Molecule:
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
