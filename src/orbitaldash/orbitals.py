import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable

from .geometry import Molecule, standardize_geometry

# Unit conversion constants
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG


@dataclass
class OrbitalData:
    grid: np.ndarray        # shape (npoints, 3)
    values: np.ndarray      # shape (npoints,) for one MO
    origin: np.ndarray      # min corner of grid box
    steps: Tuple[int, int, int]
    spacing: float


def make_grid(mol: Molecule, padding: float = 3.0, spacing: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
    mins = mol.coords.min(axis=0) - padding
    maxs = mol.coords.max(axis=0) + padding
    lengths = maxs - mins
    # type: ignore[var-annotated]
    steps = tuple(int(np.ceil(l/spacing))+1 for l in lengths)
    steps = (int(steps[0]), int(steps[1]), int(steps[2]))  # ensure 3-tuple
    xs = np.linspace(mins[0], maxs[0], steps[0])
    ys = np.linspace(mins[1], maxs[1], steps[1])
    zs = np.linspace(mins[2], maxs[2], steps[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return grid, mins, steps, spacing


def build_pyscf_mol(mol: Molecule, basis_or_mol: Any) -> Any:
    try:
        from pyscf import gto  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PySCF not available; install pyscf or supply custom evaluator") from e
    # If a PySCF mol-like object is provided, use it directly
    if hasattr(basis_or_mol, 'eval_gto') and hasattr(basis_or_mol, 'nao'):
        return basis_or_mol
    # Otherwise, treat as basis spec string and build a new molecule
    basis = basis_or_mol
    atom_str = "".join(f"{s} {x} {y} {z};" for s, (x, y, z)
                       in zip(mol.symbols, mol.coords))
    return gto.M(atom=atom_str, basis=basis, unit='Angstrom', verbose=0)


def evaluate_mos_on_grid(mol: Molecule, basis_or_mol: Any, mo_coeff: np.ndarray, grid: np.ndarray) -> np.ndarray:
    pyscf_mol = build_pyscf_mol(mol, basis_or_mol)
    # PySCF expects evaluation coordinates in Bohr. Our grid is in Angstrom.
    grid_bohr = grid * ANG_TO_BOHR
    # Try spherical first, then cart if dimensions don't match mo_coeff
    ao_vals = pyscf_mol.eval_gto('GTOval_sph', grid_bohr)
    if ao_vals.shape[1] != mo_coeff.shape[0]:
        # Try cartesian representation
        ao_vals_cart = pyscf_mol.eval_gto('GTOval', grid_bohr)
        if ao_vals_cart.shape[1] == mo_coeff.shape[0]:
            ao_vals = ao_vals_cart
        else:
            raise ValueError(
                f"AO/MO shape mismatch: AO(sph)={ao_vals.shape[1]}, AO(cart)={ao_vals_cart.shape[1]} "
                f"but mo_coeff rows={mo_coeff.shape[0]}. This often happens if evaluation basis differs "
                f"(e.g., uncontracted vs contracted) from the SCF basis used for mo_coeff. Pass the original "
                f"PySCF mol object used to compute mo_coeff as the 'basis' argument to OrbitalEvaluator/launch_app "
                f"to guarantee consistency."
            )
    # mo_coeff shape (n_ao, n_mo) -> values (npoints, n_mo)
    return ao_vals @ mo_coeff


def evaluate_mo_on_grid_single(mol: Molecule, basis_or_mol: Any, mo_vec: np.ndarray, grid: np.ndarray, chunk_points: int = 200_000) -> np.ndarray:
    """Evaluate a single MO on the grid efficiently.

    This avoids forming all MO values (npoints, n_mo) and instead evaluates AO values
    in chunks and contracts with the selected MO vector only. This reduces memory and
    extra compute when n_mo is large.
    """
    pyscf_mol = build_pyscf_mol(mol, basis_or_mol)
    mo_vec = np.asarray(mo_vec).reshape(-1)
    grid_bohr = grid * ANG_TO_BOHR

    npts = grid_bohr.shape[0]
    out = np.empty(npts, dtype=float)
    mode = None  # 'sph' or 'cart'

    start = 0
    while start < npts:
        stop = min(start + chunk_points, npts)
        chunk = grid_bohr[start:stop]
        if mode is None:
            ao = pyscf_mol.eval_gto('GTOval_sph', chunk)
            if ao.shape[1] == mo_vec.shape[0]:
                mode = 'sph'
            else:
                ao_cart = pyscf_mol.eval_gto('GTOval', chunk)
                if ao_cart.shape[1] == mo_vec.shape[0]:
                    mode = 'cart'
                    ao = ao_cart
                else:
                    raise ValueError(
                        f"AO/MO shape mismatch: AO(sph)={ao.shape[1]}, AO(cart)={ao_cart.shape[1]} "
                        f"but mo_coeff rows={mo_vec.shape[0]}. Pass the original PySCF mol object "
                        f"used to compute mo_coeff as the basis argument."
                    )
        else:
            if mode == 'sph':
                ao = pyscf_mol.eval_gto('GTOval_sph', chunk)
            else:
                ao = pyscf_mol.eval_gto('GTOval', chunk)
        out[start:stop] = ao @ mo_vec
        start = stop
    return out


def get_orbital_isovalue(values: np.ndarray, fraction: float = 0.85) -> float:
    vmax = np.abs(values).max()
    return fraction * vmax


class OrbitalEvaluator:
    def __init__(self, geometry, basis: Any, mo_coeff: np.ndarray, custom_eval: Optional[Callable[[np.ndarray], np.ndarray]] = None, spacing: float = 0.3):
        self.mol = standardize_geometry(geometry)
        self.basis = basis  # may be a basis string or a PySCF mol object
        self.mo_coeff = mo_coeff
        self._cache: Dict[int, np.ndarray] = {}
        self._grid_cache: Optional[Tuple[np.ndarray,
                                         np.ndarray, Tuple[int, int, int], float]] = None
        self._custom_eval = custom_eval
        self._spacing = spacing

    def grid(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
        if self._grid_cache is None:
            self._grid_cache = make_grid(self.mol, spacing=self._spacing)
        return self._grid_cache

    def regrid(self, spacing: float):
        """Change grid spacing (resolution) and clear caches.

        Smaller spacing -> finer grid -> more points -> higher compute cost.
        """
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        if abs(spacing - self._spacing) < 1e-9:
            return  # no change
        self._spacing = spacing
        self._grid_cache = None
        self._cache.clear()

    def mo_values(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx]
        grid, _, _, _ = self.grid()
        if self._custom_eval is not None:
            mo_vals = self._custom_eval(grid)[:, idx]
        else:
            # Efficient path: evaluate only the requested orbital using chunked AO evaluation
            mo_vec = self.mo_coeff[:, idx]
            mo_vals = evaluate_mo_on_grid_single(
                self.mol, self.basis, mo_vec, grid)
        self._cache[idx] = mo_vals
        return mo_vals

    def orbital_data(self, idx: int, isovalue: Optional[float] = None) -> OrbitalData:
        grid, origin, steps, spacing = self.grid()
        vals = self.mo_values(idx)
        if isovalue is None:
            isovalue = get_orbital_isovalue(vals)
        return OrbitalData(grid=grid, values=vals, origin=origin, steps=steps, spacing=spacing)
