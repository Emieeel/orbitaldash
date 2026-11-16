"""Grid creation and molecular orbital evaluation utilities.

This module provides:

- Uniform Cartesian grid generation around a molecule.
- Optional PySCF-backed AO/MO evaluation on arbitrary grids.
- A pluggable :class:`OrbitalEvaluator` with caching and adjustable spacing.

Units:
    All geometry and grid coordinates are in Angstrom internally. Conversion
    to Bohr is handled when calling PySCF.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable, Union
import threading
import logging
import time
import json
import os
import contextlib
from pathlib import Path
import hashlib

from .geometry import Molecule, standardize_geometry
from .trajectory import TrajectoryData

# Unit conversion constants
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

logger = logging.getLogger(__name__)


@dataclass
class OrbitalData:
    """Container for orbital data sampled on a grid.

    Attributes:
        grid: Flat array of grid points with shape (P, 3) in Angstrom.
        values: Flattened orbital values with shape (P,) for a single MO.
        origin: Minimum corner (x_min, y_min, z_min) of the grid box.
        steps: Number of steps in each dimension (nx, ny, nz).
        spacing: Grid spacing in Angstrom.
    """
    grid: np.ndarray        # shape (npoints, 3)
    values: np.ndarray      # shape (npoints,) for one MO
    origin: np.ndarray      # min corner of grid box
    steps: Tuple[int, int, int]
    spacing: float


def make_grid(mol: Molecule, padding: float = 3.0, spacing: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
    """Create a 3D Cartesian grid around a molecule.

    The grid spans the bounding box of the molecule extended by ``padding``
    in all directions, with uniform spacing along each axis.

    Args:
        mol: Molecule containing symbols and coordinates (Angstrom).
        padding: Extra space (Å) added around the molecular extents.
        spacing: Distance (Å) between adjacent grid points.

    Returns:
        Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
            - grid: Array of shape (P, 3) with grid point coordinates.
            - origin: Minimum corner of the grid box (3,).
            - steps: Number of steps in each dimension (nx, ny, nz).
            - spacing: The spacing used (Å).
    """
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


def make_grid_from_bounds(mins: np.ndarray, maxs: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
    """Create a Cartesian grid from precomputed bounds."""
    mins = np.asarray(mins, dtype=float)
    maxs = np.asarray(maxs, dtype=float)
    lengths = maxs - mins
    steps = tuple(int(np.ceil(l / spacing)) + 1 for l in lengths)
    steps = (int(steps[0]), int(steps[1]), int(steps[2]))
    xs = np.linspace(mins[0], maxs[0], steps[0])
    ys = np.linspace(mins[1], maxs[1], steps[1])
    zs = np.linspace(mins[2], maxs[2], steps[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    return grid, mins, steps, spacing


def make_global_grid(molecules: list[Molecule], padding: float = 3.0, spacing: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
    """Create one shared grid for all molecules in a trajectory."""
    if not molecules:
        raise ValueError("molecules must not be empty")
    mins = np.min(np.stack([m.coords.min(axis=0) for m in molecules], axis=0), axis=0) - float(padding)
    maxs = np.max(np.stack([m.coords.max(axis=0) for m in molecules], axis=0), axis=0) + float(padding)
    return make_grid_from_bounds(mins, maxs, spacing)


# Coordinates and MO coefficients are hashed after rounding to this many decimals. Re-running the
# same calculation differs at the ~1e-11 level (e.g. threaded BLAS), which must not change the key;
# differences below 1e-5 are invisible once grid values are stored as float32.
_CACHE_KEY_DECIMALS = 5


def _quantized_bytes(x: Any) -> bytes:
    """Shape plus integer-quantized values; integers also avoid the -0.0 vs 0.0 byte mismatch."""
    q = np.rint(np.asarray(x, dtype=np.float64) * 10.0**_CACHE_KEY_DECIMALS).astype(np.int64)
    return f"{q.shape};".encode("utf-8") + q.tobytes()


def trajectory_cache_key(trajectory: TrajectoryData, spacing: float, padding: float = 3.0) -> str:
    """Build a cache key for trajectory + grid settings that is stable under numerical noise."""
    h = hashlib.sha256()
    h.update(f"spacing={float(spacing):.6f};padding={float(padding):.6f};frames={len(trajectory.frames)}".encode("utf-8"))
    for frame in trajectory.frames:
        mol = standardize_geometry(frame.geometry)
        h.update(";".join(mol.symbols).encode("utf-8"))
        h.update(_quantized_bytes(mol.coords))
        h.update(_quantized_bytes(frame.mo_coeff))
    return h.hexdigest()[:16]


@contextlib.contextmanager
def _atomic_output_path(path: Path):
    """Yield a temporary sibling path that is renamed to ``path`` only if the block succeeds.

    An interrupted precompute then never leaves a partial file at ``path`` that later runs would
    load as a valid cache.
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def precompute_trajectory_to_hdf5(
    trajectory: TrajectoryData,
    h5_path: Union[str, Path],
    spacing: float,
    padding: float = 3.0,
    verbose: bool = False,
) -> str:
    """Precompute trajectory orbital volumes on a shared grid and store in HDF5."""
    try:
        import h5py  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("h5py is required for precomputed trajectory mode") from e

    h5_path = str(h5_path)
    molecules = [standardize_geometry(f.geometry) for f in trajectory.frames]
    grid, origin, steps, _ = make_global_grid(molecules, padding=float(padding), spacing=float(spacing))
    grid32 = grid.astype(np.float32, copy=False)

    p = Path(h5_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _atomic_output_path(p) as tmp_path, h5py.File(tmp_path, "w") as h5:
        g_grid = h5.create_group("grid")
        g_grid.create_dataset("points", data=grid32, dtype="f4")
        g_grid.create_dataset("origin", data=np.asarray(origin, dtype=np.float32), dtype="f4")
        g_grid.create_dataset("steps", data=np.asarray(steps, dtype=np.int32), dtype="i4")
        g_grid.attrs["spacing"] = float(spacing)
        g_grid.attrs["padding"] = float(padding)

        g_frames = h5.create_group("frames")
        for i, frame in enumerate(trajectory.frames):
            mol = molecules[i]
            grp = g_frames.create_group(f"{i:04d}")
            mo_coeff = np.asarray(frame.mo_coeff)
            n_mo = int(mo_coeff.shape[1])
            dset = grp.create_dataset(
                "values",
                shape=(n_mo, grid32.shape[0]),
                dtype="f4",
                chunks=(1, min(grid32.shape[0], 262144)),
                compression="lzf",
                shuffle=True,
            )
            if frame.custom_eval is not None:
                vals = np.asarray(frame.custom_eval(grid), dtype=np.float32)
                if vals.shape != (grid32.shape[0], n_mo):
                    raise ValueError(
                        f"custom_eval for frame {i} returned {vals.shape}, expected {(grid32.shape[0], n_mo)}"
                    )
                dset[:, :] = vals.T
            else:
                for mo_idx in range(n_mo):
                    vec = mo_coeff[:, mo_idx]
                    vals = evaluate_mo_on_grid_single(mol, frame.basis, vec, grid)
                    dset[mo_idx, :] = np.asarray(vals, dtype=np.float32)

            if frame.mo_energies is not None:
                grp.create_dataset("mo_energies", data=np.asarray(frame.mo_energies, dtype=np.float32), dtype="f4")
            if frame.mo_occ is not None:
                grp.create_dataset("mo_occ", data=np.asarray(frame.mo_occ, dtype=np.float32), dtype="f4")
            grp.attrs["symbols"] = json.dumps(mol.symbols)
            grp.create_dataset("coords", data=np.asarray(mol.coords, dtype=np.float32), dtype="f4")
            grp.attrs["metadata"] = json.dumps(frame.metadata)

            if verbose:
                logger.info("Precomputed frame %d/%d", i + 1, len(trajectory.frames))

        h5.attrs["n_frames"] = int(len(trajectory.frames))
        h5.attrs["cache_key"] = trajectory_cache_key(trajectory, spacing=float(spacing), padding=float(padding))
    return h5_path


def trajectory_hdf5_is_complete(h5_path: Union[str, Path], n_frames: int) -> bool:
    """Whether ``h5_path`` is a readable, fully written cache with ``n_frames`` frames."""
    if not Path(h5_path).exists():
        return False
    try:
        import h5py  # type: ignore

        with h5py.File(str(h5_path), "r") as h5:
            # n_frames is written last, so files cut off by older versions lack it.
            finished = int(h5.attrs.get("n_frames", -1)) == int(n_frames)
            return finished and len(h5["frames"]) == int(n_frames)
    except Exception:  # truncated or not an HDF5 file
        return False


def load_trajectory_hdf5_grid(h5_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
    """Load shared trajectory grid from a precomputed HDF5 file."""
    try:
        import h5py  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("h5py is required for precomputed trajectory mode") from e

    with h5py.File(str(h5_path), "r") as h5:
        grid = np.asarray(h5["grid/points"][:], dtype=float)
        origin = np.asarray(h5["grid/origin"][:], dtype=float)
        steps_arr = np.asarray(h5["grid/steps"][:], dtype=int)
        steps = (int(steps_arr[0]), int(steps_arr[1]), int(steps_arr[2]))
        spacing = float(h5["grid"].attrs["spacing"])
    return grid, origin, steps, spacing


def load_trajectory_hdf5_values(h5_path: Union[str, Path], frame_idx: int, orbital_idx: int) -> np.ndarray:
    """Load one frame/orbital vector from precomputed HDF5."""
    try:
        import h5py  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("h5py is required for precomputed trajectory mode") from e

    with h5py.File(str(h5_path), "r") as h5:
        return np.asarray(h5[f"frames/{int(frame_idx):04d}/values"][int(orbital_idx), :], dtype=float)


def build_pyscf_mol(mol: Molecule, basis_or_mol: Any) -> Any:
    """Construct or reuse a PySCF molecule object.

    If ``basis_or_mol`` already has PySCF-like attributes (``eval_gto`` and
    ``nao``), it is returned as-is. Otherwise, a new ``gto.M`` is built using
    the provided molecule and basis specification.

    Args:
        mol: Molecule with symbols and Angstrom coordinates.
        basis_or_mol: Either a basis string or a PySCF molecule-like object.

    Returns:
        Any: A PySCF molecule object compatible with ``eval_gto``.

    Raises:
        RuntimeError: If PySCF is not available in the environment.
    """
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
    """Evaluate all molecular orbitals on a grid using PySCF.

    Args:
        mol: Molecule used to build the PySCF object when needed.
        basis_or_mol: Basis specification or a PySCF mol-like object.
        mo_coeff: MO coefficient matrix with shape (n_ao, n_mo).
        grid: Grid points (P, 3) in Angstrom.

    Returns:
        np.ndarray: MO values on the grid with shape (P, n_mo).

    Raises:
        ValueError: If AO and MO dimensions are incompatible between spherical
            and cartesian representations.
    """
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
    """Evaluate a single MO on the grid efficiently using PySCF.

    This avoids forming all MO values (P, n_mo). Instead, it evaluates AO values
    in chunks and contracts with the selected MO vector only. This reduces memory
    and compute when ``n_mo`` is large.

    Args:
        mol: Molecule used to build the PySCF object when needed.
        basis_or_mol: Basis specification or a PySCF mol-like object.
        mo_vec: Single MO coefficient vector with shape (n_ao,).
        grid: Grid points (P, 3) in Angstrom.
        chunk_points: Maximum number of grid points per AO evaluation batch.

    Returns:
        np.ndarray: Flattened MO values on the grid with shape (P,).

    Raises:
        ValueError: If AO and MO dimensions are incompatible between spherical
            and cartesian representations.
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
    """Compute an isovalue as a fraction of the maximum absolute value.

    Args:
        values: Flattened MO values on the grid.
        fraction: Fraction of ``max(|values|)`` used to define the isovalue.

    Returns:
        float: The isovalue.
    """
    vmax = np.abs(values).max()
    return fraction * vmax


class OrbitalEvaluator:
    """Evaluate and cache molecular orbital values on a grid.

    Two evaluation modes are supported:

    1. PySCF-backed evaluation based on ``geometry`` and ``basis``.
    2. A custom evaluator callable ``custom_eval(grid) -> (P, n_mo)``.

    Grid spacing can be adjusted via :meth:`regrid`, which invalidates caches.

    Args:
        geometry: Geometry in any format accepted by ``standardize_geometry``.
        basis: Basis string or PySCF mol-like object; ignored if ``custom_eval``
            is provided.
        mo_coeff: MO coefficient matrix with shape (n_ao, n_mo).
        custom_eval: Optional callable mapping grid (P, 3) to MO values
            (P, n_mo).
        spacing: Initial grid spacing in Angstrom.
    """

    def __init__(
        self,
        geometry,
        basis: Any,
        mo_coeff: np.ndarray,
        custom_eval: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        spacing: float = 0.3,
        verbose: bool = False,
    ):
        self.mol = standardize_geometry(geometry)
        self.basis = basis  # may be a basis string or a PySCF mol object
        self.mo_coeff = mo_coeff
        self._cache: Dict[int, np.ndarray] = {}
        self._grid_cache: Optional[Tuple[np.ndarray,
                                         np.ndarray, Tuple[int, int, int], float]] = None
        self._custom_eval = custom_eval
        self._spacing = spacing
        self._verbose = bool(verbose)
        # Dash callbacks may be processed concurrently (threaded Flask server).
        # Guard grid/caches so we never mix a grid snapshot with values from another.
        self._lock = threading.RLock()

    def grid(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
        """Get or build the current grid.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], float]:
                - grid: Array of shape (P, 3) with grid point coordinates.
                - origin: Minimum corner of the grid box (3,).
                - steps: Number of steps in each dimension (nx, ny, nz).
                - spacing: The spacing used (Å).
        """
        with self._lock:
            if self._grid_cache is None:
                t0 = time.perf_counter()
                self._grid_cache = make_grid(self.mol, spacing=self._spacing)
                if self._verbose:
                    grid, _origin, steps, spacing = self._grid_cache
                    logger.info(
                        "Grid built: spacing=%.3f Å steps=%s points=%d (%.2f ms)",
                        float(spacing),
                        steps,
                        int(grid.shape[0]),
                        1e3 * (time.perf_counter() - t0),
                    )
            return self._grid_cache

    def regrid(self, spacing: float):
        """Change grid spacing (resolution) and clear caches.

        Smaller spacing -> finer grid -> more points -> higher compute cost.

        Args:
            spacing: New grid spacing in Angstrom (must be positive).

        Raises:
            ValueError: If ``spacing`` is not positive.
        """
        with self._lock:
            if spacing <= 0:
                raise ValueError("spacing must be positive")
            if abs(spacing - self._spacing) < 1e-9:
                return  # no change
            if self._verbose:
                logger.info("Regrid: %.3f Å -> %.3f Å",
                            float(self._spacing), float(spacing))
            self._spacing = spacing
            self._grid_cache = None
            self._cache.clear()

    def mo_values(self, idx: int) -> np.ndarray:
        """Get MO values for the selected index on the current grid.

        Uses cached values when available; otherwise evaluates using either the
        custom evaluator or PySCF.

        Args:
            idx: Orbital index (0-based).

        Returns:
            np.ndarray: Flattened values on the grid with shape (P,).
        """
        with self._lock:
            if idx in self._cache:
                return self._cache[idx]
            grid, _, _, _ = self.grid()
            t0 = time.perf_counter()
            if self._custom_eval is not None:
                mo_vals = self._custom_eval(grid)[:, idx]
            else:
                # Efficient path: evaluate only the requested orbital using chunked AO evaluation
                mo_vec = self.mo_coeff[:, idx]
                mo_vals = evaluate_mo_on_grid_single(
                    self.mol, self.basis, mo_vec, grid)
            self._cache[idx] = mo_vals
            if self._verbose:
                logger.info(
                    "MO %d evaluated: points=%d max|psi|=%.6g (%.2f ms)",
                    int(idx),
                    int(mo_vals.shape[0]),
                    float(np.max(np.abs(mo_vals))) if mo_vals.size else 0.0,
                    1e3 * (time.perf_counter() - t0),
                )
            return mo_vals

    def orbital_data(self, idx: int, isovalue: Optional[float] = None) -> OrbitalData:
        """Construct an :class:`OrbitalData` bundle for a given orbital.

        Args:
            idx: Orbital index (0-based).
            isovalue: Optional isovalue; if not provided, a default fraction
                of the maximum absolute value is used.

        Returns:
            OrbitalData: Grid, values, origin, steps, and spacing.
        """
        # Ensure grid + values are from the same snapshot.
        with self._lock:
            grid, origin, steps, spacing = self.grid()
            vals = self.mo_values(idx)
            if vals.shape[0] != grid.shape[0]:
                # Defensive recovery: clear this orbital cache and recompute.
                self._cache.pop(idx, None)
                vals = self.mo_values(idx)
            if isovalue is None:
                isovalue = get_orbital_isovalue(vals)
            return OrbitalData(grid=grid, values=vals, origin=origin, steps=steps, spacing=spacing)
