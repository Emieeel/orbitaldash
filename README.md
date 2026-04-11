# orbitaldash

A lightweight Python package providing a Dash + Plotly browser interface to explore molecular orbitals as isosurfaces.

## Features

- Supply your own geometry (atomic numbers + coordinates), basis specification, and MO coefficients
- Optional: MO energies and occupations for sorting/filtering
- Interactive Dash UI: select orbital index, adjust isovalue, positive/negative coloring, opacity
- 3D molecule view with ball & stick + orbital isosurface
- Pluggable evaluator: use provided PySCF helper or inject your own AO/MO evaluator

## Install

```bash
pip install -e .
```

For development/tests:

```bash
pip install -e .
pip install pytest
```

## Quick Start

```python
from orbitaldash.app import launch_app
from orbitaldash.example_data import water_pyscf_hf

geom, mo_coeff, mo_energy, mo_occ, basis = water_pyscf_hf()
launch_app(geometry=geom, mo_coeff=mo_coeff, mo_energies=mo_energy, mo_occ=mo_occ, basis=basis)
```

This opens a local Dash server in your browser.

### PySCF Direct Example

See `examples/h2o_demo.py` for a minimal Hartree-Fock run then launching the UI.

#### Uncontracted basis example (benzene)

If you work with uncontracted bases, pass the original PySCF `mol` object to ensure the AO basis used for grid evaluation matches your `mo_coeff`. The example below decontracts cc-pVDZ for benzene and runs RHF:

- Run: activate your environment, then `python examples/benzene_unc_ccpvdz.py`
- The script passes the uncontracted `pmol` directly into `launch_app(basis=pmol, ...)` to avoid AO/MO dimension mismatches.

See `examples/benzene_unc_ccpvdz.py`.

### Custom Evaluator

You can generate MO values yourself and supply a callable directly to `launch_app`:

```python
from orbitaldash import launch_app
import numpy as np

geometry = {"symbols": ["H","H"], "coords": np.array([[0,0,0],[0,0,0.74]])}
mo_coeff = np.eye(2)  # determines n_mo only in custom mode

def custom_eval(grid):
  # Return shape (npoints, n_mo). Here two fake Gaussians.
  g1 = np.exp(- (np.linalg.norm(grid - geometry['coords'][0], axis=1)**2) / 0.5)
  g2 = np.exp(- (np.linalg.norm(grid - geometry['coords'][1], axis=1)**2) / 0.5)
  return np.stack([g1, g2], axis=1)

launch_app(geometry=geometry, basis='custom', mo_coeff=mo_coeff, custom_eval=custom_eval)
```

See `examples/custom_eval_demo.py` for a complete runnable script. Even though PySCF is now a standard dependency, providing `custom_eval` bypasses the PySCF AO/MO evaluation path.

## Trajectory Mode

Use trajectory mode when you want to browse multiple geometries/frames (for example, along a bond scan or reaction path).

Pass `trajectory=` into `launch_app(...)` as either:

- a list of frames, where each frame is a dict, or
- a dict with key `frames` containing that list.

Each frame must provide:

- `geometry`: same accepted geometry formats as single-frame mode
- `mo_coeff`: MO coefficient matrix for that frame
- `basis`: basis spec for that frame (or a custom-eval setup)
- `metadata["pes"]`: dict of labeled PES values for plotting, e.g. `{"HF": -108.95, "CASCI": -109.12}`

Optional per frame:

- `mo_energies`: orbital energies for frame-aware HOMO/LUMO marking
- `mo_occ`: orbital occupations for frame-aware HOMO/LUMO marking
- `custom_eval`: callable `(grid) -> (npoints, n_mo)`

Example frame list:

```python
trajectory = [
  {
    "geometry": geom_0,
    "basis": basis_0,
    "mo_coeff": mo_coeff_0,
    "mo_energies": mo_e_0,
    "mo_occ": mo_occ_0,
    "metadata": {
      "pes": {
        "HF": hf_energy_0,
        "CASCI": casci_energy_0,
      }
    },
  },
  {
    "geometry": geom_1,
    "basis": basis_1,
    "mo_coeff": mo_coeff_1,
    "mo_energies": mo_e_1,
    "mo_occ": mo_occ_1,
    "metadata": {
      "pes": {
        "HF": hf_energy_1,
        "CASCI": casci_energy_1,
      }
    },
  },
]

launch_app(
  trajectory=trajectory,
  mode="auto",              # "auto", "precomputed", or "lazy"
  precomputed_h5_path=None,  # optional path/dir for cache files
  spacing_options=[0.30, 0.25, 0.20],
)
```

Notes:

- `mode="auto"`/`"precomputed"` uses HDF5 cache files for smoother playback and quality switching.
- `mode="lazy"` evaluates each requested frame/orbital on demand.
- If you provide `custom_eval` for a frame, that frame can bypass the PySCF AO evaluation path.
- Trajectory examples: `examples/h2_dissociation_trajectory.py`, `examples/n2_dissociation_trajectory.py`, and `examples/precompute_h2_trajectory_h5.py`.

## Data Conventions

- geometry: dict with keys `symbols` (list[str]) and `coords` (np.ndarray shape (N,3) in Angstrom) OR a tuple (Z, coords) where Z are atomic numbers (list[int] or np.ndarray)
- mo_coeff: np.ndarray shape (n_ao, n_mo)
- mo_energies (optional): shape (n_mo,)
- mo_occ (optional): shape (n_mo,)
- basis: minimal string or PySCF-compliant basis spec (currently only used if PySCF grid evaluation is requested)

## Roadmap

- Density and spin-density visualization
- Multiple isosurface levels
- Export of cube/mesh
- Performance improvements (vectorized batch MO evaluation)
- Accept precomputed evaluator object in `launch_app`
