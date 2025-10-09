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
pip install -e .[full]
```

(omit `[full]` if you will not use PySCF helper utilities)

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

### Custom Evaluator (No PySCF)

You can generate MO values yourself and supply a callable:

```python
from orbitaldash import launch_app
from orbitaldash.orbitals import OrbitalEvaluator
import numpy as np

geometry = {"symbols": ["H","H"], "coords": np.array([[0,0,0],[0,0,0.74]])}
mo_coeff = np.eye(2)  # placeholder

def custom_eval(grid):
  # Return shape (npoints, n_mo). Here two fake gaussians.
  g1 = np.exp(- (np.linalg.norm(grid - geometry['coords'][0], axis=1)**2) / 0.5)
  g2 = np.exp(- (np.linalg.norm(grid - geometry['coords'][1], axis=1)**2) / 0.5)
  return np.stack([g1, g2], axis=1)

evaluator = OrbitalEvaluator(geometry, basis='custom', mo_coeff=mo_coeff, custom_eval=custom_eval)
# Use evaluator inside a customized flow or adapt launch_app to accept it (future feature)
```

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
