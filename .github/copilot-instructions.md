# Copilot instructions for orbitaldash

Purpose: Help AI coding agents be productive quickly in this repo by capturing architecture, workflows, and project-specific conventions.

## Big picture

- This is a small Python package that serves a Dash + Plotly UI to visualize molecular orbitals as isosurfaces.
- Core modules (under `src/orbitaldash/`):
  - `geometry.py`: Molecule data model, element metadata, and simple bond inference from covalent radii. Provides `standardize_geometry` that normalizes various user inputs to a `Molecule`.
  - `orbitals.py`: Grid creation, PySCF integration for AO/MO evaluation, and an `OrbitalEvaluator` class with caching and adjustable grid spacing.
  - `app.py`: Dash app wiring. Builds molecule traces and orbital isosurfaces; exposes `launch_app(...)` as the package entry point.
- Typical flow: user provides geometry + basis + `mo_coeff` (and optionally energies/occupations). `launch_app` creates a `Molecule`, instantiates `OrbitalEvaluator`, and serves an interactive `dcc.Graph` figure with sliders for orbital index, isovalue fraction, and quality (grid spacing).

## Data shapes and contracts

- geometry: either
  - dict: `{ "symbols": list[str], "coords": np.ndarray (N,3) in Angstrom }`, or
  - tuple: `([Z] | [symbols], coords)`; converted via `standardize_geometry`.
- `mo_coeff`: np.ndarray `(n_ao, n_mo)`.
- `OrbitalEvaluator.grid()` returns `(grid (P,3), origin(3,), steps(3), spacing)`.
- `OrbitalEvaluator.mo_values(i)` returns `(P,)` for orbital i; cached by index until regrid.
- `OrbitalData` contains `grid`, `values`, `origin`, `steps`, `spacing` and is used by plotting.

## Conventions and gotchas

- Units: geometry coordinates must be in Angstrom for all internal computations. Examples explicitly convert from Bohr when using PySCF. See `examples/h2o_demo.py` and `examples/h2o_debug.py`.
- PySCF is optional. The package can evaluate MOs either via PySCF (`evaluate_mos_on_grid`) or a user-supplied `custom_eval` callable passed to `OrbitalEvaluator`.
- Grid spacing (“quality”) is controlled by a vertical slider; smaller spacing = finer grid = higher compute. Re-gridding clears caches via `OrbitalEvaluator.regrid`.
- Bonds are inferred with `Molecule.infer_bonds(scale=1.2)` using simple covalent radii; visuals draw bonds after atom markers so lines stay visible. Atom marker sizes and bond line widths scale with camera zoom to preserve readability.
- Plot uses dual isosurfaces for positive/negative phases with `RdBu` colorscale and hidden caps, see `_orbital_isosurface`.

## Developer workflows

- Install (editable, with PySCF extras):
  - `pip install -e .[full]` (omit `[full]` if you don’t use PySCF)
- Run the example Dash app (requires PySCF):
  - `python examples/h2o_demo.py`
  - For debugging bonds and units, use `python examples/h2o_debug.py`
- Launch programmatically:
  - `from orbitaldash import launch_app` then call `launch_app(geometry, basis, mo_coeff, mo_energies=None, mo_occ=None, host='127.0.0.1', port=8050, auto_run=True, debug=False)`
- Tests (pytest):
  - `pytest` or `pip install -e .[dev] && pytest`
  - Unit tests do not require PySCF; they use a `custom_eval` to avoid heavy deps.

## Code patterns worth reusing

- Geometry intake: always pass user inputs through `standardize_geometry` to get a `Molecule`.
- Grid-dependent caching: when any parameter affecting the grid changes (e.g., spacing), call `OrbitalEvaluator.regrid(spacing)` to invalidate caches.
- PySCF integration: build a `gto.M` via `build_pyscf_mol(Molecule, basis)`; always convert grid coordinates from Angstrom to Bohr before evaluating AO values.
- Plot composition: build atom and bond traces with `_molecule_traces`, then append orbital isosurfaces from `_orbital_isosurface`; use `build_figure` for a complete figure.
- UI state: camera zoom inference from `relayoutData` is used to scale atom sizes and bond widths; preserve via `uirevision='keep'`.

## Extensibility hints

- Alternate evaluators: pass a `custom_eval(grid) -> (P, n_mo)` to `OrbitalEvaluator` for non-PySCF backends.
- Additional visualizations (e.g., density): follow the `OrbitalData` pattern (grid + values) and add new isosurface traces in `build_figure`.
- Element data: extend `ELEMENT_COLORS`, `COVALENT_RADII`, and symbol/Z maps in `geometry.py` as needed.

## Key files

- `src/orbitaldash/app.py`: Dash app, UI callbacks, figure building.
- `src/orbitaldash/orbitals.py`: grid creation, MO evaluation, evaluator/caching.
- `src/orbitaldash/geometry.py`: molecule model and bond inference.
- `examples/h2o_demo.py`: PySCF-driven example; canonical usage.
- `examples/h2o_debug.py`: unit/bond debugging example.
- `tests/`: fast unit tests with custom evaluator and geometry helpers.

## Other instructions

If you run any python script, first activate my environment using `source /home/emielkoridon/pip_environments/dftqml/bin/activate`
