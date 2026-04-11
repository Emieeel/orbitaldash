"""Dash app for interactive molecular orbital visualization.

This module wires together geometry, orbital evaluation, and Plotly/Dash to
provide a browser-based UI for exploring orbitals as isosurfaces alongside a
ball-and-stick molecular view.

Key functions:
    - :func:`build_figure`: Compose molecule and orbital isosurfaces.
    - :func:`launch_app`: Create and run the Dash server.
"""

import numpy as np
from typing import Optional, Any, Callable
import logging
import inspect
from pathlib import Path

from .geometry import standardize_geometry, ELEMENT_COLORS, COVALENT_RADII
from .orbitals import (
    OrbitalEvaluator,
    OrbitalData,
    evaluate_mo_on_grid_single,
    make_global_grid,
    precompute_trajectory_to_hdf5,
    load_trajectory_hdf5_grid,
    load_trajectory_hdf5_values,
    trajectory_cache_key,
)
from .trajectory import standardize_trajectory, frame_reaction_coordinates

import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _running_in_notebook() -> bool:
    """Best-effort detection of a Jupyter/IPython notebook environment."""
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None:
            return False
        # Most robust signal: running under an IPython kernel.
        return bool(getattr(ip, 'config', {}).get('IPKernelApp'))
    except Exception:
        return False


def _molecule_traces(mol, show_labels: bool, show_bonds: bool, debug: bool = False,
                     atom_scale: float = 1.0, bond_width: int = 6, bond_scale: float = 1.2):
    """Create Plotly 3D traces for atoms and bonds.

    Bonds are shortened at each end using covalent radii so that sticks meet
    the spheres naturally. Bonds are added before atoms so they render behind
    spheres, improving visibility.

    Args:
        mol: A :class:`~orbitaldash.geometry.Molecule` instance.
        show_labels: Whether to draw element symbols above atoms.
        show_bonds: Whether to draw bonds inferred from distances.
        debug: If True, draw bonds in lime and enforce a minimum width.
        atom_scale: Multiplier for atom marker sizes.
        bond_width: Base bond line width (pixels) before zoom scaling.

    Returns:
        list: Plotly traces for bonds (optional) and atoms.
    """
    coords = mol.coords
    symbols = mol.symbols
    colors = [ELEMENT_COLORS.get(z, '#BBBBBB') for z in mol.atomic_numbers]
    # Atom marker sizing (retain prior nonlinear emphasis, slight opacity tweak)
    base_sizes = []
    for z in mol.atomic_numbers:
        r = COVALENT_RADII.get(int(z), 0.7)
        scale = (r ** 0.8) * 120 / 1.5
        if z == 8:
            scale *= 1.1
        base_sizes.append(scale)
    # Rebase atom scale so that UI value 1.0 corresponds to previous 0.6
    atom_scale_eff = atom_scale * 0.6
    sizes = [max(22.0, s * atom_scale_eff) for s in base_sizes]
    mode = 'markers+text' if show_labels else 'markers'
    atom_trace = go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode=mode,
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.97,
            line=dict(width=1.0, color='rgba(0,0,0,0.6)')
        ),
        name='Atoms',
        text=symbols if show_labels else None,
        textposition='top center',
        hovertext=symbols,
        hoverinfo='text'
    )
    bonds = mol.infer_bonds(scale=float(bond_scale),
                            with_order=True) if show_bonds else []
    if bonds:
        bond_xyz_by_order: dict[int, tuple[list, list, list]] = {
            1: ([], [], []),
            2: ([], [], []),
            3: ([], [], []),
        }
        for i, j, order in bonds:
            p1 = coords[i]
            p2 = coords[j]
            # Professionally trimmed sticks: shorten at each end using covalent radii
            v = p2 - p1
            d = float(np.linalg.norm(v))
            if d < 1e-8:
                continue
            u = v / d
            ri = COVALENT_RADII.get(int(mol.atomic_numbers[i]), 0.7)
            rj = COVALENT_RADII.get(int(mol.atomic_numbers[j]), 0.7)
            # Trim ends so the stick meets the spheres, but never trim so much
            # that short bonds (e.g., N≡N) collapse into tiny dashes.
            # Use conservative trimming so bonds remain visually connected,
            # especially for short bonds like N2.
            trim_i = min(d * 0.20, ri * 0.45)
            trim_j = min(d * 0.20, rj * 0.45)
            remaining = d - trim_i - trim_j
            min_remaining = max(0.35 * d, 0.28)
            if remaining < min_remaining:
                # Scale trims down proportionally to preserve a visible segment.
                total_trim = trim_i + trim_j
                if total_trim > 1e-12:
                    target_total_trim = max(0.0, d - min_remaining)
                    f = target_total_trim / total_trim
                    trim_i *= f
                    trim_j *= f
                remaining = d - trim_i - trim_j
            q1 = p1 + u * trim_i
            q2 = p2 - u * trim_j

            # For double/triple bonds, draw multiple parallel segments.
            # Choose a stable perpendicular direction in 3D.
            axis = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(u, axis))) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            perp = np.cross(u, axis)
            perp_norm = float(np.linalg.norm(perp))
            if perp_norm < 1e-8:
                # Fallback in the rare case u ~ axis
                axis = np.array([0.0, 0.0, 1.0])
                perp = np.cross(u, axis)
                perp_norm = float(np.linalg.norm(perp))
            if perp_norm < 1e-8:
                perp = np.array([0.0, 0.0, 0.0])
            else:
                perp = perp / perp_norm

            order_i = int(max(1, min(3, int(order))))
            # Offset between parallel lines (kept small so it reads like a bond, not a block)
            delta = min(0.12, 0.06 * float(ri + rj),
                        0.25 * max(remaining, 0.0))
            if order_i == 1:
                offsets = [0.0]
            elif order_i == 2:
                offsets = [-delta, +delta]
            else:
                offsets = [-delta, 0.0, +delta]

            # Slightly shorten the "outer" lines on multi-bonds for readability.
            extra_trim = min(0.10, 0.25 * max(remaining, 0.0))

            for off in offsets:
                shift = perp * float(off)
                a = q1 + shift
                b = q2 + shift
                if order_i > 1 and abs(float(off)) > 1e-12:
                    a = a + u * extra_trim
                    b = b - u * extra_trim
                bx, by, bz = bond_xyz_by_order[order_i]
                bx += [a[0], b[0], None]
                by += [a[1], b[1], None]
                bz += [a[2], b[2], None]

        # Map UI bond width so that UI 10 corresponds to previous visual width ~13
        width_eff = int(round(bond_width * 1.3))
        bond_traces = []
        # Draw higher-order bonds a bit thinner so they don't read as a solid slab.
        width_scale = {1: 1.0, 2: 0.65, 3: 0.50}
        for bo in (3, 2, 1):
            bx, by, bz = bond_xyz_by_order[bo]
            if not bx:
                continue
            bond_traces.append(
                go.Scatter3d(
                    x=bx, y=by, z=bz,
                    mode='lines',
                    line=dict(
                        color=('lime' if debug else 'rgba(70,70,70,0.85)'),
                        width=(max(
                            2, int(round(width_eff * width_scale[bo]))) if not debug else max(width_eff, 8))
                    ),
                    name='Bonds',
                    hoverinfo='skip'
                )
            )

        # Draw bonds BEFORE atoms so they sit visually behind spheres
        return bond_traces + [atom_trace]
    else:
        return [atom_trace]


def _orbital_isosurface(data, isovalue: float):
    """Create dual-lobe isosurfaces for a single orbital.

    Args:
        data: :class:`~orbitaldash.orbitals.OrbitalData` for one orbital.
        isovalue: Absolute isovalue threshold used for both lobes.

    Returns:
        list: Two Plotly Isosurface traces (negative and positive lobes).
    """
    # reshape values into 3D
    grid = data.grid
    vals = data.values
    nx, ny, nz = data.steps
    X = grid[:, 0].reshape(nx, ny, nz)
    Y = grid[:, 1].reshape(nx, ny, nz)
    Z = grid[:, 2].reshape(nx, ny, nz)
    V = vals.reshape(nx, ny, nz)

    # Render a single isovalue surface per phase.
    # Using a wide [isomin, isomax] interval + surface_count>1 produces "layered" shells
    # that can look grainy. Here we keep the interval infinitesimally small.
    eps = 1e-12
    lighting = dict(ambient=0.25, diffuse=0.85, specular=0.35,
                    roughness=0.25, fresnel=0.15)
    lightpos = dict(x=1.2, y=1.2, z=0.8)

    # Use fixed colors per phase. With surface_count=1 and a very narrow
    # [isomin, isomax] interval, Plotly’s colorscale mapping can otherwise
    # wash out into a near-constant grey.
    red = [[0.0, 'rgb(220, 50, 47)'], [1.0, 'rgb(220, 50, 47)']]
    blue = [[0.0, 'rgb(38, 139, 210)'], [1.0, 'rgb(38, 139, 210)']]

    surf_pos = go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=V.flatten(),
        isomin=float(isovalue), isomax=float(isovalue + eps), opacity=0.55,
        surface_count=1, colorscale=red, showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=lighting,
        lightposition=lightpos,
        name='Orbital+'
    )
    surf_neg = go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=V.flatten(),
        isomin=float(-isovalue - eps), isomax=float(-isovalue), opacity=0.55,
        surface_count=1, colorscale=blue, showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=lighting,
        lightposition=lightpos,
        name='Orbital-'
    )
    return [surf_neg, surf_pos]


def _smooth_field_separable(V: np.ndarray, passes: int) -> np.ndarray:
    """Fast visual-only smoothing for a 3D scalar field.

    Uses a small separable kernel (approx Gaussian, sigma~1 voxel) applied
    along each axis. This reduces blocky/aliasing artifacts without changing
    the underlying evaluation grid.

    Args:
        V: 3D array (nx, ny, nz)
        passes: Number of smoothing passes (0 disables)

    Returns:
        Smoothed 3D array.
    """
    p = int(passes)
    if p <= 0:
        return V
    out = V
    for _ in range(p):
        # axis 0
        pad = np.pad(out, ((2, 2), (0, 0), (0, 0)), mode='edge')
        out = (pad[0:-4] + 4 * pad[1:-3] + 6 *
               pad[2:-2] + 4 * pad[3:-1] + pad[4:]) / 16.0
        # axis 1
        pad = np.pad(out, ((0, 0), (2, 2), (0, 0)), mode='edge')
        out = (pad[:, 0:-4, :] + 4 * pad[:, 1:-3, :] + 6 *
               pad[:, 2:-2, :] + 4 * pad[:, 3:-1, :] + pad[:, 4:, :]) / 16.0
        # axis 2
        pad = np.pad(out, ((0, 0), (0, 0), (2, 2)), mode='edge')
        out = (pad[:, :, 0:-4] + 4 * pad[:, :, 1:-3] + 6 *
               pad[:, :, 2:-2] + 4 * pad[:, :, 3:-1] + pad[:, :, 4:]) / 16.0
    return out


def build_figure(mol, orbital_data=None, isovalue=None, show_labels: bool = True, show_bonds: bool = True, debug: bool = False,
                 atom_scale: float = 1.0, bond_width: int = 6, bond_scale: float = 1.2, camera: Optional[dict] = None,
                 scene_ranges: Optional[dict[str, list[float]]] = None,
                 height: int = 650):
    """Compose the full 3D figure with molecule and optional isosurfaces.

    Args:
        mol: A :class:`~orbitaldash.geometry.Molecule`.
        orbital_data: Optional :class:`~orbitaldash.orbitals.OrbitalData` to render.
        isovalue: Optional explicit isovalue; if None, uses a default fraction.
        show_labels: Whether to draw element symbols above atoms.
        show_bonds: Whether to draw inferred bonds.
        debug: If True, draw bonds in a debug style.
        atom_scale: Multiplier for atom marker sizes (further scaled by zoom).
        bond_width: Base bond line width (further scaled by zoom).
        bond_scale: Distance cutoff scale for bond inference. Larger values show more/longer bonds.
        camera: Optional Plotly camera dict to apply.

    Returns:
        plotly.graph_objects.Figure: The composed 3D figure.
    """
    traces = _molecule_traces(
        mol, show_labels, show_bonds=show_bonds, debug=debug,
        atom_scale=atom_scale, bond_width=bond_width, bond_scale=bond_scale)
    if orbital_data is not None:
        iso = isovalue
        if iso is None:
            # Default fraction consistent with initial slider (0.15)
            iso = 0.15 * np.max(np.abs(orbital_data.values))
        traces += _orbital_isosurface(orbital_data, iso)
    fig = go.Figure(data=traces)
    axis_common = dict(visible=False, showspikes=False,
                       showgrid=False, zeroline=False, showbackground=False)
    if scene_ranges is not None:
        xaxis = dict(axis_common, range=scene_ranges.get('x'))
        yaxis = dict(axis_common, range=scene_ranges.get('y'))
        zaxis = dict(axis_common, range=scene_ranges.get('z'))
    else:
        xaxis = dict(axis_common)
        yaxis = dict(axis_common)
        zaxis = dict(axis_common)

    scene_cfg: dict[str, Any] = dict(
        xaxis=xaxis,
        yaxis=yaxis,
        zaxis=zaxis,
        aspectmode='data',
        dragmode='orbit'
    )
    # Apply camera: preserve user camera when provided; otherwise use a sensible default at initialization
    if camera is not None:
        scene_cfg['camera'] = camera
    else:
        scene_cfg['camera'] = dict(eye=dict(x=0.9, y=0.9, z=0.6))

    fig.update_layout(
        template='simple_white',
        scene=scene_cfg,
        showlegend=False,
        uirevision='keep',  # preserve user camera across updates
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=30),
        height=int(height),
    )
    return fig


def launch_app(geometry=None, basis: Optional[Any] = None, mo_coeff: Optional[np.ndarray] = None, mo_energies: Optional[np.ndarray] = None, mo_occ: Optional[np.ndarray] = None,
               active: Optional[list[int]] = None,
               custom_eval: Optional[Callable[[
                   np.ndarray], np.ndarray]] = None,
               trajectory: Optional[Any] = None,
               mode: str = 'auto',
               cache_dir: Optional[str] = None,
               precomputed_h5_path: Optional[str] = None,
               rebuild_cache: bool = False,
               cache_key: Optional[str] = None,
               spacing_options: Optional[list[float]] = None,
               host='127.0.0.1', port=8050, auto_run: bool = True, debug: bool = False,
               max_orbitals: Optional[int] = 70,
               verbose: bool = False,
               jupyter: Optional[bool] = None,
               jupyter_mode: str = 'inline',
               minimal_ui: Optional[bool] = None,
               bond_scale: float = 1.2):
    """Launch a Dash app to explore molecular orbitals interactively.

    The app renders a molecule with atom/bond controls and an orbital isosurface
    overlay. Orbital values are provided by :class:`~orbitaldash.orbitals.OrbitalEvaluator`.

    Args:
        geometry: Geometry accepted by ``standardize_geometry`` (dict, tuple, or Molecule). Alternatively, this
            can be a PySCF SCF object (e.g., ``pyscf.scf.hf.RHF``) or a CASCI object
            (e.g., ``pyscf.mcscf.casci.CASCI``), in which case ``basis`` and ``mo_coeff``
            are inferred automatically unless explicitly provided.
        basis: Basis specification (string) or a PySCF mol-like object. Ignored if ``custom_eval`` is provided.
            When ``geometry`` is a PySCF SCF/CASCI object, this is inferred from ``obj.mol``.
        mo_coeff: MO coefficient matrix with shape (n_ao, n_mo). When ``geometry`` is a PySCF SCF/CASCI object,
            this is inferred from ``obj.mo_coeff`` unless explicitly provided here.
        mo_energies: Optional MO energies (n_mo,).
        mo_occ: Optional occupations (n_mo,) used to initialize to HOMO.
        active: Optional list of orbital indices to highlight as "active" in the UI. If a CASCI object
            is passed, and this is None, it will default to the CAS active space indices [ncore .. ncore+ncas-1].
        custom_eval: Optional callable mapping grid (P, 3) to MO values (P, n_mo).
        trajectory: Optional trajectory input. For PES plotting, each frame metadata
            must include:
            ``metadata['pes'] = {'LabelA': energy_a, 'LabelB': energy_b, ...}``
            Any provided labels are plotted.
        host: Host interface for the Dash server.
        port: Port for the Dash server.
        auto_run: Whether to immediately start the server.
        spacing_options: Optional selectable grid spacings (Angstrom) for the quality slider.
            In trajectory mode with ``mode`` = ``precomputed`` or ``auto``:
            - if provided, these spacings become selectable quality levels;
            - if omitted, a single default spacing is used and the quality slider is disabled.
        debug: Enable debug styling for bonds.
        max_orbitals: Cap how many orbitals are exposed in the UI.
        jupyter: If True, attempt to run inside a Jupyter notebook (inline/external) when supported by Dash.
            If None (default), this is auto-detected.
        jupyter_mode: Dash Jupyter mode, typically 'inline' or 'external'. Only used when jupyter is True.
        minimal_ui: If True, use a compact UI suitable for notebook output cells (fewer controls + smaller figure).
            If None (default), this is enabled automatically when jupyter is True.
        bond_scale: Distance cutoff scale for bond inference. Larger values show more/longer bonds.

    Returns:
        dash.Dash: The configured Dash application instance.
    """
    mode = str(mode).lower().strip()
    if mode not in {'auto', 'lazy', 'precomputed'}:
        raise ValueError("mode must be one of: 'auto', 'lazy', 'precomputed'")

    trajectory_mode = trajectory is not None
    traj = None
    frames = []
    frame_molecules = []
    traj_pes_series: dict[str, np.ndarray] = {}
    traj_rc = np.array([], dtype=float)

    if trajectory_mode:
        traj = standardize_trajectory(trajectory, default_basis=basis)
        frames = list(traj.frames)
        frame_molecules = [standardize_geometry(f.geometry) for f in frames]
        if not frame_molecules:
            raise ValueError("trajectory must contain at least one frame")
        missing_occ_or_energy = [
            i for i, f in enumerate(frames)
            if (f.mo_occ is None or f.mo_energies is None)
        ]
        if missing_occ_or_energy:
            bad = ", ".join(str(i) for i in missing_occ_or_energy[:8])
            raise ValueError(
                "trajectory mode requires mo_energies and mo_occ for each frame "
                f"(missing in frame indices: {bad})"
            )
        mol = frame_molecules[0]
        n_mo = min(int(np.asarray(f.mo_coeff).shape[1]) for f in frames)
        # PES series from frame metadata; required form per frame:
        # metadata["pes"] = {"HF": ..., "CASCI": ..., "DFT": ...}
        missing_pes = [
            i for i, f in enumerate(frames)
            if not isinstance((f.metadata or {}).get('pes'), dict)
        ]
        if missing_pes:
            bad = ", ".join(str(i) for i in missing_pes[:8])
            raise ValueError(
                "trajectory mode requires metadata['pes'] mapping in each frame "
                f"(missing in frame indices: {bad})"
            )

        labels: list[str] = []
        for f in frames:
            md = f.metadata or {}
            pes = md.get('pes')
            assert isinstance(pes, dict)
            for key in pes.keys():
                s = str(key)
                if s not in labels:
                    labels.append(s)
        if not labels:
            raise ValueError("metadata['pes'] must contain at least one energy label")

        for lbl in labels:
            vals: list[float] = []
            for f in frames:
                md = f.metadata or {}
                pes = md.get('pes')
                v = None
                if isinstance(pes, dict) and lbl in pes:
                    v = pes.get(lbl)
                vals.append(float(v) if v is not None else np.nan)
            arr = np.asarray(vals, dtype=float)
            if np.any(~np.isnan(arr)):
                traj_pes_series[lbl] = arr
        if not traj_pes_series:
            raise ValueError("metadata['pes'] does not contain any numeric values across frames")
        traj_rc = frame_reaction_coordinates(traj)
    else:
        # Allow passing a PySCF SCF object directly as the first argument, e.g. launch_app(mf)
        # We detect by duck-typing to avoid importing pyscf unless needed.
        if geometry is not None and not isinstance(geometry, dict):
            # Detect a PySCF-like object by presence of required attributes
            if hasattr(geometry, 'mo_coeff') and hasattr(geometry, 'mol'):
                obj = geometry  # type: ignore[assignment]
                pyscf_mol = getattr(obj, 'mol')
                # Extract geometry in Angstrom
                try:
                    coords_ang = pyscf_mol.atom_coords(unit='Angstrom')
                except Exception:
                    # Fallback: request Bohr and convert locally
                    BOHR_TO_ANG = 0.529177210903
                    coords_ang = pyscf_mol.atom_coords(unit='Bohr') * BOHR_TO_ANG
                # Prefer pure symbols if available
                try:
                    symbols = [pyscf_mol.atom_pure_symbol(
                        i) for i in range(pyscf_mol.natm)]
                except Exception:
                    symbols = [pyscf_mol.atom_symbol(i)
                               for i in range(pyscf_mol.natm)]
                geometry = {"symbols": symbols, "coords": coords_ang}
                basis = pyscf_mol
                # Only override mo_* when not explicitly provided
                if mo_coeff is None:
                    mo_coeff = getattr(obj, 'mo_coeff')
                if mo_energies is None and hasattr(obj, 'mo_energy'):
                    mo_energies = getattr(obj, 'mo_energy')
                if mo_occ is None and hasattr(obj, 'mo_occ'):
                    mo_occ = getattr(obj, 'mo_occ')
                # If this is a CASCI object (duck-type via ncore/ncas), build default active indices
                if active is None and hasattr(obj, 'ncore') and hasattr(obj, 'ncas'):
                    try:
                        ncore = int(getattr(obj, 'ncore'))
                        ncas = int(getattr(obj, 'ncas'))
                        active = list(range(ncore, ncore + ncas))
                    except Exception:
                        active = None

        # Standardize geometry path (works for both dict and Molecule inputs)
        if geometry is None:
            raise ValueError(
                "geometry must be provided, or pass a PySCF SCF object as the first argument to launch_app(mf)")
        mol = standardize_geometry(geometry)
        if mo_coeff is None:
            raise ValueError(
                "mo_coeff must be provided unless a PySCF SCF object is passed as the first argument")
        mo_coeff = np.asarray(mo_coeff)
        n_mo = int(np.asarray(mo_coeff).shape[1])
    # Limit how many orbitals are exposed in the UI by default to keep it usable
    if max_orbitals is not None:
        try:
            n_mo_display = max(1, min(int(max_orbitals), n_mo))
        except Exception:
            n_mo_display = min(70, n_mo)
    else:
        n_mo_display = n_mo
    # If custom_eval is provided, OrbitalEvaluator will use it and not require PySCF.
    evaluator = None
    if not trajectory_mode:
        assert mo_coeff is not None
        evaluator = OrbitalEvaluator(
            mol, basis, mo_coeff, custom_eval=custom_eval, verbose=bool(verbose))

    def _spacing_tag(spacing: float) -> str:
        return f"{float(spacing):.3f}".replace('.', 'p')

    def _build_precomputed_path(spacing: float) -> str:
        sp_tag = _spacing_tag(spacing)
        if precomputed_h5_path:
            p = Path(precomputed_h5_path)
            if p.suffix.lower() == '.h5':
                if len(quality_values) <= 1:
                    return str(p)
                return str(p.with_name(f"{p.stem}_s{sp_tag}{p.suffix}"))
            # Treat non-.h5 path as a directory when multiple spacing caches are needed.
            p.mkdir(parents=True, exist_ok=True)
            if cache_key:
                key = str(cache_key)
            else:
                assert traj is not None
                key = trajectory_cache_key(traj, spacing=float(spacing), padding=3.0)
            return str(p / f"trajectory_{key}_s{sp_tag}.h5")
        base_dir = Path(cache_dir or ".orbitaldash_cache")
        if cache_key:
            key = str(cache_key)
        else:
            assert traj is not None
            key = trajectory_cache_key(traj, spacing=float(spacing), padding=3.0)
        return str(base_dir / f"trajectory_{key}_s{sp_tag}.h5")

    _traj_grid_cache: dict[float, tuple[np.ndarray, np.ndarray, tuple[int, int, int], float]] = {}
    _traj_vals_cache: dict[tuple[float, int, int], np.ndarray] = {}
    _traj_h5_by_spacing: dict[float, str] = {}

    def _ensure_h5(spacing: float) -> str:
        if float(spacing) in _traj_h5_by_spacing and not rebuild_cache:
            return _traj_h5_by_spacing[float(spacing)]
        assert traj is not None
        path = _build_precomputed_path(spacing)
        needs_build = rebuild_cache or (not Path(path).exists())
        if needs_build:
            precompute_trajectory_to_hdf5(
                traj,
                h5_path=path,
                spacing=float(spacing),
                padding=3.0,
                verbose=bool(verbose),
            )
        _traj_h5_by_spacing[float(spacing)] = path
        return path

    def _trajectory_grid(spacing: float):
        sp = float(spacing)
        if sp in _traj_grid_cache and not rebuild_cache:
            return _traj_grid_cache[sp]
        if mode in {'auto', 'precomputed'}:
            h5_path = _ensure_h5(sp)
            out = load_trajectory_hdf5_grid(h5_path)
        else:
            assert frame_molecules
            out = make_global_grid(frame_molecules, padding=3.0, spacing=sp)
        _traj_grid_cache[sp] = out
        return out

    def _trajectory_values(frame_idx: int, orbital_idx: int, spacing: float) -> np.ndarray:
        sp = float(spacing)
        key = (sp, int(frame_idx), int(orbital_idx))
        if key in _traj_vals_cache and not rebuild_cache:
            return _traj_vals_cache[key]

        if mode in {'auto', 'precomputed'}:
            h5_path = _ensure_h5(sp)
            vals = load_trajectory_hdf5_values(h5_path, frame_idx=frame_idx, orbital_idx=orbital_idx)
        else:
            grid, _origin, _steps, _ = _trajectory_grid(sp)
            frame = frames[int(frame_idx)]
            coeff = np.asarray(frame.mo_coeff)
            orb_idx = int(max(0, min(orbital_idx, coeff.shape[1] - 1)))
            if frame.custom_eval is not None:
                vals = np.asarray(frame.custom_eval(grid)[:, orb_idx], dtype=float)
            else:
                vals = evaluate_mo_on_grid_single(frame_molecules[int(frame_idx)], frame.basis, coeff[:, orb_idx], grid)

        _traj_vals_cache[key] = vals
        return vals

    # Determine initial orbital index and compute HOMO/LUMO for marks
    # HOMO definition (requested): orbital with HIGHEST energy among those with occupation > 1
    # LUMO (heuristic): orbital with LOWEST energy among those with occupation <= 1
    initial_idx = 0
    homo_idx: Optional[int] = None
    lumo_idx: Optional[int] = None
    occ_for_marks: Optional[np.ndarray] = None
    en_for_marks: Optional[np.ndarray] = None
    if trajectory_mode and frames:
        if frames[0].mo_occ is not None:
            occ_for_marks = np.asarray(frames[0].mo_occ).reshape(-1)
        if frames[0].mo_energies is not None:
            en_for_marks = np.asarray(frames[0].mo_energies).reshape(-1)
    elif mo_occ is not None:
        occ_for_marks = np.asarray(mo_occ).reshape(-1)
        if mo_energies is not None:
            en_for_marks = np.asarray(mo_energies).reshape(-1)

    if occ_for_marks is not None:
        try:
            occ = occ_for_marks
            if occ.shape[0] == n_mo:
                if en_for_marks is not None and en_for_marks.shape[0] == n_mo:
                    en = en_for_marks
                    occ_mask = occ > 1.0
                    virt_mask = occ <= 1.0
                    if np.any(occ_mask):
                        occ_idx = np.where(occ_mask)[0]
                        homo_idx = int(occ_idx[np.argmax(en[occ_idx])])
                        initial_idx = homo_idx
                    if np.any(virt_mask):
                        virt_idx = np.where(virt_mask)[0]
                        lumo_idx = int(virt_idx[np.argmin(en[virt_idx])])
                else:
                    # Fallback: index-based HOMO as highest occupied index
                    occupied = np.where(occ > 1.0)[0]
                    if occupied.size > 0:
                        homo_idx = int(occupied.max())
                        initial_idx = homo_idx
                        if homo_idx + 1 < n_mo:
                            lumo_idx = homo_idx + 1
        except Exception:
            initial_idx = 0
    # Clamp initial selection into displayed range
    initial_idx = min(initial_idx, n_mo_display - 1)

    app = dash.Dash(__name__)
    use_jupyter = _running_in_notebook() if jupyter is None else bool(jupyter)
    use_minimal_ui = (use_jupyter if minimal_ui is None else bool(minimal_ui))

    # Heuristic: if AO count is huge (e.g., uncontracted on large systems), start at a coarser grid.
    if trajectory_mode:
        n_ao = int(np.asarray(frames[0].mo_coeff).shape[0])
    else:
        assert mo_coeff is not None
        n_ao = int(mo_coeff.shape[0])
    coarse_default = 0.25 if n_ao < 400 else (0.3 if n_ao < 800 else 0.35)

    # Quality slider options: in precomputed trajectory mode, use requested spacing set
    # (or one default spacing if none supplied), and disable slider when there is only one.
    if trajectory_mode and mode in {'auto', 'precomputed'}:
        if spacing_options:
            raw = [float(s) for s in spacing_options if float(s) > 0]
            if not raw:
                raise ValueError("spacing_options must contain positive values")
            quality_values = sorted({round(v, 3) for v in raw}, reverse=True)
        else:
            quality_values = [round(float(coarse_default), 3)]
    else:
        # Legacy interactive quality list.
        quality_values = [round(x, 2) for x in np.arange(0.5, 0.099, -0.05)]

    quality_marks = {i: f"{s:.2f}" for i, s in enumerate(quality_values)}
    quality_slider_disabled = len(quality_values) <= 1

    try:
        default_quality_index = quality_values.index(coarse_default)
    except ValueError:
        # fallback to nearest available value
        default_quality_index = min(range(len(quality_values)), key=lambda i: abs(
            quality_values[i]-coarse_default))

    if trajectory_mode and mode == 'precomputed':
        # Ensure all selected quality levels are available for smooth switching.
        for sp in quality_values:
            _ensure_h5(float(sp))

    # Normalize active indices to a set for quick membership checks
    active_set = set(int(i) for i in (active or []) if 0 <= int(i) < n_mo)

    def _compute_homo_lumo(occ: np.ndarray, en: Optional[np.ndarray], n: int) -> tuple[Optional[int], Optional[int], int]:
        """Return HOMO/LUMO indices and suggested initial orbital index."""
        homo_local: Optional[int] = None
        lumo_local: Optional[int] = None
        initial_local = 0
        if occ.shape[0] < n:
            return homo_local, lumo_local, initial_local
        occ = occ[:n]
        en_eff: Optional[np.ndarray] = None
        if en is not None and en.shape[0] >= n:
            en_eff = en[:n]

        if en_eff is not None:
            occ_mask = occ > 1.0
            virt_mask = occ <= 1.0
            if np.any(occ_mask):
                occ_idx = np.where(occ_mask)[0]
                homo_local = int(occ_idx[np.argmax(en_eff[occ_idx])])
                initial_local = homo_local
            if np.any(virt_mask):
                virt_idx = np.where(virt_mask)[0]
                lumo_local = int(virt_idx[np.argmin(en_eff[virt_idx])])
        else:
            occupied = np.where(occ > 1.0)[0]
            if occupied.size > 0:
                homo_local = int(occupied.max())
                initial_local = homo_local
                if homo_local + 1 < n:
                    lumo_local = homo_local + 1
        return homo_local, lumo_local, initial_local

    def _orbital_marks(n: int, homo_mark: Optional[int], lumo_mark: Optional[int]):
        # Reduce label clutter for large numbers of orbitals
        if n <= 20:
            step = 1
        elif n <= 50:
            step = 5
        else:
            step = 10
        marks: dict[int, Any] = {}
        # Base sparse numeric marks
        for i in range(0, n, step):
            marks[i] = {"label": str(i)}
        if (n - 1) not in marks:
            marks[n - 1] = {"label": str(n - 1)}
        # HOMO/LUMO markers
        if homo_mark is not None and 0 <= homo_mark < n:
            marks[homo_mark] = {"label": "H", "style": {
                "color": "#2e7d32", "fontWeight": "bold"}}
        if lumo_mark is not None and 0 <= lumo_mark < n:
            marks[lumo_mark] = {"label": "L", "style": {
                "color": "#6a1b9a", "fontWeight": "bold"}}
        # Active orbital dots (orange) – append dot if a label exists, otherwise just a dot
        for a in sorted(active_set):
            if 0 <= a < n:
                if a in marks and isinstance(marks[a], dict):
                    lbl = str(marks[a].get("label", ""))
                    if lbl not in ("H", "L"):
                        marks[a] = {"label": f"{lbl} •", "style": {
                            "color": "#e65100", "fontWeight": "bold"}}
                else:
                    marks[a] = {"label": "•", "style": {
                        "color": "#e65100", "fontWeight": "bold"}}
        return marks

    # Fix scene ranges to the molecule extents so switching orbitals doesn't
    # trigger Plotly autoscale/camera jumps (core orbitals are very small).
    pad = 2.5
    if trajectory_mode:
        mins = np.min(np.stack([m.coords.min(axis=0)
                      for m in frame_molecules], axis=0), axis=0) - pad
        maxs = np.max(np.stack([m.coords.max(axis=0)
                      for m in frame_molecules], axis=0), axis=0) + pad
    else:
        mins = mol.coords.min(axis=0) - pad
        maxs = mol.coords.max(axis=0) + pad
    scene_ranges = {
        'x': [float(mins[0]), float(maxs[0])],
        'y': [float(mins[1]), float(maxs[1])],
        'z': [float(mins[2]), float(maxs[2])],
    }

    default_camera = dict(eye=dict(x=0.9, y=0.9, z=0.6))

    show_pes = trajectory_mode and len(traj_pes_series) > 0

    def _pes_figure(frame_idx: int):
        if not show_pes:
            return go.Figure()
        any_series = next(iter(traj_pes_series.values()))
        i = int(max(0, min(frame_idx, len(any_series) - 1)))
        fig = go.Figure()
        palette = ['#1f77b4', '#ff8f00', '#2e7d32', '#8e24aa', '#6d4c41', '#00897b']
        symbols = ['circle', 'diamond', 'square', 'triangle-up', 'cross', 'x']
        for j, (lbl, yvals) in enumerate(traj_pes_series.items()):
            color = palette[j % len(palette)]
            marker_symbol = symbols[j % len(symbols)]
            fig.add_trace(go.Scatter(x=traj_rc, y=yvals,
                          mode='lines+markers', name=f'{lbl} PES', line=dict(color=color)))
            fig.add_trace(go.Scatter(x=[traj_rc[i]], y=[yvals[i]],
                          mode='markers', marker=dict(size=11, color=color, symbol=marker_symbol), name=f'{lbl} current'))
        fig.update_layout(
            template='simple_white',
            margin=dict(l=30, r=10, t=20, b=30),
            height=220,
            showlegend=True,
            xaxis_title='Reaction coordinate',
            yaxis_title='Energy',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0.0),
        )
        return fig

    # Force a light background for notebook embedding.
    # In Jupyter dark mode, the output area can be dark; Dash components may
    # inherit colors in surprising ways. A white page + dark text keeps labels
    # readable regardless of the surrounding theme.
    if use_minimal_ui:
        fig_height = 295
        fig_width = 560
        responsive_container = {
            'width': '100%',
            'maxWidth': f'{fig_width}px',
            'margin': '0 auto',
            'padding': '0 0.5rem',
            'boxSizing': 'border-box',
        }
        app.layout = html.Div([
            html.H4('Orbital Viewer', style={'margin': '0.25rem 0'}),
            dcc.Store(id='camera-state',
                      data={'base_eye_norm': None, 'last_eye_norm': None, 'camera': default_camera}),
            (dcc.Store(id='frame-play-state', data=False) if trajectory_mode else html.Div()),
            (dcc.Interval(id='frame-interval', interval=600, n_intervals=0, disabled=True) if trajectory_mode else html.Div()),
            html.Div([
                html.Label(
                    f'Orbital index (showing 0..{n_mo_display-1} of total {n_mo})'),
                dcc.Slider(id='orbital-idx', min=0, max=n_mo_display-1, step=1, value=initial_idx,
                           marks=_orbital_marks(n_mo_display, homo_idx, lumo_idx)),  # type: ignore[arg-type]
                html.Div(style={'height': '0.35rem'}),
                (html.Div([
                    html.Label('Frame index'),
                    dcc.Slider(id='frame-idx', min=0, max=len(frames)-1, step=1, value=0,
                               marks={0: '0', len(frames)-1: str(len(frames)-1)}),
                    html.Div([
                        html.Button('Play', id='play-frames', n_clicks=0,
                                    style={'marginRight': '0.4rem'}),
                        dcc.Input(
                            id='frame-speed',
                            type='number',
                            value=600,
                            min=50,
                            max=60000,
                            step=50,
                            debounce=True,
                            style={'width': '78px', 'fontSize': '0.9em'},
                        ),
                        html.Span('ms', style={'marginLeft': '0.25rem', 'fontSize': '0.85em', 'color': '#555'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '0.35rem'})
                ]) if trajectory_mode else html.Div()),
                html.Div(style={'height': '0.35rem'}),
                html.Label('Isovalue fraction (of max)'),
                dcc.Slider(
                    id='iso-frac',
                    min=0.03,
                    max=0.80,
                    step=0.01,
                    value=0.15,
                    marks={0.03: '0.03', 0.15: '0.15', 0.30: '0.30', 0.50: '0.50', 0.80: '0.80'}
                ),
                html.Div(style={'height': '0.35rem'}),
                html.Label('Quality (grid spacing)'),
                dcc.Slider(
                    id='quality',
                    min=0, max=len(quality_values)-1, step=1,
                    value=default_quality_index,
                    marks=quality_marks,
                    disabled=quality_slider_disabled,
                    tooltip={'always_visible': False, 'placement': 'bottom'}
                ),  # type: ignore[arg-type]
            ], style=responsive_container),
            dcc.Graph(
                id='orbital-plot',
                figure=build_figure(
                    mol, show_labels=True, show_bonds=True, debug=debug,
                    bond_scale=bond_scale, camera=default_camera, scene_ranges=scene_ranges, height=fig_height),
                config={'responsive': True, 'displaylogo': False},
                style={
                    'height': f'{fig_height}px',
                    'width': '100%',
                    'maxWidth': f'{fig_width}px',
                    'margin': '0 auto',
                }
            ),
            (dcc.Graph(id='pes-plot', figure=_pes_figure(0), style={
                'height': '220px',
                'width': '100%',
                'maxWidth': f'{fig_width}px',
                'margin': '0 auto',
            }) if show_pes else html.Div()),
            html.Div(id='orbital-info',
                     style={'width': '100%', 'maxWidth': f'{fig_width}px', 'margin': '0.25rem auto 0 auto'})
        ], style={
            'backgroundColor': 'white',
            'color': '#111',
            'padding': '0.25rem 0.5rem',
        })
    else:
        fig_height = 620
        app.layout = html.Div([
            html.H3('Orbital Viewer'),
            dcc.Store(id='camera-state',
                      data={'base_eye_norm': None, 'last_eye_norm': None, 'camera': default_camera}),
            (dcc.Store(id='frame-play-state', data=False) if trajectory_mode else html.Div()),
            (dcc.Interval(id='frame-interval', interval=600, n_intervals=0, disabled=True) if trajectory_mode else html.Div()),
            html.Div([
                html.Div([
                    html.Label(
                        f'Orbital index (showing 0..{n_mo_display-1} of total {n_mo})'),
                    dcc.Slider(id='orbital-idx', min=0, max=n_mo_display-1, step=1, value=initial_idx,
                               marks=_orbital_marks(n_mo_display, homo_idx, lumo_idx)),  # type: ignore[arg-type]
                    (html.Div([
                        html.Label('Frame index'),
                        dcc.Slider(id='frame-idx', min=0, max=len(frames)-1, step=1, value=0,
                                   marks={0: '0', len(frames)-1: str(len(frames)-1)}),
                        html.Div([
                            html.Button('Play', id='play-frames', n_clicks=0,
                                        style={'marginRight': '0.4rem'}),
                            dcc.Input(
                                id='frame-speed',
                                type='number',
                                value=600,
                                min=50,
                                max=60000,
                                step=50,
                                debounce=True,
                                style={'width': '78px', 'fontSize': '0.9em'},
                            ),
                            html.Span('ms', style={'marginLeft': '0.25rem', 'fontSize': '0.85em', 'color': '#555'}),
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '0.35rem'})
                    ]) if trajectory_mode else html.Div()),
                    html.Label('Isovalue fraction (of max)'),
                    dcc.Slider(
                        id='iso-frac',
                        min=0.03,
                           max=0.80,
                        step=0.01,
                        value=0.15,
                           marks={0.03: '0.03', 0.15: '0.15', 0.30: '0.30',
                               0.50: '0.50', 0.80: '0.80'}
                    ),
                    html.Div([
                        dcc.Checklist(id='label-toggle', options=[{'label': 'Labels', 'value': 'labels'}],
                                      value=['labels'], inline=True, style={'display': 'inline-block', 'marginRight': '1em'}),
                        dcc.Checklist(id='bond-toggle', options=[{'label': 'Bonds', 'value': 'bonds'}],
                                      value=['bonds'], inline=True, style={'display': 'inline-block'}),
                        (html.Span("DEBUG", style={
                         'color': 'crimson', 'fontWeight': 'bold', 'marginLeft': '0.75em'}) if debug else html.Span())
                    ], style={'marginTop': '0.5em'})
                ], style={'width': '100%', 'padding': '0 1em'})
            ], style={'display': 'flex', 'flexDirection': 'column'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='orbital-plot', figure=build_figure(mol, show_labels=True,
                              show_bonds=True, debug=debug, bond_scale=bond_scale, camera=default_camera, scene_ranges=scene_ranges, height=fig_height), style={'height': f'{fig_height}px'})
                ], style={'flex': '1'}),
                html.Div([
                    (dcc.Graph(id='pes-plot', figure=_pes_figure(0), style={'height': '220px', 'marginBottom': '0.5em'}) if show_pes else html.Div(style={'height': '0.5em'})),
                    html.Div([
                        html.Div([
                            html.Label('Quality (grid spacing)'),
                            dcc.Slider(id='quality', min=0, max=len(quality_values)-1, step=1,
                                       value=default_quality_index, vertical=True,
                                       marks=quality_marks, disabled=quality_slider_disabled,
                                       tooltip={'always_visible': True, 'placement': 'left'})  # type: ignore[arg-type]
                        ], style={'padding': '0 0.5em'}),
                        html.Div([
                            html.Label('Smooth (visual)'),
                            dcc.Slider(id='smooth', min=0, max=3, step=1,
                                       value=1, vertical=True,
                                       marks={0: '0', 1: '1', 2: '2', 3: '3'},
                                       tooltip={'always_visible': False, 'placement': 'left'})
                        ], style={'padding': '0 0.5em'}),
                        html.Div([
                            html.Label('Atom scale'),
                            dcc.Slider(id='atom-scale', min=0.4, max=1.6, step=0.05, value=1.0, vertical=True,
                                       marks={0.4: '0.4', 0.6: '0.6', 1.0: '1.0', 1.6: '1.6'}, tooltip={'always_visible': False, 'placement': 'left'})
                        ], style={'padding': '0 0.5em'}),
                        html.Div([
                            html.Label('Bond width'),
                            dcc.Slider(id='bond-width', min=2, max=18, step=1, value=10, vertical=True,
                                       marks={2: '2', 6: '6', 10: '10', 14: '14', 18: '18'}, tooltip={'always_visible': False, 'placement': 'left'})
                        ], style={'padding': '0 0.5em'})
                    ], style={'display': 'flex', 'flexDirection': 'row', 'height': f'{fig_height - 220}px', 'marginTop': '0.35em'})
                ], style={'width': '500px', 'paddingLeft': '0.4em'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start'}),
            html.Div(id='orbital-info', style={'marginTop': '0.5em'})
        ], style={
            'backgroundColor': 'white',
            'color': '#111',
            'minHeight': '100vh',
            'padding': '0.25rem 0.5rem',
        })

    def eye_norm(data):
        if not isinstance(data, dict):
            return None
        eye = data.get('scene.camera.eye')
        if isinstance(eye, dict):
            x = eye.get('x')
            y = eye.get('y')
            z = eye.get('z')
        else:
            x = data.get('scene.camera.eye.x')
            y = data.get('scene.camera.eye.y')
            z = data.get('scene.camera.eye.z')
        if x is None or y is None or z is None:
            return None
        try:
            return float((float(x)**2 + float(y)**2 + float(z)**2) ** 0.5)
        except (TypeError, ValueError):
            return None

    def extract_camera(data):
        if not isinstance(data, dict):
            return None
        cam = data.get('scene.camera')
        if isinstance(cam, dict):
            return cam
        has_any = False
        cam_out = {'eye': {}, 'center': {}, 'up': {}}
        for key in ['eye', 'center', 'up']:
            for comp in ['x', 'y', 'z']:
                v = data.get(f'scene.camera.{key}.{comp}')
                if v is not None:
                    has_any = True
                    cam_out[key][comp] = v
        return cam_out if has_any else None

    def _update_common(idx, frame_idx, iso_frac, quality_idx, smooth_passes, show_labels, show_bonds, atom_scale, bond_width, relayout, cam_state):
        spacing = quality_values[quality_idx]
        frame_idx = int(frame_idx or 0)

        if not trajectory_mode:
            assert evaluator is not None
            evaluator.regrid(spacing)

        current_camera = extract_camera(relayout or {})
        current_eye = eye_norm(relayout or {})
        base_eye = cam_state.get('base_eye_norm') if isinstance(
            cam_state, dict) else None
        if base_eye is None:
            base_eye = (0.9**2 + 0.9**2 + 0.6**2) ** 0.5
        zoom_factor = base_eye / \
            current_eye if (current_eye and current_eye > 0) else 1.0

        if trajectory_mode:
            grid, origin, steps, spacing_actual = _trajectory_grid(spacing)
            frame = frames[frame_idx]
            frame_mo = int(np.asarray(frame.mo_coeff).shape[1])
            idx_eff = int(max(0, min(int(idx), frame_mo - 1)))
            vals = _trajectory_values(frame_idx, idx_eff, spacing)
            data = OrbitalData(grid=grid, values=vals, origin=origin, steps=steps, spacing=spacing_actual)
            mol_local = frame_molecules[frame_idx]
        else:
            idx_eff = int(idx)
            assert evaluator is not None
            data = evaluator.orbital_data(idx_eff)
            mol_local = mol

        nx, ny, nz = data.steps
        V = data.values.reshape(nx, ny, nz)
        V_disp = _smooth_field_separable(V, int(smooth_passes or 0))
        values_disp = V_disp.reshape(-1)
        iso_val = float(iso_frac) * float(np.max(np.abs(data.values)))

        if verbose:
            logger.info(
                "UI update: idx=%d spacing=%.3f smooth=%d iso_frac=%.3f points=%d",
                int(idx_eff),
                float(spacing),
                int(smooth_passes or 0),
                float(iso_frac),
                int(data.grid.shape[0]),
            )

        data_disp = OrbitalData(
            grid=data.grid,
            values=values_disp,
            origin=data.origin,
            steps=data.steps,
            spacing=data.spacing,
        )
        atom_scale_eff = float(atom_scale) * float(zoom_factor)
        bond_width_eff = int(round(int(bond_width) * float(zoom_factor)))
        camera_to_apply = current_camera if current_camera is not None else (
            cam_state.get('camera') if isinstance(cam_state, dict) else default_camera)

        fig = build_figure(
            mol_local, data_disp, iso_val, show_labels=show_labels, show_bonds=show_bonds, debug=debug,
            atom_scale=atom_scale_eff, bond_width=bond_width_eff, bond_scale=bond_scale, camera=camera_to_apply,
            scene_ranges=scene_ranges, height=fig_height)

        bond_count = len(mol_local.infer_bonds(
            scale=float(bond_scale))) if show_bonds else 0
        info = f"Orbital {idx_eff} | max |psi| = {np.max(np.abs(data.values)):.3f} | spacing={spacing:.2f} Å | smooth={int(smooth_passes or 0)} | points={data.grid.shape[0]} | bonds={bond_count} | zoom×={zoom_factor:.2f} | showing 0..{n_mo_display-1} of {n_mo}"
        if trajectory_mode:
            info = f"Frame {frame_idx} | " + info
            frame = frames[frame_idx]
            if frame.mo_energies is not None and idx_eff < len(frame.mo_energies):
                info += f" | MO energy = {float(frame.mo_energies[idx_eff]):.6f} a.u."
            if frame.mo_occ is not None and idx_eff < len(frame.mo_occ):
                info += f" | occ = {float(frame.mo_occ[idx_eff]):.2f}"
            for lbl, arr in traj_pes_series.items():
                v = float(arr[int(frame_idx)])
                if not np.isnan(v):
                    info += f" | {lbl} = {v:.6f}"
        else:
            if mo_energies is not None:
                info += f" | energy = {mo_energies[idx_eff]:.6f} a.u."
            if mo_occ is not None:
                info += f" | occ = {mo_occ[idx_eff]:.2f}"
        if idx_eff in active_set:
            info += " | active"
        new_cam_state = {
            'base_eye_norm': base_eye,
            'last_eye_norm': current_eye or base_eye,
            'camera': camera_to_apply
        }
        if show_pes:
            return fig, _pes_figure(frame_idx), info, new_cam_state
        return fig, info, new_cam_state

    if use_minimal_ui:
        if show_pes:
            @app.callback(
                Output('orbital-plot', 'figure'),
                Output('pes-plot', 'figure'),
                Output('orbital-info', 'children'),
                Output('camera-state', 'data'),
                Input('orbital-idx', 'value'),
                Input('frame-idx', 'value'),
                Input('iso-frac', 'value'),
                Input('quality', 'value'),
                Input('orbital-plot', 'relayoutData'),
                Input('camera-state', 'data')
            )
            def update_orbital(idx, frame_idx, iso_frac, quality_idx, relayout, cam_state):  # type: ignore
                return _update_common(
                    idx=idx,
                    frame_idx=frame_idx,
                    iso_frac=iso_frac,
                    quality_idx=quality_idx,
                    smooth_passes=1,
                    show_labels=True,
                    show_bonds=True,
                    atom_scale=1.0,
                    bond_width=10,
                    relayout=relayout,
                    cam_state=cam_state,
                )
        else:
            @app.callback(
                Output('orbital-plot', 'figure'),
                Output('orbital-info', 'children'),
                Output('camera-state', 'data'),
                Input('orbital-idx', 'value'),
                Input('iso-frac', 'value'),
                Input('quality', 'value'),
                Input('orbital-plot', 'relayoutData'),
                Input('camera-state', 'data')
            )
            def update_orbital(idx, iso_frac, quality_idx, relayout, cam_state):  # type: ignore
                return _update_common(
                    idx=idx,
                    frame_idx=0,
                    iso_frac=iso_frac,
                    quality_idx=quality_idx,
                    smooth_passes=1,
                    show_labels=True,
                    show_bonds=True,
                    atom_scale=1.0,
                    bond_width=10,
                    relayout=relayout,
                    cam_state=cam_state,
                )
    else:
        if show_pes:
            @app.callback(
                Output('orbital-plot', 'figure'),
                Output('pes-plot', 'figure'),
                Output('orbital-info', 'children'),
                Output('camera-state', 'data'),
                Input('orbital-idx', 'value'),
                Input('frame-idx', 'value'),
                Input('iso-frac', 'value'),
                Input('quality', 'value'),
                Input('smooth', 'value'),
                Input('label-toggle', 'value'),
                Input('bond-toggle', 'value'),
                Input('atom-scale', 'value'),
                Input('bond-width', 'value'),
                Input('orbital-plot', 'relayoutData'),
                Input('camera-state', 'data')
            )
            def update_orbital(idx, frame_idx, iso_frac, quality_idx, smooth_passes, label_values, bond_values, atom_scale, bond_width, relayout, cam_state):  # type: ignore
                show_labels = 'labels' in (label_values or [])
                show_bonds = 'bonds' in (bond_values or [])
                return _update_common(
                    idx=idx,
                    frame_idx=frame_idx,
                    iso_frac=iso_frac,
                    quality_idx=quality_idx,
                    smooth_passes=smooth_passes,
                    show_labels=show_labels,
                    show_bonds=show_bonds,
                    atom_scale=atom_scale,
                    bond_width=bond_width,
                    relayout=relayout,
                    cam_state=cam_state,
                )
        else:
            @app.callback(
                Output('orbital-plot', 'figure'),
                Output('orbital-info', 'children'),
                Output('camera-state', 'data'),
                Input('orbital-idx', 'value'),
                Input('iso-frac', 'value'),
                Input('quality', 'value'),
                Input('smooth', 'value'),
                Input('label-toggle', 'value'),
                Input('bond-toggle', 'value'),
                Input('atom-scale', 'value'),
                Input('bond-width', 'value'),
                Input('orbital-plot', 'relayoutData'),
                Input('camera-state', 'data')
            )
            def update_orbital(idx, iso_frac, quality_idx, smooth_passes, label_values, bond_values, atom_scale, bond_width, relayout, cam_state):  # type: ignore
                show_labels = 'labels' in (label_values or [])
                show_bonds = 'bonds' in (bond_values or [])
                return _update_common(
                    idx=idx,
                    frame_idx=0,
                    iso_frac=iso_frac,
                    quality_idx=quality_idx,
                    smooth_passes=smooth_passes,
                    show_labels=show_labels,
                    show_bonds=show_bonds,
                    atom_scale=atom_scale,
                    bond_width=bond_width,
                    relayout=relayout,
                    cam_state=cam_state,
                )

    if trajectory_mode:
        @app.callback(
            Output('orbital-idx', 'marks'),
            Input('frame-idx', 'value'),
        )
        def update_orbital_marks(frame_idx):  # type: ignore
            i = int(max(0, min(int(frame_idx or 0), len(frames) - 1)))
            frame = frames[i]
            occ = np.asarray(frame.mo_occ).reshape(-1)
            en = np.asarray(frame.mo_energies).reshape(-1)
            homo_mark, lumo_mark, _ = _compute_homo_lumo(occ, en, n_mo_display)
            return _orbital_marks(n_mo_display, homo_mark, lumo_mark)

        @app.callback(
            Output('frame-play-state', 'data'),
            Output('play-frames', 'children'),
            Output('frame-interval', 'disabled'),
            Output('frame-interval', 'interval'),
            Input('play-frames', 'n_clicks'),
            Input('frame-speed', 'value'),
            State('frame-play-state', 'data'),
            State('frame-interval', 'interval'),
            prevent_initial_call=True,
        )
        def toggle_play(_n_clicks, speed_ms, playing, current_interval):  # type: ignore
            try:
                interval_ms = int(float(speed_ms)) if speed_ms is not None else int(current_interval or 600)
            except (TypeError, ValueError):
                interval_ms = int(current_interval or 600)
            # Guard rails for manual input.
            interval_ms = max(50, min(60000, interval_ms))
            triggered = getattr(dash, 'ctx', None)
            triggered_id = getattr(triggered, 'triggered_id', None) if triggered is not None else None
            if triggered_id == 'play-frames':
                is_playing = not bool(playing)
            else:
                is_playing = bool(playing)
            return is_playing, ('Pause' if is_playing else 'Play'), (not is_playing), interval_ms

        @app.callback(
            Output('frame-idx', 'value'),
            Input('frame-interval', 'n_intervals'),
            State('frame-idx', 'value'),
            State('frame-play-state', 'data'),
            prevent_initial_call=True,
        )
        def advance_frame(_n_intervals, frame_idx, playing):  # type: ignore
            if not bool(playing):
                return no_update
            max_frame = len(frames) - 1
            i = int(frame_idx or 0)
            return 0 if i >= max_frame else i + 1

    # Always print a user-friendly URL
    display_host = '127.0.0.1' if str(host) in (
        '0.0.0.0', '::') else str(host)
    url = f"http://{display_host}:{port}/"
    if auto_run:
        print(f"OrbitalDash running at: {url}", flush=True)
    else:
        print(
            f"OrbitalDash app initialized. Start the server to open: {url}", flush=True)

    # Dash >= (deprecation release) replaced run_server with run()/r
    # Provide broad compatibility across versions.
    if auto_run:
        if hasattr(app, 'run'):
            # In Dash versions that support notebooks, app.run accepts jupyter_mode.
            try:
                sig = inspect.signature(app.run)  # type: ignore[misc]
                supports_jupyter_mode = 'jupyter_mode' in sig.parameters
            except Exception:
                supports_jupyter_mode = False

            if use_jupyter and supports_jupyter_mode:
                app.run(debug=False, host=host, port=port,
                        jupyter_mode=str(jupyter_mode))
            else:
                # Fall back to standard server run if not in a notebook, or if the
                # installed Dash doesn't support notebook embedding.
                app.run(debug=False, host=host, port=port)
        elif hasattr(app, 'r'):
            # type: ignore[attr-defined]
            app.r(debug=False, host=host, port=port)
        else:  # fallback for older Dash
            # type: ignore[attr-defined]
            app.run_server(debug=False, host=host, port=port)
    return app


__all__ = ["launch_app"]
