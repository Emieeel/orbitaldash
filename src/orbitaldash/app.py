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

from .geometry import standardize_geometry, ELEMENT_COLORS, COVALENT_RADII
from .orbitals import OrbitalEvaluator

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go


def _molecule_traces(mol, show_labels: bool, show_bonds: bool, debug: bool = False,
                     atom_scale: float = 1.0, bond_width: int = 6):
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
    bonds = mol.infer_bonds() if show_bonds else []
    if bonds:
        bond_x = []
        bond_y = []
        bond_z = []
        for i, j in bonds:
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
            # Trim a fraction of each radius to meet the spheres more naturally
            trim_i = min(d * 0.45, ri * 0.85)
            trim_j = min(d * 0.45, rj * 0.85)
            q1 = p1 + u * trim_i
            q2 = p2 - u * trim_j
            bond_x += [q1[0], q2[0], None]
            bond_y += [q1[1], q2[1], None]
            bond_z += [q1[2], q2[2], None]
        # Map UI bond width so that UI 10 corresponds to previous visual width ~13
        width_eff = int(round(bond_width * 1.3))
        bond_trace = go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z,
            mode='lines',
            line=dict(
                color=('lime' if debug else 'rgba(70,70,70,0.9)'),
                width=(width_eff if not debug else max(width_eff, 8))
            ),
            name='Bonds',
            hoverinfo='skip'
        )
        # Draw bonds BEFORE atoms so they sit visually behind spheres
        return [bond_trace, atom_trace]
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
    # Plot both positive and negative isosurfaces (simple, no extra lighting)
    surf_pos = go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=V.flatten(),
        isomin=isovalue, isomax=V.max(), opacity=0.6,
        surface_count=2, colorscale='RdBu', showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Orbital+'
    )
    surf_neg = go.Isosurface(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=V.flatten(),
        isomin=V.min(), isomax=-isovalue, opacity=0.6,
        surface_count=2, colorscale='RdBu', showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name='Orbital-'
    )
    return [surf_neg, surf_pos]


def build_figure(mol, orbital_data=None, isovalue=None, show_labels: bool = True, show_bonds: bool = True, debug: bool = False,
                 atom_scale: float = 1.0, bond_width: int = 6, camera: Optional[dict] = None):
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
        camera: Optional Plotly camera dict to apply.

    Returns:
        plotly.graph_objects.Figure: The composed 3D figure.
    """
    traces = _molecule_traces(
        mol, show_labels, show_bonds=show_bonds, debug=debug,
        atom_scale=atom_scale, bond_width=bond_width)
    if orbital_data is not None:
        iso = isovalue
        if iso is None:
            # Default fraction consistent with initial slider (0.2)
            iso = 0.2 * np.max(np.abs(orbital_data.values))
        traces += _orbital_isosurface(orbital_data, iso)
    fig = go.Figure(data=traces)
    scene_cfg: dict[str, Any] = dict(
        xaxis=dict(visible=False, showspikes=False,
                   showgrid=False, zeroline=False, showbackground=False),
        yaxis=dict(visible=False, showspikes=False,
                   showgrid=False, zeroline=False, showbackground=False),
        zaxis=dict(visible=False, showspikes=False,
                   showgrid=False, zeroline=False, showbackground=False),
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
        margin=dict(l=0, r=0, b=0, t=30),
        height=650,
    )
    return fig


def launch_app(geometry, basis: str, mo_coeff, mo_energies: Optional[np.ndarray] = None, mo_occ: Optional[np.ndarray] = None,
               custom_eval: Optional[Callable[[
                   np.ndarray], np.ndarray]] = None,
               host='127.0.0.1', port=8050, auto_run: bool = True, debug: bool = False,
               max_orbitals: Optional[int] = 70):
    """Launch a Dash app to explore molecular orbitals interactively.

    The app renders a molecule with atom/bond controls and an orbital isosurface
    overlay. Orbital values are provided by :class:`~orbitaldash.orbitals.OrbitalEvaluator`.

    Args:
        geometry: Geometry accepted by ``standardize_geometry`` (dict, tuple, or Molecule).
        basis: Basis specification (string) or a PySCF mol-like object. Ignored if ``custom_eval`` is provided.
        mo_coeff: MO coefficient matrix with shape (n_ao, n_mo).
        mo_energies: Optional MO energies (n_mo,).
        mo_occ: Optional occupations (n_mo,) used to initialize to HOMO.
        custom_eval: Optional callable mapping grid (P, 3) to MO values (P, n_mo).
        host: Host interface for the Dash server.
        port: Port for the Dash server.
        auto_run: Whether to immediately start the server.
        debug: Enable debug styling for bonds.
        max_orbitals: Cap how many orbitals are exposed in the UI.

    Returns:
        dash.Dash: The configured Dash application instance.
    """
    mol = standardize_geometry(geometry)
    mo_coeff = np.asarray(mo_coeff)
    n_mo = mo_coeff.shape[1]
    # Limit how many orbitals are exposed in the UI by default to keep it usable
    if max_orbitals is not None:
        try:
            n_mo_display = max(1, min(int(max_orbitals), n_mo))
        except Exception:
            n_mo_display = min(70, n_mo)
    else:
        n_mo_display = n_mo
    # If custom_eval is provided, OrbitalEvaluator will use it and not require PySCF.
    evaluator = OrbitalEvaluator(mol, basis, mo_coeff, custom_eval=custom_eval)

    # Determine initial orbital index (HOMO if occupations provided)
    initial_idx = 0
    if mo_occ is not None:
        try:
            occ = np.asarray(mo_occ).reshape(-1)
            if occ.shape[0] == n_mo:
                # HOMO = highest index with occupation > tiny threshold
                occ_thr = 1e-6
                occupied = np.where(occ > occ_thr)[0]
                if occupied.size > 0:
                    initial_idx = int(occupied.max())
        except Exception:
            initial_idx = 0
    # Clamp initial selection into displayed range
    initial_idx = min(initial_idx, n_mo_display - 1)

    app = dash.Dash(__name__)

    # Quality slider spacings from 0.5 to 0.1 step 0.05 (0.5,0.45,...,0.1)
    quality_values = [round(x, 2) for x in np.arange(0.5, 0.099, -0.05)]
    quality_marks = {i: f"{s:.2f}" for i, s in enumerate(quality_values)}

    # Heuristic: if AO count is huge (e.g., uncontracted on large systems), start at a coarser grid
    # We infer AO count from mo_coeff rows
    n_ao = mo_coeff.shape[0]
    coarse_default = 0.25 if n_ao < 400 else (0.3 if n_ao < 800 else 0.35)
    try:
        default_quality_index = quality_values.index(coarse_default)
    except ValueError:
        # fallback to nearest available value
        default_quality_index = min(range(len(quality_values)), key=lambda i: abs(
            quality_values[i]-coarse_default))

    def _orbital_marks(n: int):
        # Reduce label clutter for large numbers of orbitals
        if n <= 20:
            step = 1
        elif n <= 50:
            step = 5
        else:
            step = 10
        marks = {i: str(i) for i in range(0, n, step)}
        marks.setdefault(n - 1, str(n - 1))
        return marks

    app.layout = html.Div([
        html.H3('Orbital Viewer'),
        dcc.Store(id='camera-state',
                  data={'base_eye_norm': None, 'last_eye_norm': None, 'camera': None}),
        html.Div([
            html.Div([
                html.Label(
                    f'Orbital index (showing 0..{n_mo_display-1} of total {n_mo})'),
                dcc.Slider(id='orbital-idx', min=0, max=n_mo_display-1, step=1, value=initial_idx,
                           marks=_orbital_marks(n_mo_display)),  # type: ignore[arg-type]
                html.Label('Isovalue fraction (of max)'),
                dcc.Slider(id='iso-frac', min=0.1, max=0.95, step=0.05, value=0.2,
                           marks={round(v, 2): str(round(v, 2)) for v in np.linspace(0.1, 0.95, 6)}),
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
                          show_bonds=True, debug=debug), style={'height': '650px'})
            ], style={'flex': '1'}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label('Quality (grid spacing)'),
                        dcc.Slider(id='quality', min=0, max=len(quality_values)-1, step=1,
                                   value=default_quality_index, vertical=True,
                                   marks=quality_marks, tooltip={'always_visible': True, 'placement': 'left'})  # type: ignore[arg-type]
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
                ], style={'display': 'flex', 'flexDirection': 'row', 'height': '650px'})
            ])
        ], style={'display': 'flex', 'flexDirection': 'row'}),
        html.Div(id='orbital-info', style={'marginTop': '0.5em'})
    ])

    @app.callback(
        Output('orbital-plot', 'figure'),
        Output('orbital-info', 'children'),
        Output('camera-state', 'data'),
        Input('orbital-idx', 'value'),
        Input('iso-frac', 'value'),
        Input('quality', 'value'),
        Input('label-toggle', 'value'),
        Input('bond-toggle', 'value'),
        Input('atom-scale', 'value'),
        Input('bond-width', 'value'),
        Input('orbital-plot', 'relayoutData'),
        Input('camera-state', 'data')
    )
    def update_orbital(idx, iso_frac, quality_idx, label_values, bond_values, atom_scale, bond_width, relayout, cam_state):  # type: ignore
        spacing = quality_values[quality_idx]
        evaluator.regrid(spacing)
        show_labels = 'labels' in (label_values or [])
        show_bonds = 'bonds' in (bond_values or [])
        # Derive zoom factor from camera eye changes

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

        # Extract full camera (eye/center/up) when available
        def extract_camera(data):
            if not isinstance(data, dict):
                return None
            cam = data.get('scene.camera')
            if isinstance(cam, dict):
                # May contain eye/center/up dicts
                return cam
            # Fallback: build from flattened keys if present
            has_any = False
            cam_out = {'eye': {}, 'center': {}, 'up': {}}
            for key in ['eye', 'center', 'up']:
                for comp in ['x', 'y', 'z']:
                    v = data.get(f'scene.camera.{key}.{comp}')
                    if v is not None:
                        has_any = True
                        cam_out[key][comp] = v
            return cam_out if has_any else None

        current_camera = extract_camera(relayout or {})
        current_eye = eye_norm(relayout or {})
        base_eye = cam_state.get('base_eye_norm') if isinstance(
            cam_state, dict) else None
        if base_eye is None:
            base_eye = (0.9**2 + 0.9**2 + 0.6**2) ** 0.5
        zoom_factor = base_eye / \
            current_eye if (current_eye and current_eye > 0) else 1.0

        data = evaluator.orbital_data(idx)
        iso_val = iso_frac * np.max(np.abs(data.values))
        # Apply zoom factor so atoms and bonds scale with zoom approximately
        atom_scale_eff = float(atom_scale) * float(zoom_factor)
        bond_width_eff = int(round(int(bond_width) * float(zoom_factor)))
        # Preserve camera across updates: prefer current relayout camera, else last stored
        camera_to_apply = current_camera if current_camera is not None else (
            cam_state.get('camera') if isinstance(cam_state, dict) else None)

        fig = build_figure(
            mol, data, iso_val, show_labels=show_labels, show_bonds=show_bonds, debug=debug,
            atom_scale=atom_scale_eff, bond_width=bond_width_eff, camera=camera_to_apply)
        bond_count = len(mol.infer_bonds()) if show_bonds else 0
        info = f"Orbital {idx} | max |psi| = {np.max(np.abs(data.values)):.3f} | spacing={spacing:.2f} Å | points={data.grid.shape[0]} | bonds={bond_count} | atomScale={atom_scale:.2f} | bondWidth={int(bond_width)} | zoom×={zoom_factor:.2f} | showing 0..{n_mo_display-1} of {n_mo}"
        if mo_energies is not None:
            info += f" | energy = {mo_energies[idx]:.6f} a.u."
        if mo_occ is not None:
            info += f" | occ = {mo_occ[idx]:.2f}"
        # Update camera store with base/last eye norms
        new_cam_state = {
            'base_eye_norm': base_eye,
            'last_eye_norm': current_eye or base_eye,
            'camera': camera_to_apply
        }
        return fig, info, new_cam_state

    # Dash >= (deprecation release) replaced run_server with run()/r
    # Provide broad compatibility across versions.
    if auto_run:
        if hasattr(app, 'run'):
            app.run(debug=False, host=host, port=port)
        elif hasattr(app, 'r'):
            # type: ignore[attr-defined]
            app.r(debug=False, host=host, port=port)
        else:  # fallback for older Dash
            # type: ignore[attr-defined]
            app.run_server(debug=False, host=host, port=port)
    return app


__all__ = ["launch_app"]
