import json
import logging
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
from werkzeug.serving import make_server

from orbitaldash import launch_app


def _serialize_dependency(dep):
    if isinstance(dep, dict):
        return {"id": dep.get("id"), "property": dep.get("property")}
    # dash.dependencies.Input/State
    cid = getattr(dep, "component_id", None)
    prop = getattr(dep, "component_property", None)
    if cid is not None and prop is not None:
        return {"id": cid, "property": prop}
    raise TypeError(f"Unsupported dependency spec type: {type(dep)!r}")


def _callback_output_key(app, required_tokens: list[str]) -> str:
    for key in app.callback_map.keys():
        if all(t in key for t in required_tokens):
            return key
    raise AssertionError("Could not find expected callback in callback_map")


def _build_update_payload_from_callback(app, output_key: str, *, overrides_by_id_prop: dict[tuple[str, str], object] | None = None):
    cb = app.callback_map[output_key]
    outputs_spec = cb.get("outputs")
    if outputs_spec is None:
        outputs_spec = cb.get("output")
    if outputs_spec is None:
        raise AssertionError("Could not determine callback outputs spec")
    if not isinstance(outputs_spec, list):
        outputs_spec = [outputs_spec]
    outputs_json = [_serialize_dependency(o) for o in outputs_spec]

    inputs_spec = cb.get("inputs") or []
    if not isinstance(inputs_spec, list):
        inputs_spec = [inputs_spec]
    inputs_json = []
    for dep in inputs_spec:
        d = _serialize_dependency(dep)
        cid = d["id"]
        prop = d["property"]

        value = None
        if overrides_by_id_prop is not None and (cid, prop) in overrides_by_id_prop:
            value = overrides_by_id_prop[(cid, prop)]
        else:
            comp = _find_component_by_id(app.layout, str(cid))
            if comp is not None and hasattr(comp, prop):
                value = getattr(comp, prop)
            else:
                # Common case: relayout leads to None in tests
                value = None

        inputs_json.append({"id": cid, "property": prop, "value": value})

    state_spec = cb.get("state") or []
    if not isinstance(state_spec, list):
        state_spec = [state_spec]
    state_json = []
    for dep in state_spec:
        d = _serialize_dependency(dep)
        cid = d["id"]
        prop = d["property"]
        value = None
        if overrides_by_id_prop is not None and (cid, prop) in overrides_by_id_prop:
            value = overrides_by_id_prop[(cid, prop)]
        else:
            comp = _find_component_by_id(app.layout, str(cid))
            if comp is not None and hasattr(comp, prop):
                value = getattr(comp, prop)
        state_json.append({"id": cid, "property": prop, "value": value})

    changed = [
        f"{inputs_json[0]['id']}.{inputs_json[0]['property']}"] if inputs_json else []
    return {
        "output": output_key,
        "outputs": outputs_json,
        "inputs": inputs_json,
        "state": state_json,
        "changedPropIds": changed,
    }


def _find_component_by_id(component, target_id: str):
    """Best-effort recursive search through Dash component tree."""
    if component is None:
        return None
    if getattr(component, "id", None) == target_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        for child in children:
            found = _find_component_by_id(child, target_id)
            if found is not None:
                return found
        return None

    return _find_component_by_id(children, target_id)


def _http_get(url: str, timeout_s: float = 5.0):
    req = urllib.request.Request(url=url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
        return resp.status, resp.headers, body


def _http_post_json(url: str, payload: dict, timeout_s: float = 10.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        method="POST",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read()
        return resp.status, resp.headers, body


@pytest.fixture()
def dash_server_url():
    # Minimal geometry (Angstrom)
    geometry = {
        "symbols": ["H", "H"],
        "coords": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
    }

    # mo_coeff is required by launch_app even when custom_eval is used.
    mo_coeff = np.zeros((1, 2), dtype=float)

    def custom_eval(grid: np.ndarray) -> np.ndarray:
        # Deterministic, non-trivial field; returns (P, n_mo)
        x = grid[:, 0]
        y = grid[:, 1]
        z = grid[:, 2]
        mo0 = np.sin(x) + np.cos(y) + 0.1 * z
        mo1 = np.cos(x) - np.sin(y) + 0.2 * z
        return np.stack([mo0, mo1], axis=1)

    app = launch_app(
        geometry=geometry,
        basis=None,
        mo_coeff=mo_coeff,
        custom_eval=custom_eval,
        auto_run=False,
        port=0,
        minimal_ui=True,
        verbose=True,
    )

    server = make_server("127.0.0.1", 0, app.server)
    port = int(server.server_port)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"

    # Wait until server is reachable
    deadline = time.time() + 5.0
    last_exc = None
    while time.time() < deadline:
        try:
            status, _headers, _body = _http_get(f"{base_url}/")
            if status == 200:
                break
        except Exception as e:  # pragma: no cover
            last_exc = e
            time.sleep(0.05)
    else:  # pragma: no cover
        raise RuntimeError("Dash server did not start") from last_exc

    try:
        yield base_url, app
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


@pytest.fixture()
def dash_server_url_full_ui():
    geometry = {
        "symbols": ["H", "H"],
        "coords": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
    }
    mo_coeff = np.zeros((1, 4), dtype=float)
    mo_energies = np.array([-0.9, -0.5, 0.1, 0.3], dtype=float)
    mo_occ = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)

    def custom_eval(grid: np.ndarray) -> np.ndarray:
        x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
        # (P, n_mo)
        out = np.stack(
            [
                np.sin(x) + 0.1 * z,
                np.cos(y) + 0.2 * z,
                np.sin(x + y) + 0.3 * z,
                np.cos(x - y) + 0.4 * z,
            ],
            axis=1,
        )
        return out

    app = launch_app(
        geometry=geometry,
        basis=None,
        mo_coeff=mo_coeff,
        mo_energies=mo_energies,
        mo_occ=mo_occ,
        active=[1],
        custom_eval=custom_eval,
        auto_run=False,
        port=0,
        minimal_ui=False,
        verbose=True,
    )

    server = make_server("127.0.0.1", 0, app.server)
    port = int(server.server_port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    deadline = time.time() + 5.0
    last_exc = None
    while time.time() < deadline:
        try:
            status, _headers, _body = _http_get(f"{base_url}/")
            if status == 200:
                break
        except Exception as e:  # pragma: no cover
            last_exc = e
            time.sleep(0.05)
    else:  # pragma: no cover
        raise RuntimeError("Dash server did not start") from last_exc

    try:
        yield base_url, app
    finally:
        server.shutdown()
        thread.join(timeout=5.0)
        server.server_close()


def test_launch_app_serves_dash_pages(dash_server_url):
    base_url, _app = dash_server_url

    status, _headers, body = _http_get(f"{base_url}/")
    assert status == 200
    html = body.decode("utf-8", errors="replace")
    # Dash renders the actual layout client-side; the initial HTML is mostly a shell.
    assert "DashRenderer" in html

    status, _headers, body = _http_get(f"{base_url}/_dash-layout")
    assert status == 200
    layout = json.loads(body.decode("utf-8"))
    # Sanity check that Dash layout JSON looks valid
    assert isinstance(layout, dict)
    assert "props" in layout

    status, _headers, body = _http_get(f"{base_url}/_dash-dependencies")
    assert status == 200
    deps = json.loads(body.decode("utf-8"))
    assert isinstance(deps, list)


def test_launch_app_verbose_logs_on_callback_update(dash_server_url, caplog):
    base_url, app = dash_server_url

    caplog.set_level(logging.INFO, logger="orbitaldash.app")
    caplog.set_level(logging.INFO, logger="orbitaldash.orbitals")

    output_key = _callback_output_key(
        app, ["orbital-plot", "figure", "camera-state"])
    payload = _build_update_payload_from_callback(app, output_key)

    status, _headers, body = _http_post_json(
        f"{base_url}/_dash-update-component", payload)
    assert status == 200

    resp = json.loads(body.decode("utf-8"))
    assert isinstance(resp, dict)

    # Verbose=True should emit at least the UI update log line.
    assert "UI update:" in caplog.text


def test_launch_app_full_ui_layout_and_callback(dash_server_url_full_ui, caplog):
    base_url, app = dash_server_url_full_ui

    # Exercise the full UI layout branch.
    status, _headers, body = _http_get(f"{base_url}/_dash-layout")
    assert status == 200
    layout_json = body.decode("utf-8", errors="replace")
    # IDs that exist only in the non-minimal UI.
    for expected in ("smooth", "label-toggle", "bond-toggle", "atom-scale", "bond-width"):
        assert expected in layout_json

    caplog.set_level(logging.INFO, logger="orbitaldash.app")
    caplog.set_level(logging.INFO, logger="orbitaldash.orbitals")

    output_key = _callback_output_key(
        app, ["orbital-plot", "figure", "camera-state"])
    # Force a smooth=0 update so we cover that branch too.
    payload = _build_update_payload_from_callback(
        app,
        output_key,
        overrides_by_id_prop={
            ("orbital-idx", "value"): 1,
            ("smooth", "value"): 0,
        },
    )
    status, _headers, _body = _http_post_json(
        f"{base_url}/_dash-update-component", payload)
    assert status == 200
    assert "UI update:" in caplog.text


def test_homo_initialization_from_energies_and_occ():
    # HOMO definition in app.py: highest energy among occ > 1.0
    geometry = {
        "symbols": ["H", "H"],
        "coords": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
    }
    mo_coeff = np.zeros((1, 4), dtype=float)
    mo_occ = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)
    mo_energies = np.array([-1.0, -0.5, 0.2, 0.3], dtype=float)

    def custom_eval(grid: np.ndarray) -> np.ndarray:
        return np.zeros((grid.shape[0], 4), dtype=float)

    app = launch_app(
        geometry=geometry,
        basis=None,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ,
        mo_energies=mo_energies,
        custom_eval=custom_eval,
        auto_run=False,
        minimal_ui=True,
    )

    slider = _find_component_by_id(app.layout, "orbital-idx")
    assert slider is not None
    assert int(getattr(slider, "value")) == 1


def test_info_includes_energy_occ_and_active(dash_server_url_full_ui):
    base_url, app = dash_server_url_full_ui

    output_key = _callback_output_key(
        app, ["orbital-plot", "figure", "camera-state"])
    payload = _build_update_payload_from_callback(
        app,
        output_key,
        overrides_by_id_prop={
            ("orbital-idx", "value"): 1,
        },
    )
    status, _headers, body = _http_post_json(
        f"{base_url}/_dash-update-component", payload)
    assert status == 200
    resp = json.loads(body.decode("utf-8"))
    txt = json.dumps(resp)
    # The orbital info string is returned as the output for orbital-info.children
    assert "energy =" in txt
    assert "occ =" in txt
    assert "active" in txt


def test_build_figure_traces_and_isosurfaces():
    from orbitaldash.geometry import standardize_geometry
    from orbitaldash.orbitals import OrbitalData
    from orbitaldash.app import build_figure

    mol = standardize_geometry(
        {
            "symbols": ["H", "H"],
            "coords": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
        }
    )

    # Small 2x2x2 grid
    xs = np.array([-0.5, 0.5])
    ys = np.array([-0.5, 0.5])
    zs = np.array([-0.5, 0.5])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    values = (grid[:, 0] - grid[:, 1] + 0.1 * grid[:, 2]).astype(float)
    data = OrbitalData(
        grid=grid,
        values=values,
        origin=np.array([xs.min(), ys.min(), zs.min()]),
        steps=(2, 2, 2),
        spacing=1.0,
    )

    fig = build_figure(mol, orbital_data=data, isovalue=0.2,
                       show_labels=True, show_bonds=True)
    # Expect: bond trace(s) + atom trace + 2 isosurfaces
    assert len(fig.data) >= 4

    fig2 = build_figure(mol, orbital_data=data, isovalue=0.2,
                        show_labels=False, show_bonds=False)
    # No bonds => 1 atom trace + 2 isosurfaces
    assert len(fig2.data) == 3


def test_smooth_field_separable_branches():
    from orbitaldash.app import _smooth_field_separable

    V = np.zeros((4, 4, 4), dtype=float)
    V[2, 2, 2] = 1.0
    out0 = _smooth_field_separable(V, passes=0)
    assert out0 is V
    out1 = _smooth_field_separable(V, passes=1)
    assert out1.shape == V.shape
    assert float(out1.max()) <= 1.0
