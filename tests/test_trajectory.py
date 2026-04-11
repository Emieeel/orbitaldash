import numpy as np
import pytest

from orbitaldash.geometry import Molecule
from orbitaldash.orbitals import (
    load_trajectory_hdf5_grid,
    load_trajectory_hdf5_values,
    make_global_grid,
    precompute_trajectory_to_hdf5,
    trajectory_cache_key,
)
from orbitaldash.trajectory import (
    TrajectoryData,
    TrajectoryFrame,
    frame_energies,
    frame_reaction_coordinates,
    standardize_trajectory,
)


def _frame(offset: float, basis=None, include_meta: bool = True) -> dict:
    coords = np.array([[0.0, 0.0, -0.4 + offset], [0.0, 0.0, 0.4 + offset]], dtype=float)

    def custom_eval(grid: np.ndarray) -> np.ndarray:
        x = grid[:, 0]
        z = grid[:, 2]
        # Two orbitals with a frame-dependent shift.
        return np.stack([x + offset, z - offset], axis=1)

    meta = {"energy": -1.0 + offset, "reaction_coordinate": 1.5 + offset} if include_meta else {}
    return {
        "geometry": {"symbols": ["H", "H"], "coords": coords},
        "mo_coeff": np.eye(2, dtype=float),
        "basis": basis,
        "mo_energies": np.array([-0.5 + offset, 0.2 + offset], dtype=float),
        "mo_occ": np.array([2.0, 0.0], dtype=float),
        "metadata": meta,
        "custom_eval": custom_eval,
    }


def test_standardize_trajectory_from_sequence_applies_default_basis():
    traj = standardize_trajectory([_frame(0.0, basis=None)], default_basis="sto-3g")
    assert isinstance(traj, TrajectoryData)
    assert len(traj.frames) == 1
    assert traj.frames[0].basis == "sto-3g"
    assert isinstance(traj.frames[0].geometry, Molecule)


def test_standardize_trajectory_mapping_uses_global_basis_and_metadata():
    traj = standardize_trajectory(
        {
            "frames": [_frame(0.0, basis=None), _frame(0.2, basis=None)],
            "global_basis": "cc-pvdz",
            "metadata": {"name": "scan"},
        }
    )
    assert traj.metadata["name"] == "scan"
    assert all(f.basis == "cc-pvdz" for f in traj.frames)


def test_standardize_trajectory_data_normalizes_existing_frames():
    data = TrajectoryData(
        frames=[
            TrajectoryFrame(geometry={"symbols": ["H"], "coords": np.array([[0.0, 0.0, 0.0]])}, mo_coeff=np.ones((1, 1)))
        ],
        global_basis="x",
    )
    out = standardize_trajectory(data)
    assert isinstance(out.frames[0].geometry, Molecule)
    assert out.frames[0].basis == "x"


def test_standardize_trajectory_errors_on_invalid_inputs():
    with pytest.raises(ValueError, match="Unsupported trajectory"):
        standardize_trajectory(42)

    with pytest.raises(ValueError, match="must include 'frames'"):
        standardize_trajectory({"metadata": {"x": 1}})

    with pytest.raises(ValueError, match="must contain at least one frame"):
        standardize_trajectory([])


def test_frame_energies_and_reaction_coordinates_defaults():
    traj = standardize_trajectory([_frame(0.0, include_meta=False), _frame(0.1, include_meta=False)])
    e = frame_energies(traj)
    rc = frame_reaction_coordinates(traj)
    assert np.isnan(e).all()
    assert np.array_equal(rc, np.array([0.0, 1.0]))


def test_make_global_grid_and_cache_key_behavior():
    traj = standardize_trajectory([_frame(0.0), _frame(0.2)])
    molecules = [f.geometry for f in traj.frames]
    grid, origin, steps, spacing = make_global_grid(molecules, spacing=0.4)
    assert grid.shape[1] == 3
    assert len(origin) == 3
    assert steps[0] > 0 and steps[1] > 0 and steps[2] > 0
    assert spacing == 0.4

    key1 = trajectory_cache_key(traj, spacing=0.4)
    key2 = trajectory_cache_key(traj, spacing=0.4)
    key3 = trajectory_cache_key(traj, spacing=0.3)
    assert key1 == key2
    assert key1 != key3


def test_make_global_grid_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        make_global_grid([])


def test_precompute_and_load_hdf5_roundtrip(tmp_path):
    traj = standardize_trajectory([_frame(0.0), _frame(0.2)])
    h5_path = tmp_path / "traj.h5"

    out_path = precompute_trajectory_to_hdf5(traj, h5_path=h5_path, spacing=0.35)
    assert out_path == str(h5_path)
    assert h5_path.exists()

    grid, origin, steps, spacing = load_trajectory_hdf5_grid(h5_path)
    assert grid.shape[1] == 3
    assert origin.shape == (3,)
    assert steps[0] * steps[1] * steps[2] == grid.shape[0]
    assert spacing == 0.35

    v00 = load_trajectory_hdf5_values(h5_path, frame_idx=0, orbital_idx=0)
    v10 = load_trajectory_hdf5_values(h5_path, frame_idx=1, orbital_idx=0)
    assert v00.shape == (grid.shape[0],)
    assert v10.shape == (grid.shape[0],)
    # Frame-dependent custom evaluator should produce different values.
    assert not np.allclose(v00, v10)


def test_precompute_raises_on_bad_custom_eval_shape(tmp_path):
    def bad_eval(grid: np.ndarray) -> np.ndarray:
        # Wrong second dimension: expected 2 orbitals.
        return np.zeros((grid.shape[0], 1), dtype=float)

    bad_frame = _frame(0.0)
    bad_frame["custom_eval"] = bad_eval
    traj = standardize_trajectory([bad_frame])

    with pytest.raises(ValueError, match="custom_eval"):
        precompute_trajectory_to_hdf5(traj, h5_path=tmp_path / "bad.h5", spacing=0.3)
