"""Precompute H2 dissociation trajectory to HDF5 cache.

Run:
    python examples/precompute_h2_trajectory_h5.py
"""

from __future__ import annotations

from pathlib import Path

from orbitaldash.orbitals import precompute_trajectory_to_hdf5
from orbitaldash.trajectory import standardize_trajectory

from h2_dissociation_trajectory import build_trajectory


def main() -> None:
    trajectory = standardize_trajectory(build_trajectory(n_frames=13))
    out = Path("./.orbitaldash_cache/h2_dissociation_precomputed.h5")
    path = precompute_trajectory_to_hdf5(
        trajectory,
        h5_path=out,
        spacing=0.25,
        padding=3.0,
        verbose=True,
    )
    print(f"Wrote precomputed cache: {path}")


if __name__ == "__main__":
    main()
