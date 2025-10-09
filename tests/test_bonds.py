import numpy as np
from orbitaldash.geometry import standardize_geometry


def test_water_bonds_present():
    # Same geometry as example (approximate water geometry)
    geom = {"symbols": ["O", "H", "H"], "coords": np.array([
        [0.0, 0.0, 0.0],
        [0.0, -0.757, 0.587],
        [0.0, 0.757, 0.587]
    ])}
    mol = standardize_geometry(geom)
    bonds = mol.infer_bonds()
    # Expect exactly two O-H bonds and no H-H bond (due to distance and radii)
    oh_pairs = {tuple(sorted(b)) for b in bonds if 0 in b}
    hh_pairs = {tuple(sorted(b)) for b in bonds if b[0] != 0 and b[1] != 0}
    assert len(oh_pairs) == 2, f"Expected 2 O-H bonds, got {oh_pairs}"
    assert len(hh_pairs) == 0, f"Unexpected H-H bond detected: {hh_pairs}"
