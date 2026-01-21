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


def test_n2_triple_bond_order():
    # Approximate N2 equilibrium geometry
    geom = {"symbols": ["N", "N"], "coords": np.array(
        [[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]])}
    mol = standardize_geometry(geom)
    bonds = mol.infer_bonds(with_order=True)
    assert len(bonds) == 1
    i, j, order = bonds[0]
    assert {i, j} == {0, 1}
    # Visualization heuristic should classify this as triple
    assert order == 3
