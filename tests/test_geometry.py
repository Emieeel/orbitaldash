import numpy as np
from orbitaldash.geometry import standardize_geometry, Molecule


def test_standardize_from_dict():
    geom = {"symbols": ["O", "H", "H"], "coords": np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)}
    mol = standardize_geometry(geom)
    assert isinstance(mol, Molecule)
    assert mol.coords.shape == (3, 3)
    assert mol.symbols == ["O", "H", "H"]


def test_standardize_from_atomic_numbers():
    Z = [8, 1, 1]
    coords = np.zeros((3, 3))
    mol = standardize_geometry((Z, coords))
    assert mol.symbols == ["O", "H", "H"]


def test_bond_inference_simple_water():
    geom = {"symbols": ["O", "H", "H"], "coords": np.array(
        [[0, 0, 0], [0, 0, 0.96], [0, 0.76, -0.26]])}
    mol = standardize_geometry(geom)
    bonds = mol.infer_bonds()
    # Expect at least two O-H bonds
    oh_bonds = [b for b in bonds if 0 in b]
    assert len(oh_bonds) >= 2
