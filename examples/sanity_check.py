import math
import numpy as np
from orbitaldash.geometry import Molecule
from orbitaldash.orbitals import OrbitalEvaluator

# Fake minimal system: H2 with dummy coefficients
symbols = ['H', 'H']
coords = np.array([[0, 0, 0], [0, 0, 0.74]])
# two AOs, two MOs (identity + bonding)
mo_coeff = np.array([[1.0, 1/np.sqrt(2)], [0.0, 1/np.sqrt(2)]])

mol = Molecule(symbols=symbols, coords=coords)

# Provide custom evaluator returning simple gaussian-like AOs to emulate shape


def custom_eval(grid):
    # two primitive gaussians centered on each H
    g1 = np.exp(- (np.linalg.norm(grid - coords[0], axis=1)**2) / 0.5)
    g2 = np.exp(- (np.linalg.norm(grid - coords[1], axis=1)**2) / 0.5)
    ao = np.stack([g1, g2], axis=1)
    return ao @ mo_coeff  # shape (npoints, n_mo)

# Wrap to give MO matrix shape indexing


class Wrapper:
    def __call__(self, grid):
        return custom_eval(grid)


w = Wrapper()

# Evaluator using custom function
# We present mo_coeff anyway for shape; only custom_eval used


# build dummy mo coeff for shape alignment
mo_coeff_dummy = np.eye(2)

# The custom evaluator returns shape (npoints, n_mo), we index inside OrbitalEvaluator

evaluator = OrbitalEvaluator(
    mol, basis='custom', mo_coeff=mo_coeff_dummy, custom_eval=w)
orb0 = evaluator.orbital_data(0)
orb1 = evaluator.orbital_data(1)
print('Grid points:', orb0.grid.shape[0])
print('Orb0 max amplitude:', np.max(np.abs(orb0.values)))
print('Orb1 max amplitude:', np.max(np.abs(orb1.values)))
