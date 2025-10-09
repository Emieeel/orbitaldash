import numpy as np
from orbitaldash.orbitals import OrbitalEvaluator


def test_grid_and_mo_shapes():
    geom = {"symbols": ["H", "H"], "coords": np.array(
        [[0, 0, 0], [0, 0, 0.74]])}
    # fake mo coeff (2 ao -> 2 mo identity)
    mo_coeff = np.eye(2)

    def custom_eval(grid):
        g1 = np.exp(-np.linalg.norm(grid - geom['coords'][0], axis=1)**2)
        g2 = np.exp(-np.linalg.norm(grid - geom['coords'][1], axis=1)**2)
        return np.stack([g1, g2], axis=1)

    evaluator = OrbitalEvaluator(
        geom, basis='custom', mo_coeff=mo_coeff, custom_eval=custom_eval)
    grid, origin, steps, spacing = evaluator.grid()
    assert grid.shape[1] == 3
    vals0 = evaluator.mo_values(0)
    vals1 = evaluator.mo_values(1)
    assert vals0.shape == vals1.shape
    # caching: second call should return same object reference
    vals0_again = evaluator.mo_values(0)
    assert vals0_again is vals0

    data = evaluator.orbital_data(0)
    assert data.values.shape[0] == grid.shape[0]
