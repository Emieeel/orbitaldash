import numpy as np
from orbitaldash.orbitals import OrbitalEvaluator
import threading


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


def test_orbital_data_consistent_under_regrid():
    geom = {"symbols": ["H", "H"], "coords": np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])}
    mo_coeff = np.eye(2)

    def custom_eval(grid):
        # Deterministic fake orbitals
        g1 = np.exp(-np.linalg.norm(grid - geom['coords'][0], axis=1) ** 2)
        g2 = np.exp(-np.linalg.norm(grid - geom['coords'][1], axis=1) ** 2)
        return np.stack([g1, g2], axis=1)

    evaluator = OrbitalEvaluator(
        geom, basis='custom', mo_coeff=mo_coeff, custom_eval=custom_eval)

    errors: list[BaseException] = []

    def worker_regrid():
        try:
            for k in range(40):
                evaluator.regrid(0.5 if (k % 2 == 0) else 0.45)
        except BaseException as e:
            errors.append(e)

    def worker_query():
        try:
            for _ in range(40):
                data = evaluator.orbital_data(0)
                nx, ny, nz = data.steps
                assert data.grid.shape[0] == nx * ny * nz
                assert data.values.shape[0] == data.grid.shape[0]
        except BaseException as e:
            errors.append(e)

    t1 = threading.Thread(target=worker_regrid)
    t2 = threading.Thread(target=worker_query)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    if errors:
        raise errors[0]
