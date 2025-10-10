"""
Example: Launch the app using a custom evaluator (no PySCF required).

This script fabricates two Gaussian-like orbitals for H2 and feeds them to the UI.
Run it after activating your environment:

  python examples/custom_eval_demo.py
"""

import numpy as np
from orbitaldash import launch_app

# H2 geometry in Angstrom
geometry = {"symbols": ["H", "H"], "coords": np.array(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])}

# For a custom evaluator, mo_coeff is only used to determine n_mo (columns)
# We'll set identity for two mock orbitals
mo_coeff = np.eye(2)

# Simple Gaussian lobes centered on each H; returns (npoints, n_mo)
# You can replace this with any callable that maps grid -> MO values

sigma2 = 0.25  # width control


def custom_eval(grid: np.ndarray) -> np.ndarray:
    g1 = np.exp(-np.linalg.norm(grid -
                geometry['coords'][0], axis=1) ** 2 / sigma2)
    g2 = np.exp(-np.linalg.norm(grid -
                geometry['coords'][1], axis=1) ** 2 / sigma2)
    return np.stack([g1, g2], axis=1)


if __name__ == "__main__":
    # We can pass any placeholder basis string since custom_eval will be used
    launch_app(geometry=geometry, basis="custom", mo_coeff=mo_coeff,
               custom_eval=custom_eval, debug=False)
