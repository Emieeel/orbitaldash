"""Minimal trajectory example: H2 dissociation with a simple PES.

This example uses a lightweight toy orbital model (no PySCF needed) and
shows how to run trajectory mode with automatic HDF5 precompute cache.
"""

from __future__ import annotations

import numpy as np

from orbitaldash import launch_app


def morse_like_energy(r: float, de: float = 0.20, a: float = 1.6, re: float = 0.74) -> float:
    """Simple Morse-like potential in arbitrary units."""
    x = np.exp(-a * (r - re))
    return float(de * (1.0 - x) ** 2 - de)


def make_frame_custom_eval(r: float):
    """Return a simple 2-orbital model evaluator for H2 at bond length r."""
    centers = np.array([[0.0, 0.0, -0.5 * r], [0.0, 0.0, 0.5 * r]], dtype=float)

    def custom_eval(grid: np.ndarray) -> np.ndarray:
        d0 = np.linalg.norm(grid - centers[0], axis=1)
        d1 = np.linalg.norm(grid - centers[1], axis=1)
        # Two crude 1s-like atomic orbitals
        phi0 = np.exp(-1.2 * d0)
        phi1 = np.exp(-1.2 * d1)
        sigma = phi0 + phi1
        sigma_star = phi0 - phi1
        return np.stack([sigma, sigma_star], axis=1)

    return custom_eval


def build_trajectory(n_frames: int = 11):
    rs = np.linspace(0.50, 3.00, n_frames)
    frames = []
    for i, r in enumerate(rs):
        coords = np.array([[0.0, 0.0, -0.5 * r], [0.0, 0.0, 0.5 * r]], dtype=float)
        frames.append(
            {
                "geometry": {"symbols": ["H", "H"], "coords": coords},
                # Placeholder shape (n_ao, n_mo) for trajectory API consistency.
                "mo_coeff": np.eye(2, dtype=float),
                "custom_eval": make_frame_custom_eval(float(r)),
                "mo_energies": np.array([-0.5, 0.2], dtype=float),
                "mo_occ": np.array([2.0, 0.0], dtype=float),
                "metadata": {
                    "reaction_coordinate": float(r),
                    "energy": morse_like_energy(float(r)),
                    "label": f"r = {r:.2f} A",
                    "frame": i,
                },
            }
        )
    return frames


def main() -> None:
    trajectory = build_trajectory(n_frames=13)
    launch_app(
        trajectory=trajectory,
        mode="auto",
        cache_dir="./.orbitaldash_cache",
        auto_run=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
