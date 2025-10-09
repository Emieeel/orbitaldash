"""Example: Biphenyl with uncontracted cc-pVDZ, launched in the orbital viewer.

This example demonstrates how to run SCF on an uncontracted basis in PySCF and
then pass the exact PySCF mol object into `launch_app` to ensure the AO basis
used for grid evaluation matches the one that produced `mo_coeff`.

Requires: pyscf
"""
from __future__ import annotations

import numpy as np

try:
    from pyscf import gto, scf  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install pyscf to run this example: pip install pyscf")

from orbitaldash import launch_app

BOHR_TO_ANG = 0.529177210903  # exact conversion


def biphenyl_geometry() -> tuple[list[str], np.ndarray]:
    # Biphenyl geometry (C12H10), Angstrom. See e.g. PubChem CID 7095.
    symbols = [
        'H', 'H', 'H', 'C', 'H', 'H', 'H', 'C',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
        'C', 'C', 'H', 'H', 'H', 'H'
    ]
    coords = np.array([
        [0.,  2.1428, -0.9015],
        [0., -2.1428, -0.9015],
        [1.5152, -1.5152,  0.9015],
        [0.,  0.,  3.5119],
        [-1.5152,  1.5152,  0.9015],
        [1.5118, -1.5118,  3.3603],
        [-1.5118,  1.5118,  3.3603],
        [0.,  0.,  0.7416],
        [0.8532, -0.8532,  1.4399],
        [-0.8532,  0.8532,  1.4399],
        [0.,  0., -0.7416],
        [0.8510, -0.8510,  2.8187],
        [-0.8510,  0.8510,  2.8187],
        [0.,  0., -3.5119],
        [0.,  1.2036, -2.8187],
        [0., -1.2036, -2.8187],
        [0.,  1.2066, -1.4399],
        [0., -1.2066, -1.4399],
        [0.,  0., -4.5918],
        [0.,  2.1380, -3.3603],
        [0., -2.1380, -3.3603],
        [0.,  0.,  4.5918],
    ])
    return symbols, coords


def main():
    # Build a contracted cc-pVDZ molecule first
    symbols, coords = biphenyl_geometry()
    mol = gto.M(
        atom=list(zip(symbols, coords)),
        basis="unc-cc-pvdz",
        unit="Angstrom",
        verbose=0,
    )

    # Decontract the basis to produce an uncontracted Mole object (spherical)
    # pmol: uncontracted mol; ctr_coeff: per-shell contraction back-transform (unused here)
    pmol, ctr_coeff = mol.decontract_basis()  # type: ignore[assignment]
    # Ensure the uncontracted mol is initialized
    if getattr(pmol, "_built", 0) == 0:
        pmol.build(0, 0)

    # Run SCF on the uncontracted basis
    mf = scf.RHF(pmol)
    mf.chkfile = f'biphenyl.chk'
    mf.run()
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ

    # IMPORTANT: Use the same PySCF mol object (pmol) for evaluation to avoid AO/MO mismatch
    # Geometry must be in Angstrom for the viewer, PySCF gives Bohr by default
    coords_bohr = pmol.atom_coords(unit="Bohr")  # type: ignore[arg-type]
    geometry = {"symbols": [pmol.atom_pure_symbol(i) for i in range(pmol.natm)],
                "coords": coords_bohr * BOHR_TO_ANG}

    # Pass the uncontracted pmol directly as the "basis" argument
    launch_app(
        geometry=geometry,
        basis=pmol,  # pass PySCF mol to guarantee AO basis consistency
        mo_coeff=mo_coeff,
        mo_energies=mo_energy,
        mo_occ=mo_occ,
        port=8053,
    )


if __name__ == "__main__":
    main()
