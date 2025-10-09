"""Example: Benzene with uncontracted cc-pVDZ, launched in the orbital viewer.

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


def benzene_xyz() -> str:
    """Return a planar benzene geometry (C6H6) in Angstrom as a gto.M atom string."""
    # Common symmetric benzene coordinates (planar, z=0). Units: Angstrom.
    # C-C ~1.397 Å, C-H ~1.09 Å
    return (
        "C  1.40272  0.00000  0.00000; "
        "C  0.70136  1.21479  0.00000; "
        "C -0.70136  1.21479  0.00000; "
        "C -1.40272  0.00000  0.00000; "
        "C -0.70136 -1.21479  0.00000; "
        "C  0.70136 -1.21479  0.00000; "
        "H  2.49029  0.00000  0.00000; "
        "H  1.24515  2.15666  0.00000; "
        "H -1.24515  2.15666  0.00000; "
        "H -2.49029  0.00000  0.00000; "
        "H -1.24515 -2.15666  0.00000; "
        "H  1.24515 -2.15666  0.00000"
    )


def main():
    # Build a contracted cc-pVDZ molecule first
    mol = gto.M(
        atom=benzene_xyz(),
        basis="cc-pvdz",
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
    mf = scf.RHF(pmol).run()
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
