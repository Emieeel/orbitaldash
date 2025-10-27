"""Example: run HF on water with PySCF then launch orbital viewer.

Demonstrates two ways to launch:
1) New: pass the PySCF SCF object directly: launch_app(mf)
2) Original: pass geometry/basis/mo_coeff explicitly.
"""
import numpy as np
try:
    from pyscf import gto, scf  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install pyscf to run this example: pip install pyscf")

from orbitaldash import launch_app

BOHR_TO_ANG = 0.529177210903  # exact conversion


def build_water() -> "gto.M":  # type: ignore[name-defined]
    return gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis='sto-3g',
        unit='Angstrom',  # internal specification
        verbose=0
    )


def run_new_way():
    """Launch by passing the SCF object directly."""
    mol = build_water()
    mf = scf.RHF(mol).run()
    # Directly pass the SCF object; geometry/basis/MOs are inferred
    launch_app(mf, port=8051)


def run_old_way():
    """Launch by passing explicit geometry/basis/mo_coeff (original API)."""
    mol = build_water()
    mf = scf.RHF(mol).run()
    mo_coeff = mf.mo_coeff  # shape (n_ao, n_mo)
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    # IMPORTANT: Request Bohr then convert to Angstrom for clarity
    coords_bohr = mol.atom_coords(unit='Bohr')  # type: ignore[arg-type]
    geometry = {"symbols": ['O', 'H', 'H'],
                "coords": coords_bohr * BOHR_TO_ANG}
    launch_app(geometry=geometry, basis='sto-3g', mo_coeff=mo_coeff,
               mo_energies=mo_energy, mo_occ=mo_occ, port=8052)


if __name__ == "__main__":
    # Choose which demo to run.
    # By default, run the new, simpler call signature. Switch to run_old_way()
    # to see the original explicit form.
    run_new_way()
