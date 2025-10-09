"""Example: run HF on water with PySCF then launch orbital viewer."""
import numpy as np
try:
    from pyscf import gto, scf  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install pyscf to run this example: pip install pyscf")

from orbitaldash import launch_app

BOHR_TO_ANG = 0.529177210903  # exact conversion

mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis='sto-3g',
    unit='Angstrom',  # internal specification
    verbose=0
)

mf = scf.RHF(mol).run()
mo_coeff = mf.mo_coeff  # shape (n_ao, n_mo)
mo_energy = mf.mo_energy
mo_occ = mf.mo_occ

# IMPORTANT: PySCF's mol.atom_coords() returns coordinates in Bohr by default
# unless a unit argument is supplied. We explicitly request Bohr then convert
# to Angstrom for clarity and robustness.
coords_bohr = mol.atom_coords(unit='Bohr')  # type: ignore[arg-type]
geometry = {"symbols": ['O', 'H', 'H'], "coords": coords_bohr * BOHR_TO_ANG}

launch_app(geometry=geometry, basis='sto-3g', mo_coeff=mo_coeff,
           mo_energies=mo_energy, mo_occ=mo_occ)
