"""Debug example to inspect bond inference distances for water.

Run:
    python examples/h2o_debug.py

It will print pairwise distances and launch the orbital viewer in debug mode.
"""
from __future__ import annotations
import numpy as np

from orbitaldash import launch_app

try:
    from pyscf import gto, scf  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install pyscf to run this example: pip install pyscf")

BOHR_TO_ANG = 0.529177210903

# Build PySCF molecule in Angstrom explicitly
mol = gto.M(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="sto-3g",
    unit="Angstrom",
    verbose=0,
)

mf = scf.RHF(mol).run()
mo_coeff = mf.mo_coeff
mo_energy = mf.mo_energy
mo_occ = mf.mo_occ

coords_ang = mol.atom_coords()  # already in Angstrom
symbols = ['O', 'H', 'H']

# Construct a BOHR version intentionally (to test auto detection ideas)
coords_bohr = coords_ang / BOHR_TO_ANG

print("Distances (Angstrom coordinates):")
for i in range(len(coords_ang)):
    for j in range(i+1, len(coords_ang)):
        d = np.linalg.norm(coords_ang[i] - coords_ang[j])
        print(f"  {i}-{j}: {d:.4f} Å")

print("\nDistances (Bohr coordinates, if misinterpreted as Å):")
for i in range(len(coords_bohr)):
    for j in range(i+1, len(coords_bohr)):
        d = np.linalg.norm(coords_bohr[i] - coords_bohr[j])
        print(f"  {i}-{j}: {d:.4f} (would be Å if not converted)")

print("\nLaunching viewer with correct Angstrom geometry (should show bonds=2).")
geometry_ang = {"symbols": symbols, "coords": coords_ang}

# You can toggle the line below to test the Bohr case (expect bonds=0):
# geometry_ang = {"symbols": symbols, "coords": coords_bohr}  # WRONG on purpose

launch_app(geometry=geometry_ang, basis='sto-3g', mo_coeff=mo_coeff,
           mo_energies=mo_energy, mo_occ=mo_occ, debug=True)
