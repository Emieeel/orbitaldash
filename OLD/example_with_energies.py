"""Example: generating a Molden file with explicit MO energies and occupations.

Run this file to produce `hf_with_meta.molden` which includes HF orbital
energies and occupations. Avogadro will display these values in the MO /
Surfaces panel when you open the file.

How to view in Avogadro:
 1. File -> Open -> select hf_with_meta.molden
 2. Open the "Surfaces" (or "Orbitals") dialog
 3. Choose an orbital index; the energy (in Hartree) and occupation appear.

Note: Avogadro (via Open Babel) reads the Molden [MO] blocks. Energies are
written after the Ene= token and occupations after the Occup= token. This
script ensures they are populated by supplying them to `write_molden`.
"""
from __future__ import annotations
import numpy as np
from pyscf import gto, scf
from molden_writer import write_molden


def main():
    geom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
    basis = '6-31g'
    mol = gto.M(atom=geom, basis=basis, charge=0, spin=0)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError(
            'SCF did not converge; cannot write Molden file with MO metadata.')

    # Extract canonical MO data (ensure numpy arrays)
    coeffs = np.asarray(mf.mo_coeff)
    energies = np.asarray(mf.mo_energy)   # Hartree
    occ = np.asarray(mf.mo_occ)           # Typically 2 or 0 for RHF

    # Optionally, you could reorder or select a subset here.

    out_file = write_molden(
        geometry=geom,
        basis=basis,
        coeffs=coeffs,
        output='hf_with_meta.molden',
        charge=0,
        spin=0,
        mo_energies=energies,
        mo_occ=occ,
        ignore_h=True,
    )
    print(f'Wrote {out_file}')
    print('Open this file in Avogadro to inspect energies and occupations.')


if __name__ == '__main__':
    main()
