"""Utility to create a Molden file from user-supplied MO coefficients.

This wraps PySCF's molden writer but allows you to provide:
 - Geometry (any format PySCF `gto.M` accepts: string or list)
 - Basis set (name or dict)
 - Charge and spin (optional)
 - Molecular orbital coefficients (restricted or unrestricted)
 - (Optional) orbital energies and occupations

Supported coefficient shapes:
 - RHF / RKS like: 2D array (nao, nmo)
 - UHF / UKS like: tuple/list of two 2D arrays (nao, nmo_alpha) and (nao, nmo_beta)

If energies/occupations are not supplied they are omitted (Avogadro still shows orbitals).
If occupations are supplied with RHF and are length nmo, they are written.
"""
from __future__ import annotations
from typing import Union, Sequence, Optional
import numpy as np
from pyscf import gto
from pyscf.tools import molden

ArrayLike = np.ndarray
CoeffsType = Union[ArrayLike, Sequence[ArrayLike]]


def _validate_coeff_shape(mol, coeffs: CoeffsType):
    nao = mol.nao_nr()
    if isinstance(coeffs, (list, tuple)):
        if len(coeffs) != 2:
            raise ValueError(
                "Unrestricted coefficients must be a 2-sequence (alpha,beta)")
        ca, cb = coeffs
        ca = np.asarray(ca)
        cb = np.asarray(cb)
        if ca.ndim != 2 or cb.ndim != 2:
            raise ValueError("Alpha and beta coeff arrays must be 2D")
        if ca.shape[0] != nao or cb.shape[0] != nao:
            raise ValueError(
                f"First dimension of each coeff array must equal nao ({nao})")
        return 'UHF', (ca, cb)
    else:
        c = np.asarray(coeffs)
        if c.ndim != 2:
            raise ValueError("Restricted coeff matrix must be 2D (nao,nmo)")
        if c.shape[0] != nao:
            raise ValueError(f"First dimension (rows) must equal nao ({nao})")
        return 'RHF', c


def write_molden(
    geometry: Union[str, Sequence],
    basis: Union[str, dict],
    coeffs: CoeffsType,
    output: str = 'output.molden',
    charge: int = 0,
    spin: int = 0,
    mo_energies: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    mo_occ: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    ignore_h: bool = True,
):
    """Write a Molden file from supplied MO coefficients (RHF or UHF).

    Parameters
    ----------
    geometry : str | sequence
        Geometry specification accepted by PySCF (e.g. 'H 0 0 0; F 0 0 1.1').
    basis : str | dict
        Basis set spec accepted by PySCF.
    coeffs : array or (array,array)
        MO coefficient matrix/matrices. Shape (nao,nmo). For unrestricted, supply (C_alpha, C_beta).
    output : str
        Output filename (.molden extension recommended).
    charge : int
        Total molecular charge.
    spin : int
        2S value (#alpha - #beta). For closed-shell neutral even-electron molecules use 0.
    mo_energies : array or (array,array), optional
        Orbital energies. If provided length must match number of orbitals per spin.
    mo_occ : array or (array,array), optional
        Occupation numbers; if omitted they are inferred for closed-shell RHF when possible or left blank.
    ignore_h : bool
        Pass-through to PySCF molden writer (omit hydrogen basis detail for some viewers).
    """
    mol = gto.M(atom=geometry, basis=basis, charge=charge, spin=spin)
    kind, coeffs_obj = _validate_coeff_shape(mol, coeffs)

    # Build a lightweight dummy SCF-like object to hand to molden.from_scf / molden.dump_scf
    class DummySCF:
        def __init__(self, mol, kind, coeffs, mo_energy, mo_occ):
            self.mol = mol
            if kind == 'UHF':
                self.mo_coeff = (coeffs[0], coeffs[1])
                self.mo_energy = (
                    mo_energy[0] if mo_energy is not None else np.zeros(
                        coeffs[0].shape[1]),
                    mo_energy[1] if mo_energy is not None else np.zeros(
                        coeffs[1].shape[1]),
                )
                self.mo_occ = (
                    mo_occ[0] if mo_occ is not None else np.zeros(
                        coeffs[0].shape[1]),
                    mo_occ[1] if mo_occ is not None else np.zeros(
                        coeffs[1].shape[1]),
                )
            else:
                self.mo_coeff = coeffs
                nmo = coeffs.shape[1]
                self.mo_energy = mo_energy if mo_energy is not None else np.zeros(
                    nmo)
                if mo_occ is not None:
                    self.mo_occ = mo_occ
                else:
                    # Closed-shell guess: fill lowest nelec/2 orbitals doubly if possible
                    nelec = mol.nelectron
                    doubly = nelec // 2
                    occ = np.zeros(nmo)
                    occ[:doubly] = 2.0
                    self.mo_occ = occ

    # Normalize provided occupation arrays if necessary
    if mo_occ is not None:
        def _as_array(x):
            return np.asarray(x, dtype=float)
        if kind == 'UHF':
            if not isinstance(mo_occ, (list, tuple)) or len(mo_occ) != 2:
                raise ValueError(
                    "For UHF mo_occ must be a 2-sequence (alpha,beta)")
            mo_occ = (_as_array(mo_occ[0]), _as_array(mo_occ[1]))
        else:
            mo_occ = _as_array(mo_occ)

    if mo_energies is not None:
        def _as_array_e(x):
            return np.asarray(x, dtype=float)
        if kind == 'UHF':
            if not isinstance(mo_energies, (list, tuple)) or len(mo_energies) != 2:
                raise ValueError(
                    "For UHF mo_energies must be a 2-sequence (alpha,beta)")
            mo_energies = (_as_array_e(
                mo_energies[0]), _as_array_e(mo_energies[1]))
        else:
            mo_energies = _as_array_e(mo_energies)

    dummy = DummySCF(mol, kind, coeffs_obj, mo_energies, mo_occ)

    # Use lower-level molden.dump_scf (handles UHF vs RHF automatically)
    molden.dump_scf(dummy, output, ignore_h=ignore_h)
    return output


if __name__ == '__main__':
    # Demonstration with a water molecule
    geom = 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587'
    basis = 'sto-3g'
    mol = gto.M(atom=geom, basis=basis)
    nao = mol.nao_nr()
    nmo = nao
    # Random orthonormal coefficient matrix
    rng = np.random.default_rng(0)
    A = rng.standard_normal((nao, nmo))
    S = mol.intor('int1e_ovlp')
    M = A.T @ S @ A
    evals, evecs = np.linalg.eigh(M)
    C = A @ evecs @ np.diag(evals**-0.5)
    write_molden(geom, basis, C, output='water_random.molden')
    print('Wrote water_random.molden')
