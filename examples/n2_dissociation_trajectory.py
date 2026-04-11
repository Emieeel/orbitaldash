"""N2 dissociation trajectory with CAS(6,6) valence active space and NOs.

Workflow:
1) Compute RHF at each geometry for a real HF PES.
2) Run CASCI(6,6) using a symmetry-selected valence guess
    (sigma, 2*pi, 2*pi*, sigma*).
3) Build natural orbitals (NOs) from PySCF and align active NOs to a
    reference frame so active-orbital identities stay consistent along the scan.
4) Visualize the aligned NO coefficients in OrbitalDash.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pyscf import gto, scf, mcscf, symm
from pyscf.mcscf import addons

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from orbitaldash import launch_app


def _select_valence_cas_guess_indices(mol, hf, ncas: int) -> list[int]:
    """Select valence CAS guess indices from RHF orbitals using symmetry labels.

    Strategy (Dooh-like naming):
    - sigma (bonding): highest-energy occupied A1g
    - pi (bonding): two highest-energy occupied E1u
    - pi* (antibonding): two lowest-energy virtual E1g
    - sigma* (antibonding): lowest-energy virtual A1u

    For CAS(6,6), this yields exactly 6 valence-like orbitals.
    """
    irrep_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
    irrep_names = [symm.irrep_id2name(mol.groupname, int(i)) for i in irrep_ids]
    mo_e = np.asarray(hf.mo_energy, dtype=float)
    mo_occ = np.asarray(hf.mo_occ, dtype=float)

    def _is_pi_u(name: str) -> bool:
        return name.startswith('E1u')

    def _is_pi_g(name: str) -> bool:
        return name.startswith('E1g')

    occ_sigma_g = [i for i, n in enumerate(irrep_names) if n == 'A1g' and mo_occ[i] > 1.0]
    occ_pi_u = [i for i, n in enumerate(irrep_names) if _is_pi_u(n) and mo_occ[i] > 1.0]
    virt_pi_g = [i for i, n in enumerate(irrep_names) if _is_pi_g(n) and mo_occ[i] <= 1.0]
    virt_sigma_u = [i for i, n in enumerate(irrep_names) if n == 'A1u' and mo_occ[i] <= 1.0]

    occ_pi_u.sort(key=lambda i: mo_e[i], reverse=True)
    virt_pi_g.sort(key=lambda i: mo_e[i])
    occ_sigma_g.sort(key=lambda i: mo_e[i], reverse=True)
    virt_sigma_u.sort(key=lambda i: mo_e[i])

    if ncas != 6:
        raise RuntimeError(f"This selector is implemented for CAS(6,6); got ncas={ncas}.")

    if len(occ_sigma_g) < 1 or len(occ_pi_u) < 2 or len(virt_pi_g) < 2 or len(virt_sigma_u) < 1:
        raise RuntimeError(
            "Could not find enough symmetry-selected valence orbitals for CAS(6,6). "
            f"Counts: occ A1g={len(occ_sigma_g)}, occ E1u={len(occ_pi_u)}, "
            f"virt E1g={len(virt_pi_g)}, virt A1u={len(virt_sigma_u)}."
        )

    selected = [
        occ_sigma_g[0],
        occ_pi_u[0], occ_pi_u[1],
        virt_pi_g[0], virt_pi_g[1],
        virt_sigma_u[0],
    ]

    # Deduplicate while preserving order.
    uniq = []
    seen = set()
    for i in selected:
        if i not in seen:
            uniq.append(int(i))
            seen.add(int(i))

    if len(uniq) < ncas:
        raise RuntimeError(
            f"Could not find enough symmetry-selected valence orbitals for CAS({ncas}, ...). "
            f"Found {len(uniq)} candidates."
        )

    return uniq[:ncas]


def _compute_alignment_order(ref_block: np.ndarray, cur_block: np.ndarray, s_cross: np.ndarray) -> list[int]:
    """Return column order that best aligns current active block to reference."""
    overlap = ref_block.T @ s_cross @ cur_block
    abs_overlap = np.abs(overlap)
    n = abs_overlap.shape[0]

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(-abs_overlap)
        order = [0] * n
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            order[int(r)] = int(c)
        return order

    # Fallback greedy matching if scipy is unavailable.
    order = []
    used: set[int] = set()
    for r in range(n):
        row = abs_overlap[r].copy()
        while True:
            c = int(np.argmax(row))
            if c not in used:
                order.append(c)
                used.add(c)
                break
            row[c] = -1.0
    return order


def _align_active_no_block(
    mo_no: np.ndarray,
    mo_energy: np.ndarray,
    mo_occ: np.ndarray,
    active_cols: list[int],
    mol,
    ref_active_block: np.ndarray,
    ref_mol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align active NO columns to the reference frame using AO overlap."""
    s_cross = gto.intor_cross('int1e_ovlp', ref_mol, mol)

    cur_block = mo_no[:, active_cols]
    order = _compute_alignment_order(ref_active_block, cur_block, s_cross)
    aligned_block = cur_block[:, order].copy()

    # Fix sign phase for each aligned active orbital.
    for i in range(aligned_block.shape[1]):
        s = float(ref_active_block[:, i].T @ s_cross @ aligned_block[:, i])
        if s < 0:
            aligned_block[:, i] *= -1.0

    mo_no_aligned = np.asarray(mo_no, dtype=float).copy()
    mo_no_aligned[:, active_cols] = aligned_block

    e_aligned = np.asarray(mo_energy, dtype=float).copy()
    occ_aligned = np.asarray(mo_occ, dtype=float).copy()
    e_aligned[active_cols] = e_aligned[np.asarray(active_cols)[order]]
    occ_aligned[active_cols] = occ_aligned[np.asarray(active_cols)[order]]
    return mo_no_aligned, e_aligned, occ_aligned


def build_frame(
    r_nn: float,
    cas_guess_idx: Optional[list[int]],
    ncas: int,
    nelecas: int,
    ref_active_block: Optional[np.ndarray],
    ref_mol,
):
    """Build one trajectory frame and aligned NO coefficients at R(N-N)."""
    mol = gto.M(
        atom=f"N 0 0 {-0.5 * r_nn}; N 0 0 {0.5 * r_nn}",
        basis="cc-pvdz",
        symmetry=True,
        unit="Angstrom",
        verbose=0,
    )

    hf = scf.RHF(mol)
    hf.conv_tol = 1e-10
    hf.max_cycle = 300
    hf.kernel()
    if not hf.converged:
        raise RuntimeError(f"RHF did not converge for N2 at R={r_nn:.3f} Angstrom")

    # CASCI active space from a symmetry-selected valence guess in RHF ordering.
    if not cas_guess_idx:
        cas_guess_idx = _select_valence_cas_guess_indices(mol, hf, ncas=ncas)

    casci = mcscf.CASCI(hf, ncas, nelecas)
    casci.canonicalization = False
    mo_cas = casci.sort_mo(cas_guess_idx, base=0)
    casci.kernel(mo_cas)
    if float(casci.e_tot) > float(hf.e_tot) + 1e-7:
        raise RuntimeError(
            f"CASCI energy above HF at R={r_nn:.3f} A: E_CASCI={casci.e_tot:.10f}, E_HF={hf.e_tot:.10f}. "
            f"Selected guess indices: {cas_guess_idx}"
        )

    # Build natural orbitals (NO) from PySCF recanonicalization utilities.
    mo_no, _ci_no, no_occ = addons.cas_natorb(casci, mo_coeff=casci.mo_coeff, ci=casci.ci, sort=False)

    # Explicit orbital energies from CAS effective Fock in the NO basis.
    # These are expectation values <phi_i|F_eff|phi_i> for displayed orbitals.
    f_eff_ao = casci.get_fock(mo_coeff=mo_no, ci=casci.ci)
    mo_energy = np.einsum('pi,pq,qi->i', mo_no, f_eff_ao, mo_no)
    mo_energy = np.asarray(mo_energy, dtype=float)

    # Align active subset consistently across frames.
    ncore = (mol.nelectron - nelecas) // 2
    active_cols = list(range(int(ncore), int(ncore + ncas)))

    # Debug print to inspect selected active orbitals and CAS effective-Fock energies.
    irrep_ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, hf.mo_coeff)
    irrep_names = [symm.irrep_id2name(mol.groupname, int(i)) for i in irrep_ids]
    guess_labels = [irrep_names[i] for i in cas_guess_idx]
    guess_hf_energies = np.array2string(np.asarray(hf.mo_energy)[cas_guess_idx], precision=6, separator=', ')
    cas_active_energies = np.array2string(mo_energy[active_cols], precision=6, separator=', ')
    print(
        f"[N2 CASCI] R={r_nn:.3f} A | guess idx={cas_guess_idx} | "
        f"guess labels={guess_labels} | guess HF eps={guess_hf_energies} | "
        f"active cols={active_cols} | F_eff energies(active)={cas_active_energies}",
        flush=True,
    )

    if ref_active_block is not None and ref_mol is not None:
        mo_no, mo_energy, no_occ = _align_active_no_block(
            mo_no=mo_no,
            mo_energy=mo_energy,
            mo_occ=np.asarray(no_occ, dtype=float),
            active_cols=active_cols,
            mol=mol,
            ref_active_block=ref_active_block,
            ref_mol=ref_mol,
        )

    else:
        no_occ = np.asarray(no_occ, dtype=float)

    coords = mol.atom_coords(unit="Angstrom")
    symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]

    return {
        "geometry": {"symbols": symbols, "coords": np.asarray(coords, dtype=float)},
        "basis": mol,
        "mo_coeff": np.asarray(mo_no, dtype=float),
        "mo_energies": np.asarray(mo_energy, dtype=float),
        "mo_occ": np.asarray(no_occ, dtype=float),
        "metadata": {
            "reaction_coordinate": float(r_nn),
            "pes": {"HF": float(hf.e_tot), "CASCI": float(casci.e_tot)},
            "label": f"R = {r_nn:.2f} A",
        },
    }, active_cols, mol, np.asarray(mo_no[:, active_cols], dtype=float)


def build_trajectory(
    n_frames: int = 11,
    r_min: float = 0.90,
    r_max: float = 3.00,
    cas_guess_idx: Optional[list[int]] = None,
    ncas: int = 6,
    nelecas: int = 6,
) -> tuple[list[dict], list[int]]:
    """Compute trajectory using aligned CAS natural orbitals.

    ``cas_guess_idx`` are optional 0-based RHF MO indices for CAS guess.
    If omitted, pi/pi* guess orbitals are selected automatically by symmetry
    labels at each frame.
    """
    rs = np.linspace(float(r_min), float(r_max), int(n_frames))

    frames: list[dict] = []
    active_cols_ref: Optional[list[int]] = None
    ref_mol = None
    ref_active_block = None

    for r in rs:
        frame, active_cols, mol, active_block = build_frame(
            float(r),
            cas_guess_idx=cas_guess_idx,
            ncas=ncas,
            nelecas=nelecas,
            ref_active_block=ref_active_block,
            ref_mol=ref_mol,
        )
        frames.append(frame)
        if ref_active_block is None:
            active_cols_ref = list(active_cols)
            ref_active_block = np.asarray(active_block, dtype=float)
            ref_mol = mol

    assert active_cols_ref is not None
    return frames, active_cols_ref


def main() -> None:
    cache_dir = Path("./.orbitaldash_cache")
    trajectory, active_idx = build_trajectory(n_frames=21, r_min=0.90, r_max=3.00)

    launch_app(
        trajectory=trajectory,
        active=active_idx,
        mode=["lazy", "precomputed"][1],
        spacing_options=[0.20],
        cache_dir=str(cache_dir),
        auto_run=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
