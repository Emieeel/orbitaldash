"""C2 CASCI example: show two launch modes with explicit active selection.

We use a non-Fermi active selection via sort_mo([5,6,8,9]) and demonstrate:

1) Reordered MO launch: pass the reordered MO matrix to the viewer so the
    active orbitals appear contiguous and labeled by index in that reordered basis.

2) HF-order launch: pass only the CASCI object and the active orbital indices
    (converted to 0-based for the viewer). This shows the same physical orbitals
    but in the original RHF orbital order. The viewer highlights the specified
    active indices relative to that order.

Important indexing notes:
- PySCF sort_mo expects 1-based orbital indices.
- OrbitalDash expects 0-based indices for the optional "active" argument.

Requires: pyscf
"""
from __future__ import annotations

import numpy as np
try:
    from pyscf import gto, scf, mcscf  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install pyscf to run this example: pip install pyscf")

from orbitaldash import launch_app


def run_reordered_mo():
    mol = gto.M(
        verbose=0,
        atom='H 0 0 0; F 0 0 1.2',
        basis='ccpvdz',
        symmetry=0,
    )

    myhf = scf.RHF(mol)
    myhf.kernel()

    # 4 orbitals, 4 electrons (CASCI)
    mycas = mcscf.CASCI(myhf, 4, 4)
    mycas.canonicalization = False
    # sort_mo takes 1-based indices; here we pick a set not centered on Fermi.
    act_idx = [2, 4, 6, 8]
    mo = mycas.sort_mo(act_idx, base=0)
    mycas.kernel(mo)  # type: ignore[attr-defined]

    # Launch with the reordered MO matrix. The viewer will render in the mo order provided.
    return launch_app(mycas, mo_coeff=mo, auto_run=False, port=8055), mo


def run_hf_order_with_active():
    mol = gto.M(
        verbose=0,
        atom='H 0 0 0; F 0 0 1.2',
        basis='ccpvdz',
        symmetry=0,
    )

    myhf = scf.RHF(mol)
    myhf.kernel()

    mycas = mcscf.CASCI(myhf, 4, 4)
    mycas.canonicalization = False
    act_idx = [2, 4, 6, 8]
    mo = mycas.sort_mo(act_idx, base=0)  # type: ignore[attr-defined]
    mycas.kernel(mo)  # type: ignore[attr-defined]

    # Pass only the active indices (converted to 0-based) so the viewer uses
    # the default CASCI mycas.mo_coeff (which matches HF order for CASCI) and
    # highlights the requested active orbitals in that order.

    return launch_app(mycas, active=act_idx, auto_run=False, port=8056), myhf.mo_coeff


# if __name__ == "__main__":
# Pick one to run at a time. The first shows reordered basis; the second preserves HF order.
app1, mo1 = run_reordered_mo()
app2, mo2 = run_hf_order_with_active()

# Show which mos are different: compute a column permutation mapping (mo1 -> mo2)


def _normalize_cols(X):
    norms = np.linalg.norm(X, axis=0)
    return X / norms[np.newaxis, :]


A = _normalize_cols(mo1)
B = _normalize_cols(mo2)
n = min(A.shape[1], B.shape[1])

# overlap matrix (signed); use absolute values to match ignoring global sign
overlap = A.T @ B  # shape (n, m)
# assume same number of columns; slice defensively
abs_overlap = np.abs(overlap[:, :n])

# try optimal assignment (maximizing absolute overlap); fallback to greedy if scipy not available
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    # maximize by minimizing negative
    row_ind, col_ind = linear_sum_assignment(-abs_overlap)
    perm = np.full(n, -1, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
except Exception:
    perm = np.full(n, -1, dtype=int)
    used = set()
    for i in range(n):
        j = int(np.argmax(abs_overlap[i]))
        # avoid duplicates
        while j in used:
            abs_overlap[i, j] = -1.0
            j = int(np.argmax(abs_overlap[i]))
        perm[i] = j
        used.add(j)

# print a fancy mapping with correlation and sign information
print("Column permutation mapping (mo1_index → mo2_index) with |overlap| and relative sign:")
for i in range(n):
    j = int(perm[i])
    corr = abs(overlap[i, j])
    sign = "-" if overlap[i, j] < 0 else "+"
    print(f"  mo1[{i}] \u21D2 mo2[{j}]   |overlap|={corr:0.6f}   sign={sign}")

# summary permutation as a python list (perm[i] = index in mo2 corresponding to mo1[i])
print("\nPermutation (mo1 -> mo2):", perm.tolist())
# print("Different MOs:", diff)

app1.run(host='localhost', port=8055)
app2.run(host='localhost', port=8056)
# run_hf_order_with_active()
