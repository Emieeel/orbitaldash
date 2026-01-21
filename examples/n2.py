"""Example: run HF on N2 with PySCF then launch OrbitalDash.

This is a quick sanity-check example to verify:
- Orbital rendering quality (try different Quality + Smooth sliders)
- Bond multiplicity visualization (N≡N should show as a triple bond)

Requires: pyscf

Run:
  source /home/emielkoridon/pip_environments/dftqml/bin/activate
  python examples/n2.py
"""

try:
    from pyscf import gto, scf  # type: ignore
except ImportError:  # pragma: no cover
    raise SystemExit("Install pyscf to run this example: pip install pyscf")

from orbitaldash import launch_app


# type: ignore[name-defined]
def build_n2(distance_angstrom: float = 1.0977, basis: str = "cc-pvdz") -> "gto.M":
    """Build an N2 molecule at a given bond length.

    Args:
        distance_angstrom: N-N distance (Å). ~1.0977 Å is near equilibrium.
        basis: PySCF basis name.

    Returns:
        PySCF gto.M molecule.
    """
    d = float(distance_angstrom)
    return gto.M(
        atom=f"N 0 0 {-d / 2:.8f}; N 0 0 {+d / 2:.8f}",
        basis=basis,
        unit="Angstrom",
        verbose=0,
    )


def main():
    mol = build_n2()
    mf = scf.RHF(mol).run()

    # Simplest usage: pass the SCF object directly.
    # OrbitalDash will infer geometry, basis (mol), and mo_coeff/mo_energy/mo_occ.
    launch_app(mf, port=8054)


if __name__ == "__main__":
    main()
