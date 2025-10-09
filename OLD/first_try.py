from pyscf import gto, scf
from pyscf.tools import molden

mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='sto-3g')
mf = scf.RHF(mol).run()
molden.from_scf(mf, 'hf.molden')
