"""Validacao estatistica: kernel Numba vs implementacao de referencia literal.

Roda varias sementes em ambos e compara media +- erro padrao do sigma2/N e a
distribuicao de m.  Como os RNGs sao diferentes, espera-se concordancia dentro
de ~2 erros padrao (o modelo e o mesmo; so a sequencia aleatoria muda).
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from emg import run_type1, run_type2
import reference as ref

N, T, NGEN, P = 51, 2000, 80, 0.1
MMIN, MMAX = 1, 6
SEEDS = [1, 2, 3, 4]
WIN = slice(NGEN - 20, NGEN)


def stats(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def compare(name, kern, refv, unit="sigma2/N"):
    mk, sk = stats(kern)
    mr, sr = stats(refv)
    d = abs(mk - mr)
    sig = np.sqrt(sk ** 2 + sr ** 2)
    print(f"  {name:14s} kernel {mk:8.4f} +- {sk:6.4f} | ref {mr:8.4f} +- {sr:6.4f}"
          f" | dif {d:6.4f} ({d/sig:.1f} sigma)")


def main():
    t0 = time.time()
    print(f"Config: N={N} T={T} ngen={NGEN} p={P} m in [{MMIN},{MMAX}], "
          f"media sobre geracoes {WIN.start}-{WIN.stop-1}\n")

    for tag, fn, reffn in [("Type-I", run_type1, ref.run_type1),
                           ("Type-II", run_type2, ref.run_type2)]:
        S = 2 if tag == "Type-I" else 4
        ks, rm, km, rmm = [], [], [], []
        for s in SEEDS:
            o = fn(N=N, S=S, p=P, ngen=NGEN, T=T, mmin=MMIN, mmax=MMAX, seed=s,
                   opmode=0, snap_every=0)
            ks.append(np.mean(o["sigma2"][WIN]) / N)
            km.append(np.mean(o["meanm"][WIN]))
            a, b = reffn(N=N, S=S, p=P, ngen=NGEN, T=T, mmin=MMIN,
                         mmax=MMAX, seed=100 + s, check=True)
            rm.append(np.mean(a[WIN]) / N)
            rmm.append(np.mean(b[WIN]))
        print(f"== {tag} ==")
        compare("sigma2/N", ks, rm)
        compare("m medio", km, rmm)
        print()

    print(f"Tempo: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
