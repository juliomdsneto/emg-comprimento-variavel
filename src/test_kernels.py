"""Testes de sanidade e validacao cruzada: kernels Numba vs referencia literal."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
from emg import run_type1, run_type2
import reference as ref


def test_type1():
    print("== Type-I: kernel Numba vs referencia literal (N=51, T=1000) ==")
    N, S, p, ngen, T, mmin, mmax = 51, 2, 0.1, 60, 1000, 1, 6
    t0 = time.time()
    out = run_type1(N=N, S=S, p=p, ngen=ngen, T=T, mmin=mmin, mmax=mmax,
                    seed=7, opmode=0, snap_every=10)
    t1 = time.time()
    print(f"  kernel: {t1-t0:6.2f}s  sigma2_final={out['sigma2'][-1]:.4f} "
          f"meanm_final={out['meanm'][-1]:.3f}")

    t0 = time.time()
    sref, mref = ref.run_type1(N=N, S=S, p=p, ngen=ngen, T=T, mmin=mmin,
                               mmax=mmax, seed=7, check=True)
    t1 = time.time()
    print(f"  ref   : {t1-t0:6.2f}s  sigma2_final={sref[-1]:.4f} meanm_final={mref[-1]:.3f} "
          "(assercoes estruturais OK)")
    print(f"  sigma2 medio ger 30-59: kernel {np.mean(out['sigma2'][30:]):.4f} "
          f"vs ref {np.mean(sref[30:]):.4f}")
    print(f"  meanm  medio ger 30-59: kernel {np.mean(out['meanm'][30:]):.3f} "
          f"vs ref {np.mean(mref[30:]):.3f}")


def test_type2():
    print("\n== Type-II: kernel Numba vs referencia literal (N=51, T=1000) ==")
    N, S, p, ngen, T, mmin, mmax = 51, 4, 0.1, 60, 1000, 1, 6
    t0 = time.time()
    out = run_type2(N=N, S=S, p=p, ngen=ngen, T=T, mmin=mmin, mmax=mmax,
                    seed=11, opmode=0, snap_every=10)
    t1 = time.time()
    print(f"  kernel: {t1-t0:6.2f}s  sigma2_final={out['sigma2'][-1]:.4f} "
          f"meanm_final={out['meanm'][-1]:.3f}")
    t0 = time.time()
    sref, mref = ref.run_type2(N=N, S=S, p=p, ngen=ngen, T=T, mmin=mmin,
                               mmax=mmax, seed=11, check=True)
    t1 = time.time()
    print(f"  ref   : {t1-t0:6.2f}s  sigma2_final={sref[-1]:.4f} meanm_final={mref[-1]:.3f} "
          "(assercoes estruturais OK)")
    print(f"  sigma2 medio ger 30-59: kernel {np.mean(out['sigma2'][30:]):.4f} "
          f"vs ref {np.mean(sref[30:]):.4f}")
    print(f"  meanm  medio ger 30-59: kernel {np.mean(out['meanm'][30:]):.3f} "
          f"vs ref {np.mean(mref[30:]):.3f}")


def test_physics():
    print("\n== Checagens fisicas do MG basico (N=101, S=2, 5 sementes) ==")
    for m in [1, 2, 3, 4, 5, 6, 8, 10]:
        vals, hs = [], []
        for s in range(5):
            o = run_type1(N=101, S=2, p=0.0, ngen=1, T=10000, mmin=m, mmax=m,
                          seed=100 + s, opmode=0, warm=1000, hbits=m)
            vals.append(o["sigma2"][0] / 101)
            hs.append(o["H"][0])
        print(f"  m={m:2d}  sigma2/N = {np.mean(vals):.4f} +- {np.std(vals):.4f}"
              f"   H = {np.mean(hs):.4f}")
    print("  esperado: minimo em m~3-5 (sigma2/N ~ 0.02-0.05), ->0.25 p/ m grande,")
    print("            e H>0 na fase de cooperacao (m pequeno).")


if __name__ == "__main__":
    t0 = time.time()
    test_type1()
    test_type2()
    test_physics()
    print(f"\nTotal {time.time()-t0:.1f}s")
