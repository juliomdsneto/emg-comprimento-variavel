"""Roda 12 sementes adicionais (508-519) nas series longas e grava *_20.npz.

As series originais usam as sementes 500-507 (8 execucoes). Aqui repetimos as
mesmas configuracoes com mais 12, totalizando 20 execucoes independentes por
ponto, para responder ao pedido de robustez estatistica.
"""

import os
import sys
import numpy as np
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from emg import run_type1, run_type2   # noqa: E402

RES = os.path.join(AQUI, "..", "results")
N0, T0, MMIN, MMAX = 101, 10000, 1, 15
NOVAS = list(range(508, 520))
P = [0.05, 0.10, 0.20]


def trabalho(a):
    kind, par, seed = a
    f = run_type1 if kind == "type1" else run_type2
    return f(seed=seed, **par)


def main():
    with Pool(8) as pool:
        for tag, kind, S in [("type1", "type1", 2), ("type2", "type2", 4)]:
            z = np.load(os.path.join(RES, f"long_{tag}.npz"))
            sig = z["sigma2"]
            mm = z["meanm"]
            snap = z["snapshots"]
            ps = z["ps"]
            novos_sig, novos_mm, novos_snap = [], [], []
            for p in P:
                par = dict(N=N0, S=S, p=float(p), ngen=2000, T=T0, mmin=MMIN,
                           mmax=MMAX, opmode=0, snap_every=250)
                outs = pool.map(trabalho, [(kind, par, s) for s in NOVAS])
                novos_sig.append(np.array([o["sigma2"] for o in outs]) / N0)
                novos_mm.append(np.array([o["meanm"] for o in outs]))
                novos_snap.append(np.array([o["snapshots"] for o in outs]))
                f = novos_sig[-1][:, -100:].mean(axis=1)
                print(f"  {tag} p={p}: sigma2/N final = {f.mean():.5f} "
                      f"+- {f.std(ddof=1)/np.sqrt(len(f)):.5f}  "
                      f"m={novos_mm[-1][:, -100:].mean():.2f}", flush=True)

            sig20 = np.concatenate([sig, np.array(novos_sig)], axis=1)
            mm20 = np.concatenate([mm, np.array(novos_mm)], axis=1)
            snap20 = np.concatenate([snap, np.array(novos_snap)], axis=1)
            np.savez_compressed(
                os.path.join(RES, f"long_{tag}_20.npz"),
                sigma2=sig20, meanm=mm20, snapshots=snap20, ps=ps)
            print(f"  {tag}: gravado com {sig20.shape[1]} execucoes", flush=True)


if __name__ == "__main__":
    print("rodando 12 sementes adicionais (508-519)...", flush=True)
    main()
    print("concluido")
