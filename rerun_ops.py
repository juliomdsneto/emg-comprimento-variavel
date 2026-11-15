"""Re-executa a ablação de operadores (E5) nos dois modelos, apos corrigir a
implementacao de crossover e mutacao no Tipo-II.

Salva results/op_ablation.csv com colunas: model,op,sig_final,sig_sem,meanm,gini
"""

import os
import sys
import csv
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from emg import run_type1, run_type2

OUT = os.path.join(HERE, "..", "results")

OPS = {0: "cut+splice", 1: "cut", 2: "splice", 3: "redesenho", 4: "crossover",
       5: "mutacao"}
SEEDS = list(range(8))
BASE = dict(N=101, p=0.05, ngen=400, T=10000, mmin=1, mmax=15, snap_every=0)


def job(args):
    kind, op, seed = args
    par = dict(BASE)
    par["S"] = 2 if kind == "type1" else 4
    if kind == "type1":
        o = run_type1(seed=seed, opmode=op, **par)
    else:
        o = run_type2(seed=seed, opmode=op, **par)
    return kind, op, seed, o["sigma2"], o["meanm"], o["gini"]


def main():
    args = [(k, op, s) for k in ("type1", "type2") for op in OPS for s in SEEDS]
    rows = []
    with Pool(10) as pool:
        res = pool.map(job, args)
    for kind, op, seed, sig, mm, gi in res:
        sig = np.asarray(sig) / 101.0
        rows.append(dict(model=kind, op=OPS[op], opcode=op,
                         sig_final=float(np.mean(sig[-50:])),
                         meanm=float(np.mean(np.asarray(mm)[-50:])),
                         gini=float(np.mean(np.asarray(gi)[-50:]))))

    # agrega por (model, op)
    out = []
    for kind in ("type1", "type2"):
        for op in OPS:
            r = [x for x in rows if x["model"] == kind and x["opcode"] == op]
            s = np.array([x["sig_final"] for x in r])
            out.append(dict(model=kind, op=OPS[op],
                            sig_final=s.mean(),
                            sig_sem=s.std(ddof=1) / np.sqrt(len(s)),
                            meanm=np.mean([x["meanm"] for x in r]),
                            gini=np.mean([x["gini"] for x in r])))
            print(f"{kind:6s} {OPS[op]:11s} sig={s.mean():.5f} +- "
                  f"{s.std(ddof=1)/np.sqrt(len(s)):.5f}  "
                  f"m={np.mean([x['meanm'] for x in r]):.2f}")

    path = os.path.join(OUT, "op_ablation.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "op", "sig_final", "sig_sem",
                                          "meanm", "gini"])
        w.writeheader()
        w.writerows(out)
    print("salvo:", path)


if __name__ == "__main__":
    main()
