"""
Suite de experimentos do trabalho (replicacao + extensoes).

Uso:
    python experiments.py rep        # Figs. 1-5 do artigo (replicacao)
    python experiments.py ext        # experimentos estendidos
    python experiments.py all

Saidas em ../results/*.npz e ../results/*.csv
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emg import run_type1, run_type2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

# Parametros do artigo
N0, T0, MMIN, MMAX = 101, 10000, 1, 15
SEEDS8 = list(range(8))


# ------------------------------------------------------------------ runner
def _worker(job):
    kind = job["kind"]
    par = dict(job["params"])
    seed = job["seed"]
    t0 = time.time()
    if kind == "type1":
        o = run_type1(seed=seed, **par)
    elif kind == "type2":
        o = run_type2(seed=seed, **par)
    else:
        raise ValueError(kind)
    o["elapsed"] = time.time() - t0
    o["seed"] = seed
    o["job"] = {k: v for k, v in job.items() if k != "params"}
    return o


def run_jobs(jobs, workers=8, verbose=True):
    t0 = time.time()
    out = []
    with Pool(workers) as pool:
        for i, o in enumerate(pool.imap_unordered(_worker, jobs, chunksize=1)):
            out.append(o)
            if verbose:
                print(f"   [{i+1}/{len(jobs)}] seed={o['seed']} "
                      f"{o['elapsed']:.1f}s", flush=True)
    if verbose:
        print(f"   total {time.time()-t0:.0f}s")
    out.sort(key=lambda o: o["seed"])
    return out


def stack(outs, key):
    return np.array([o[key] for o in outs])


# ------------------------------------------------------- experimentos: rep
def exp_rep(workers=8):
    """Figs. 1-4: series temporais e distribuicoes (parametros do artigo)."""
    P = [0.05, 0.1, 0.2]
    for tag, kind, S, snap in [("type1", "type1", 2, 300),
                               ("type2", "type2", 4, 100)]:
        allsig, allm, allsnap, allH, allgini = [], [], [], [], []
        for p in P:
            par = dict(N=N0, S=S, p=p, ngen=500, T=T0, mmin=MMIN, mmax=MMAX,
                       opmode=0, snap_every=snap)
            jobs = [dict(kind=kind, params=par, seed=s, p=p) for s in SEEDS8]
            outs = run_jobs(jobs, workers)
            allsig.append(stack(outs, "sigma2") / N0)
            allm.append(stack(outs, "meanm"))
            allsnap.append(stack(outs, "snapshots"))
            allgini.append(stack(outs, "gini"))
            print(f"  {tag} p={p}: sigma2/N final (ult. 50 ger) = "
                  f"{np.mean(allsig[-1][:, -50:]):.5f} +- "
                  f"{np.std(np.mean(allsig[-1][:, -50:], axis=1), ddof=1)/np.sqrt(8):.5f}")
        np.savez_compressed(
            os.path.join(OUT, f"rep_{tag}.npz"),
            sigma2=np.array(allsig), meanm=np.array(allm),
            snapshots=np.array(allsnap), gini=np.array(allgini),
            ps=np.array(P))

    # MG basico: sigma2/N vs m (curva de referencia da Fig. 5)
    print("  MG basico: varredura em m")
    rows = []
    for m in range(1, 13):
        par = dict(N=N0, S=2, p=0.0, ngen=1, T=T0, mmin=m, mmax=m,
                   warm=2000, hbits=m)
        jobs = [dict(kind="type1", params=par, seed=1000 + s) for s in range(20)]
        outs = run_jobs(jobs, workers, verbose=False)
        s = stack(outs, "sigma2") / N0
        h = stack(outs, "H")
        rows.append((m, s.mean(), s.std(ddof=1) / np.sqrt(len(s)), h.mean()))
        print(f"    m={m:2d} sigma2/N={rows[-1][1]:.4f} +- {rows[-1][2]:.4f} "
              f"H={rows[-1][3]:.4f}")
    np.savetxt(os.path.join(OUT, "mg_curve.csv"), np.array(rows), fmt="%.6f",
               header="m,sigma2_over_N,sem,H", comments="")


# ------------------------------------------ series longas (patamares)
def exp_long(workers=8):
    """Series longas (2000 geracoes) para localizar os patamares assintoticos."""
    P = [0.05, 0.1, 0.2]
    for tag, kind, S, snap in [("type1", "type1", 2, 250),
                               ("type2", "type2", 4, 250)]:
        sigs, ms, snaps = [], [], []
        for p in P:
            par = dict(N=N0, S=S, p=p, ngen=2000, T=T0, mmin=MMIN, mmax=MMAX,
                       opmode=0, snap_every=snap)
            jobs = [dict(kind=kind, params=par, seed=500 + s) for s in SEEDS8]
            outs = run_jobs(jobs, workers)
            sigs.append(stack(outs, "sigma2") / N0)
            ms.append(stack(outs, "meanm"))
            snaps.append(stack(outs, "snapshots"))
            f = np.mean(sigs[-1][:, -100:], axis=1)
            print(f"  {tag} p={p}: sigma2/N final (ult. 100 ger) = {f.mean():.5f}"
                  f" +- {f.std(ddof=1)/np.sqrt(8):.5f}"
                  f"  m={stack(outs, 'meanm')[:, -100:].mean():.2f}")
        np.savez_compressed(os.path.join(OUT, f"long_{tag}.npz"),
                            sigma2=np.array(sigs), meanm=np.array(ms),
                            snapshots=np.array(snaps), ps=np.array(P))


# ------------------------------------------ experimentos estendidos (ext)
def exp_ext(workers=8):
    rows = []

    # ---- E1: varredura em p (trade-off velocidade x patamar final)
    print("== E1: varredura em p ==")
    P = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for tag, kind, S in [("type1", "type1", 2), ("type2", "type2", 4)]:
        for p in P:
            par = dict(N=N0, S=S, p=p, ngen=600, T=T0, mmin=MMIN, mmax=MMAX,
                       opmode=0, snap_every=0)
            jobs = [dict(kind=kind, params=par, seed=s) for s in SEEDS8]
            outs = run_jobs(jobs, workers, verbose=False)
            sig = stack(outs, "sigma2") / N0
            mm = stack(outs, "meanm")
            fin = np.mean(sig[:, -50:], axis=1)
            # tempo de convergencia: 1a geracao em que a media fica a <10% do
            # valor final (usando a curva media)
            curve = sig.mean(axis=0)
            target = 1.10 * curve[-50:].mean()
            conv = int(np.argmax(curve < target)) if np.any(curve < target) else -1
            rows.append(dict(exp="p_sweep", model=tag, p=p,
                             sig_final=fin.mean(),
                             sig_sem=fin.std(ddof=1) / np.sqrt(len(fin)),
                             meanm=mm[:, -50:].mean(),
                             conv_gen=conv,
                             gini=stack(outs, "gini")[:, -50:].mean(),
                             meanA=stack(outs, "meanA")[:, -50:].mean()))
            print(f"  {tag} p={p:4.2f} sig={rows[-1]['sig_final']:.5f}"
                  f" m={rows[-1]['meanm']:.2f} conv={conv}")

    # ---- E2: sensibilidade a m_max
    print("== E2: varredura em m_max ==")
    for tag, kind, S in [("type1", "type1", 2), ("type2", "type2", 4)]:
        for mmax in [3, 5, 8, 10, 15]:
            par = dict(N=N0, S=S, p=0.05, ngen=400, T=T0, mmin=MMIN,
                       mmax=mmax, opmode=0, snap_every=200)
            jobs = [dict(kind=kind, params=par, seed=s) for s in SEEDS8]
            outs = run_jobs(jobs, workers, verbose=False)
            sig = stack(outs, "sigma2") / N0
            fin = np.mean(sig[:, -50:], axis=1)
            rows.append(dict(exp="mmax_sweep", model=tag, p=0.05, mmax=mmax,
                             sig_final=fin.mean(),
                             sig_sem=fin.std(ddof=1) / np.sqrt(len(fin)),
                             meanm=stack(outs, "meanm")[:, -50:].mean(),
                             conv_gen=-1, gini=stack(outs, "gini")[:, -50:].mean(),
                             meanA=stack(outs, "meanA")[:, -50:].mean()))
            np.savez_compressed(os.path.join(OUT, f"mmax_{tag}_{mmax}.npz"),
                                snapshots=stack(outs, "snapshots"),
                                sigma2=sig, meanm=stack(outs, "meanm"))
            print(f"  {tag} mmax={mmax:2d} sig={rows[-1]['sig_final']:.5f}"
                  f" m={rows[-1]['meanm']:.2f}")

    # ---- E3: sensibilidade a N (escala finita)
    print("== E3: varredura em N ==")
    for tag, kind, S in [("type1", "type1", 2), ("type2", "type2", 4)]:
        for N in [51, 101, 201, 501]:
            par = dict(N=N, S=S, p=0.05, ngen=400, T=T0, mmin=MMIN,
                       mmax=MMAX, opmode=0, snap_every=0)
            jobs = [dict(kind=kind, params=par, seed=s) for s in SEEDS8]
            outs = run_jobs(jobs, workers, verbose=False)
            sig = stack(outs, "sigma2") / N
            fin = np.mean(sig[:, -50:], axis=1)
            curve = sig.mean(axis=0)
            target = 1.10 * curve[-50:].mean()
            conv = int(np.argmax(curve < target)) if np.any(curve < target) else -1
            rows.append(dict(exp="N_sweep", model=tag, N=N, p=0.05,
                             sig_final=fin.mean(),
                             sig_sem=fin.std(ddof=1) / np.sqrt(len(fin)),
                             meanm=stack(outs, "meanm")[:, -50:].mean(),
                             conv_gen=conv, gini=stack(outs, "gini")[:, -50:].mean(),
                             meanA=stack(outs, "meanA")[:, -50:].mean()))
            print(f"  {tag} N={N:3d} sig={rows[-1]['sig_final']:.5f}"
                  f" m={rows[-1]['meanm']:.2f} conv={conv}")

    # ---- E4: controles estaticos (heterogeneidade sem evolucao)
    print("== E4: controles estaticos ==")
    for tag, kind, S, static in [("type1_static", "type1", 2, 0),
                                 ("type2_static", "type2", 4, 1)]:
        for p in [0.0]:
            par = dict(N=N0, S=S, p=p, ngen=20, T=T0, mmin=MMIN, mmax=MMAX,
                       opmode=0, snap_every=0)
            if kind == "type2":
                par["static"] = 1
            jobs = [dict(kind=kind, params=par, seed=s) for s in range(12)]
            outs = run_jobs(jobs, workers, verbose=False)
            sig = stack(outs, "sigma2") / N0
            rows.append(dict(exp="static", model=tag, p=0.0,
                             sig_final=sig.mean(), sig_sem=sig.std(ddof=1) / np.sqrt(len(sig)),
                             meanm=stack(outs, "meanm").mean(), conv_gen=-1,
                             gini=stack(outs, "gini").mean(),
                             meanA=stack(outs, "meanA").mean()))
            print(f"  {tag}: sig={rows[-1]['sig_final']:.5f}")

    # ---- E5: ablacao de operadores (Type-I e Type-II)
    print("== E5: ablacao de operadores ==")
    OPS = {0: "cut+splice", 1: "cut", 2: "splice", 3: "redesenho",
           4: "crossover", 5: "mutacao"}
    for tag, kind, S in [("type1", "type1", 2), ("type2", "type2", 4)]:
        for op, name in OPS.items():
            par = dict(N=N0, S=S, p=0.05, ngen=400, T=T0, mmin=MMIN,
                       mmax=MMAX, opmode=op, snap_every=200)
            jobs = [dict(kind=kind, params=par, seed=s) for s in SEEDS8]
            outs = run_jobs(jobs, workers, verbose=False)
            sig = stack(outs, "sigma2") / N0
            fin = np.mean(sig[:, -50:], axis=1)
            rows.append(dict(exp="op_ablation", model=tag, op=name, p=0.05,
                             sig_final=fin.mean(),
                             sig_sem=fin.std(ddof=1) / np.sqrt(len(fin)),
                             meanm=stack(outs, "meanm")[:, -50:].mean(),
                             conv_gen=-1, gini=stack(outs, "gini")[:, -50:].mean(),
                             meanA=stack(outs, "meanA")[:, -50:].mean()))
            np.savez_compressed(os.path.join(OUT, f"op_{tag}_{op}.npz"),
                                snapshots=stack(outs, "snapshots"),
                                sigma2=sig, meanm=stack(outs, "meanm"))
            print(f"  {tag} op={name:11s} sig={rows[-1]['sig_final']:.5f}"
                  f" m={rows[-1]['meanm']:.2f}")

    # ---- E6: ablacao do reset de pontuacoes e da regra de atualizacao
    print("== E6: ablacao reset/update ==")
    for tag, kind, S in [("type1", "type1", 2), ("type2", "type2", 4)]:
        for rs in [1, 0]:
            for ua in [1, 0]:
                par = dict(N=N0, S=S, p=0.05, ngen=400, T=T0, mmin=MMIN,
                           mmax=MMAX, opmode=0, snap_every=0, reset_scores=rs,
                           update_all=ua)
                jobs = [dict(kind=kind, params=par, seed=s) for s in SEEDS8]
                outs = run_jobs(jobs, workers, verbose=False)
                sig = stack(outs, "sigma2") / N0
                fin = np.mean(sig[:, -50:], axis=1)
                rows.append(dict(exp="reset_ablation", model=tag,
                                 op=f"reset={rs},update_all={ua}",
                                 p=0.05, sig_final=fin.mean(),
                                 sig_sem=fin.std(ddof=1) / np.sqrt(len(fin)),
                                 meanm=stack(outs, "meanm")[:, -50:].mean(),
                                 conv_gen=-1,
                                 gini=stack(outs, "gini")[:, -50:].mean(),
                                 meanA=stack(outs, "meanA")[:, -50:].mean()))
                print(f"  {tag} reset={rs} update_all={ua} "
                      f"sig={rows[-1]['sig_final']:.5f} m={rows[-1]['meanm']:.2f}")

    # ---- E7: riqueza media por memoria (por que m pequeno?)
    print("== E7: riqueza por m ==")
    for tag, kind, S in [("type1", "type1", 2)]:
        par = dict(N=N0, S=S, p=0.05, ngen=400, T=T0, mmin=MMIN, mmax=MMAX,
                   opmode=0, snap_every=0, track_wealth=1)
        jobs = [dict(kind=kind, params=par, seed=s) for s in SEEDS8]
        outs = run_jobs(jobs, workers, verbose=False)
        wsum = sum(o["wealth_by_m_sum"] for o in outs)
        wcnt = sum(o["wealth_by_m_cnt"] for o in outs)
        np.savetxt(os.path.join(OUT, "wealth_by_m.csv"),
                   np.column_stack([np.arange(len(wsum)), wsum, wcnt,
                                    np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), np.nan)]),
                   fmt="%.6f", header="m,wealth_sum,count,mean_wealth", comments="")
        print("   ok")

    # ---- E8: diagnostico Type-II variando S (assimetria do splice)
    print("== E8: Type-II variando S ==")
    for S in [2, 3, 4, 6]:
        par = dict(N=N0, S=S, p=0.05, ngen=800, T=T0, mmin=MMIN, mmax=MMAX,
                   opmode=0, snap_every=400)
        jobs = [dict(kind="type2", params=par, seed=s) for s in SEEDS8]
        outs = run_jobs(jobs, workers, verbose=False)
        sig = stack(outs, "sigma2") / N0
        fin = np.mean(sig[:, -50:], axis=1)
        rows.append(dict(exp="S_sweep", model="type2", p=0.05, mmax=MMAX,
                         N=N0, op=f"S={S}", sig_final=fin.mean(),
                         sig_sem=fin.std(ddof=1) / np.sqrt(len(fin)),
                         meanm=stack(outs, "meanm")[:, -50:].mean(),
                         conv_gen=-1, gini=stack(outs, "gini")[:, -50:].mean(),
                         meanA=stack(outs, "meanA")[:, -50:].mean()))
        print(f"  type2 S={S} sig={rows[-1]['sig_final']:.5f} "
              f"m={rows[-1]['meanm']:.2f}")

    df = []
    cols = ["exp", "model", "p", "mmax", "N", "op", "sig_final", "sig_sem",
            "meanm", "conv_gen", "gini", "meanA"]
    for r in rows:
        df.append([r.get(c, "") for c in cols])
    with open(os.path.join(OUT, "summary.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for line in df:
            f.write(",".join(str(x) for x in line) + "\n")
    print("resumo salvo em results/summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rep", "ext", "long", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.what in ("rep", "all"):
        exp_rep(a.workers)
    if a.what in ("ext", "all"):
        exp_ext(a.workers)
    if a.what in ("long", "all"):
        exp_long(a.workers)
