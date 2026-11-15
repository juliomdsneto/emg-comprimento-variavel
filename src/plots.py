"""
Geracao das figuras do artigo (PDF vetorial + PNG para pre-visualizacao).

Le os resultados de ../results/ e grava em ../artigo/figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "artigo", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.linewidth": 0.9,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 8.8,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.8",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.2,
    "errorbar.capsize": 2.4,
})

COL = {0.05: "#12457a", 0.1: "#b8451c", 0.2: "#1f7a45"}
LBL = {0.05: r"$p=0{,}05$", 0.1: r"$p=0{,}10$", 0.2: r"$p=0{,}20$"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=320)
    plt.close(fig)
    print("  fig:", name)


# ------------------------------------------------------------------ Fig. 1/3
def fig_series2(fname="fig_series"):
    """sigma^2/N vs geracao (escala log) para os dois modelos, num so painel duplo.

    A escala logaritmica no eixo x permite ver, no mesmo grafico, a janela inicial
    (onde o artigo faz suas afirmacoes) e o patamar assintotico (2000 geracoes).
    """
    zr1 = np.load(os.path.join(RES, "rep_type1.npz"))
    zr2 = np.load(os.path.join(RES, "rep_type2.npz"))
    zl1 = np.load(os.path.join(RES, "long_type1.npz"))
    zl2 = np.load(os.path.join(RES, "long_type2.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45),
                             gridspec_kw=dict(wspace=0.24))
    for ax, zr, zl, ttl in [(axes[0], zr1, zl1, "(a) Tipo-I"),
                            (axes[1], zr2, zl2, "(b) Tipo-II")]:
        ps = zr["ps"]
        xl = np.arange(zl["sigma2"].shape[2]) + 1
        for i, p in enumerate(ps):
            yl = zl["sigma2"][i]
            m = yl.mean(axis=0)
            s = yl.std(axis=0, ddof=1) / np.sqrt(yl.shape[0])
            ax.plot(xl, m, color=COL[p], label=LBL[p], lw=1.7)
            ax.fill_between(xl, m - s, m + s, color=COL[p], alpha=0.22, lw=0)
        ax.set_xscale("log")
        ax.set_xlim(1, 2000)
        ax.set_xlabel("geração (escala log)")
        ax.set_ylabel(r"$\sigma^2/N$")
        ax.set_title(ttl)
        ax.axvline(500, color="0.55", ls=":", lw=1.0)
        ax.annotate("g = 500", xy=(500, 0.03), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                    color="0.35", va="bottom")
    axes[0].legend(loc="upper right", fontsize=9)
    save(fig, fname)


def fig_series(tag, ngen_zoom, fname, ylim=None):
    """sigma^2/N vs geracao: janela do artigo (esq.) e serie longa (dir.)."""
    zr = np.load(os.path.join(RES, f"rep_{tag}.npz"))
    zl = np.load(os.path.join(RES, f"long_{tag}.npz"))
    ps = zr["ps"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.45),
                             gridspec_kw=dict(wspace=0.28))
    xr = np.arange(zr["sigma2"].shape[2]) + 1
    xl = np.arange(zl["sigma2"].shape[2]) + 1
    for i, p in enumerate(ps):
        for ax, z, x, lim in [(axes[0], zr, xr, ngen_zoom),
                              (axes[1], zl, xl, None)]:
            y = z["sigma2"][i]
            m = y.mean(axis=0)
            s = y.std(axis=0, ddof=1) / np.sqrt(y.shape[0])
            sel = slice(0, lim) if lim else slice(None)
            ax.plot(x[sel], m[sel], color=COL[p], label=LBL[p])
            ax.fill_between(x[sel], (m - s)[sel], (m + s)[sel], color=COL[p],
                            alpha=0.22, lw=0)
    axes[0].set_xlabel("geração")
    axes[1].set_xlabel("geração")
    axes[0].set_ylabel(r"$\sigma^2/N$")
    axes[1].set_ylabel(r"$\sigma^2/N$")
    axes[0].set_title("(a) janela do artigo", fontsize=8.5)
    axes[1].set_title("(b) série longa", fontsize=8.5)
    if ylim:
        for ax in axes:
            ax.set_ylim(*ylim)
    axes[0].legend(loc="upper right")
    save(fig, fname)


# ------------------------------------------------------------------ Fig. 2/4
def fig_mdist(tag, fname, gc=8.0, title="", gens_show=None):
    """Distribuicao de m em diferentes geracoes (fotos)."""
    z = np.load(os.path.join(RES, f"long_{tag}.npz"))
    snap = z["snapshots"]           # (p, run, foto, m+1)
    ps = z["ps"]
    i = 0 if ps[0] == 0.05 else int(np.argmin(np.abs(ps - 0.05)))
    d = snap[i].mean(axis=0)        # (foto, m+1)
    gens = [int(round(d[k, -1])) for k in range(d.shape[0])]
    mm = np.arange(d.shape[1] - 1)
    valid = [k for k in range(len(gens))
             if gens[k] <= gc and d[k, :-1].sum() > 0.5]
    if gens_show is None:
        gens_show = [0, 500, 1000, 1500, 1750]
    keep = [k for k in valid if any(abs(gens[k] - g) <= 130 for g in gens_show)]
    if len(keep) < 2:
        keep = valid
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(3.2, 2.35))
    for j, k in enumerate(keep):
        c = cmap(0.08 + 0.84 * j / max(len(keep) - 1, 1))
        ax.plot(mm, d[k, :-1], marker="o", ms=3.4, color=c, lw=1.5,
                label=f"$g={gens[k]}$")
    ax.set_xlabel(r"memória $m$")
    ax.set_ylabel("nº de agentes/estratégias")
    ax.set_xticks(mm)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.4, mm[-1] + 0.4)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=1, loc="upper right", fontsize=8.6, handlelength=1.4,
              borderpad=0.35, labelspacing=0.25)
    if title:
        ax.set_title(title)
    save(fig, fname)


# ------------------------------------------------------------------- Fig. 5
def load_mg():
    return np.loadtxt(os.path.join(RES, "mg_curve.csv"), skiprows=1)


def fig_mg_so(fname="fig_mg_so"):
    """Apenas a curva do MG basico (sem as linhas dos modelos evolucionarios).

    Usada no slide de contexto ("existe uma memoria otima"), onde o assunto
    ainda e' so' o MG basico.
    """
    mg = load_mg()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.errorbar(mg[:, 0], mg[:, 1], yerr=mg[:, 2], fmt="o-", ms=4.6, lw=1.7,
                color="#3d3d3d", capsize=2.6, elinewidth=1.0)
    k = int(np.argmin(mg[:, 1]))
    ax.annotate(f"mínimo em $m={int(mg[k,0])}$\n$\\sigma^2/N={mg[k,1]:.3f}$"
                .replace(".", ","),
                xy=(mg[k, 0], mg[k, 1]), xytext=(mg[k, 0] + 2.0, 0.031),
                fontsize=8.5, color="#3d3d3d",
                arrowprops=dict(arrowstyle="-", lw=0.8, color="0.5"))
    ax.axhline(0.25, color="0.65", lw=1.0, ls=":")
    ax.text(12.6, 0.275, "escolha aleatória", fontsize=8.5, color="0.40",
            ha="right", va="bottom")
    ax.set_xlabel(r"memória $m$")
    ax.set_ylabel(r"$\sigma^2/N$")
    ax.set_xlim(0.5, 13)
    ax.set_ylim(0.02, 4.0)
    ax.set_yscale("log")
    save(fig, fname)


def fig_comparacao():
    mg = load_mg()
    zl1 = np.load(os.path.join(RES, "long_type1.npz"))
    zl2 = np.load(os.path.join(RES, "long_type2.npz"))
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.errorbar(mg[:, 0], mg[:, 1], yerr=mg[:, 2], fmt="o-", ms=4.0, lw=1.4,
                color="#3d3d3d", capsize=2.2, elinewidth=0.9,
                label="MG básico (variação de $m$)")
    for z, lab, c, ls in [(zl1, "EMG Tipo-I", COL[0.05], "--"),
                          (zl2, "EMG Tipo-II", COL[0.2], "-.")]:
        i = 0 if z["ps"][0] == 0.05 else int(np.argmin(np.abs(z["ps"] - 0.05)))
        v = z["sigma2"][i][:, -100:].mean()
        ax.axhline(v, color=c, ls=ls, lw=1.8,
                   label=lab + r" (nosso, $p=0{,}05$)")
    # patamares da Fig. 5 do artigo (valores lidos do grafico, que nao os traz
    # em tabela). Ficam como uma unica entrada de legenda para nao poluir.
    for k, v in enumerate((0.016, 0.008)):
        ax.axhline(v, color="0.45", ls=(0, (1, 1.6)), lw=1.2,
                   label="artigo, Fig. 5" if k == 0 else None)
    ax.set_xlabel(r"memória $m$ (MG) / patamar evolucionário")
    ax.set_ylabel(r"$\sigma^2/N$")
    ax.set_xlim(0.5, 13)
    ax.set_ylim(0.006, 4.0)
    ax.set_yscale("log")
    ax.axhline(0.25, color="0.65", lw=0.9, ls=":")
    ax.text(12.6, 0.275, "aleatório", fontsize=8.5, color="0.40", ha="right",
            va="bottom")
    ax.legend(loc="upper right", fontsize=8.2, handlelength=1.6,
              borderpad=0.35, labelspacing=0.22)
    save(fig, "fig_comparacao")


# ------------------------------------------------------------------- Fig. 6
def fig_contraste():
    """Barras: MG homogeneo otimo, heterogeneo estatico, EMG evoluido."""
    mg = load_mg()
    zr1 = np.load(os.path.join(RES, "rep_type1.npz"))
    zr2 = np.load(os.path.join(RES, "rep_type2.npz"))
    zl1 = np.load(os.path.join(RES, "long_type1.npz"))
    zl2 = np.load(os.path.join(RES, "long_type2.npz"))
    opt = mg[:, 1].min()
    i0 = int(np.argmin(mg[:, 1]))
    labels = [f"MG\n$m={int(mg[i0,0])}$",
              "estático\nheterog.",
              "Tipo-I\n500 g", "Tipo-I\n2000 g",
              "Tipo-II\n500 g", "Tipo-II\n2000 g"]
    vals, errs = [opt, zr1["sigma2"][0][:, 0].mean()], [0.0, zr1["sigma2"][0][:, 0].std(ddof=1) / np.sqrt(8)]
    for z in (zr1, zr2, zl1, zl2):
        v = z["sigma2"][0][:, -100:]
        vals.append(np.mean(v.mean(axis=1)))
        errs.append(np.std(np.mean(v, axis=1), ddof=1) / np.sqrt(8))
    fig, ax = plt.subplots(figsize=(4.4, 2.45))
    cols = ["#444444", "#999999", COL[0.05], COL[0.05], COL[0.2], COL[0.2]]
    ax.bar(range(len(vals)), vals, yerr=errs, color=cols, width=0.7,
           capsize=2, error_kw=dict(lw=0.7))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel(r"$\sigma^2/N$ (patamar)")
    ax.set_ylim(0, max(vals) * 1.25)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.3f}", ha="center", fontsize=6.2)
    save(fig, "fig_contraste")


# ------------------------------------------------------------------- Fig. 7
def fig_p_sweep():
    import csv
    rows = list(csv.DictReader(open(os.path.join(RES, "summary.csv"),
                                    encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35),
                             gridspec_kw=dict(wspace=0.3))
    for tag, c, lab in [("type1", COL[0.05], "Tipo-I"),
                        ("type2", COL[0.2], "Tipo-II")]:
        r = [x for x in rows if x["exp"] == "p_sweep" and x["model"] == tag]
        r.sort(key=lambda x: float(x["p"]))
        p = np.array([float(x["p"]) for x in r])
        s = np.array([float(x["sig_final"]) for x in r])
        e = np.array([float(x["sig_sem"]) for x in r])
        cv = np.array([float(x["conv_gen"]) for x in r])
        axes[0].errorbar(p, s, yerr=e, marker="o", ms=3, capsize=2,
                         elinewidth=0.7, color=c, label=lab)
        axes[1].plot(p, cv, marker="s", ms=3, color=c, label=lab)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel(r"fração $p$ de agentes atualizados")
    axes[1].set_xlabel(r"fração $p$")
    axes[0].set_ylabel(r"$\sigma^2/N$ (patamar)")
    axes[1].set_ylabel("gerações até 90% da queda")
    for ax in axes:
        ax.legend()
    save(fig, "fig_p_sweep")


# ------------------------------------------------------------------- Fig. 8
def fig_mmax_N():
    import csv
    rows = list(csv.DictReader(open(os.path.join(RES, "summary.csv"),
                                    encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35),
                             gridspec_kw=dict(wspace=0.3))
    for exp, xkey, ax, xlab in [("mmax_sweep", "mmax", axes[0], r"$m_{\max}$"),
                                ("N_sweep", "N", axes[1], r"$N$")]:
        for tag, c, lab in [("type1", COL[0.05], "Tipo-I"),
                            ("type2", COL[0.2], "Tipo-II")]:
            r = [x for x in rows if x["exp"] == exp and x["model"] == tag]
            r.sort(key=lambda x: float(x[xkey]))
            x = np.array([float(z[xkey]) for z in r])
            s = np.array([float(z["sig_final"]) for z in r])
            e = np.array([float(z["sig_sem"]) for z in r])
            ax.errorbar(x, s, yerr=e, marker="o", ms=3, capsize=2,
                        elinewidth=0.7, color=c, label=lab)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"$\sigma^2/N$ (patamar)")
        ax.legend()
    axes[0].set_xlabel(r"$m_{\max}$")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-3, 2.0)
    save(fig, "fig_mmax_N")


# ------------------------------------------------------------------- Fig. 9
OP_ORDER = ["cut+splice", "cut", "splice", "redesenho", "crossover", "mutacao"]
OP_LABEL = {"cut+splice": "corte +\nemenda", "cut": "corte", "splice": "emenda",
            "redesenho": "redesenho", "crossover": "crossover",
            "mutacao": "mutação"}


def load_ops():
    """Preferencia: results/op_ablation.csv (recalculado). Fallback: summary.csv."""
    import csv
    p = os.path.join(RES, "op_ablation.csv")
    if os.path.exists(p):
        return list(csv.DictReader(open(p, encoding="utf-8")))
    rows = list(csv.DictReader(open(os.path.join(RES, "summary.csv"),
                                    encoding="utf-8")))
    return [x for x in rows if x["exp"] == "op_ablation"]


def fig_ops():
    rows = load_ops()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5),
                             gridspec_kw=dict(wspace=0.42))
    for ax, tag, c, t in [(axes[0], "type1", COL[0.05], "(a) Tipo-I"),
                          (axes[1], "type2", COL[0.2], "(b) Tipo-II")]:
        rr = [x for x in rows if x["model"] == tag]
        rr.sort(key=lambda x: OP_ORDER.index(x["op"]))
        s = [float(x["sig_final"]) for x in rr]
        e = [float(x["sig_sem"]) for x in rr]
        m = [float(x["meanm"]) for x in rr]
        x = np.arange(len(rr))
        bars = ax.bar(x, s, yerr=e, color=c, width=0.72, capsize=2.4,
                      error_kw=dict(lw=0.9), label=r"$\sigma^2/N$")
        # destacar o melhor operador
        kbest = int(np.argmin(s))
        bars[kbest].set_facecolor("#b8451c")
        bars[kbest].set_hatch("//")
        ax.set_xticks(x)
        ax.set_xticklabels([OP_LABEL[z["op"]] for z in rr],
                           rotation=32, ha="right", rotation_mode="anchor",
                           fontsize=8.6)
        ax.set_ylabel(r"$\sigma^2/N$ (patamar)")
        ax.set_title(t)
        ax.set_ylim(0, max(s) * 1.34)
        ax2 = ax.twinx()
        ax2.plot(x, m, marker="D", ms=4.6, color="#6b2d0f", lw=1.3,
                 label=r"$\bar m$ final")
        ax2.set_ylabel(r"$\bar m$ final", color="#6b2d0f", fontsize=9)
        ax2.tick_params(axis="y", colors="#6b2d0f", labelsize=8.6)
        ax2.set_ylim(min(m) * 0.72, max(m) * 1.12)
    save(fig, "fig_ops")


# ------------------------------------------------------------------ Fig. 10
def fig_riqueza():
    f = os.path.join(RES, "wealth_by_m.csv")
    if not os.path.exists(f):
        return
    d = np.loadtxt(f, skiprows=1)
    d = d[d[:, 2] > 100]
    d = d[~np.isnan(d).any(axis=1)]
    fig, ax = plt.subplots(figsize=(3.15, 2.25))
    ax.plot(d[:, 0], d[:, 3] / 10000.0, "o-", ms=4.4, lw=1.6, color=COL[0.05])
    ax.axhline(0.5, color="0.6", ls=":", lw=1.0)
    ax.text(15.4, 0.499, "50%", fontsize=8, color="0.45", ha="right", va="bottom")
    ax.set_xlabel(r"memória do agente $m$")
    ax.set_ylabel("riqueza média / $T$")
    ax.set_xticks(d[:, 0].astype(int))
    save(fig, "fig_riqueza")


if __name__ == "__main__":
    print("gerando figuras...")
    fig_series("type1", 500, "fig_type1_series")
    fig_series("type2", 500, "fig_type2_series")
    fig_mdist("type1", "fig_type1_mdist", gc=1200, title="Tipo-I ($p=0.05$)")
    fig_mdist("type2", "fig_type2_mdist", gc=1200, title="Tipo-II ($p=0.05$)")
    fig_comparacao()
    fig_contraste()
    for fn in (fig_p_sweep, fig_mmax_N, fig_ops, fig_riqueza):
        try:
            fn()
        except Exception as e:
            print("  (pulado)", fn.__name__, "->", e)
