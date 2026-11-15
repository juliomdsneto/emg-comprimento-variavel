"""
Figuras didáticas para os SLIDES (versões simplificadas, com fonte grande).

Ficam separadas das figuras do artigo (`artigo/figures/`) para não mexer nelas.
Saída: ../apresentacao/figures/

Gera:
    fig_ciclo.pdf       - diagrama do ciclo de uma geração (caixas e setas)
    fig_mg_slide.pdf    - curva do MG básico (só ela, com nomes nas variáveis)
    fig_mdist_slide.pdf - 3 "fotos" da distribuição de memória (Tipo-I)
    fig_ops_slide.pdf   - 3 barras: corte+emenda, só corte, só emenda
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "results")
FIG = os.path.join(AQUI, "..", "apresentacao", "figures")
os.makedirs(FIG, exist_ok=True)

AZUL, VINHO, VERDE, CINZA = "#12457a", "#b8451c", "#1f7a45", "#3d3d3d"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.linewidth": 1.0,
    "axes.labelsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def salvar(fig, nome):
    fig.savefig(os.path.join(FIG, nome + ".pdf"))
    fig.savefig(os.path.join(FIG, nome + ".png"), dpi=300)
    plt.close(fig)
    print("  fig:", nome)


# ------------------------------------------------------------------ ciclo
def fig_ciclo():
    """Diagrama do ciclo de uma geração (4 caixas + seta de retorno)."""
    fig, ax = plt.subplots(figsize=(10.6, 2.85))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 2.85)
    ax.axis("off")

    caixas = [
        (0.15, "1", "Os N agentes jogam\n10.000 rodadas"),
        (2.75, "2", "Cada um soma os\nacertos (riqueza)"),
        (5.35, "3", "Os p% com menos\nacertos são escolhidos"),
        (7.95, "4", "Cada um faz\ncorte OU emenda"),
    ]
    larg, alt, y0 = 2.25, 1.30, 1.00

    for x, num, texto in caixas:
        ax.add_patch(FancyBboxPatch(
            (x, y0), larg, alt, boxstyle="round,pad=0.05,rounding_size=0.16",
            facecolor="#eef2f8", edgecolor=AZUL, lw=1.6))
        ax.add_patch(plt.Circle((x + 0.26, y0 + alt - 0.02), 0.185,
                                color=AZUL, zorder=5))
        ax.text(x + 0.26, y0 + alt - 0.02, num, color="white", ha="center",
                va="center", fontsize=11.5, fontweight="bold", zorder=6)
        ax.text(x + larg / 2, y0 + alt / 2 - 0.05, texto, ha="center",
                va="center", fontsize=12, linespacing=1.3)

    for x in (0.15 + larg, 2.75 + larg, 5.35 + larg):
        ax.add_patch(FancyArrowPatch(
            (x + 0.03, y0 + alt / 2), (x + 0.30, y0 + alt / 2),
            arrowstyle="-|>", mutation_scale=18, lw=2.0, color=AZUL))

    # seta de retorno, por baixo das caixas (arco raso, longe dos textos)
    ax.add_patch(FancyArrowPatch(
        (7.95 + larg / 2, y0 - 0.10), (0.15 + larg / 2, y0 - 0.10),
        connectionstyle="arc3,rad=-0.13", arrowstyle="-|>",
        mutation_scale=18, lw=2.0, color=VINHO, linestyle="--"))
    ax.text(5.25, 0.12, "+ 1 geração  (a memória m pode mudar)",
            ha="center", va="center", fontsize=12.5, color=VINHO)

    ax.text(0.15, 2.80, "Uma geração do EMG de Yang et al. (2004)",
            fontsize=13, color=CINZA, ha="left", va="top")
    salvar(fig, "fig_ciclo")


# ------------------------------------------------------------- curva do MG
def fig_mg_slide():
    mg = np.loadtxt(os.path.join(RES, "mg_curve.csv"), skiprows=1)
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    ax.plot(mg[:, 0], mg[:, 1], "o-", color=CINZA, ms=7, lw=2.2)
    k = int(np.argmin(mg[:, 1]))
    ax.annotate("melhor memória: m = 6\n" + r"($\sigma^2/N = 0{,}060$)",
                xy=(mg[k, 0], mg[k, 1] * 0.72), xytext=(6.6, 0.031),
                fontsize=12, color=CINZA, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.45"))
    ax.axhline(0.25, color="0.65", ls=":", lw=1.4)
    ax.text(12.4, 0.27, "sem coordenação", fontsize=11.5, color="0.45",
            ha="right", va="bottom")
    ax.set_xlabel("memória m  (quantas rodadas passadas a tabela enxerga)")
    ax.set_ylabel(r"$\sigma^2/N$  (descoordenação)")
    ax.set_xlim(0.5, 12.8)
    ax.set_ylim(0.02, 3.2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
    salvar(fig, "fig_mg_slide")


# ------------------------------------------------------- distribuição de m
def fig_mdist_slide():
    z = np.load(os.path.join(RES, "long_type1.npz"))
    snap = z["snapshots"][0].mean(axis=0)      # p=0.05, média sobre as execuções
    gens = [int(round(snap[k, -1])) for k in range(snap.shape[0])]
    mm = np.arange(snap.shape[1] - 1)
    quer = [0, 750, 1750]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    cores = {0: "#9aa7b8", 750: "#4a7fb5", 1750: VERDE}
    for g in quer:
        k = min(range(len(gens)), key=lambda j: abs(gens[j] - g))
        if snap[k, :-1].sum() < 0.5:
            continue
        ax.plot(mm[1:], snap[k, 1:-1], "o-", ms=6, lw=2.0, color=cores[g],
                label="geração %d" % gens[k])
    ax.set_xlabel("memória m")
    ax.set_ylabel("nº de agentes")
    ax.set_xticks(mm[1::2])
    ax.set_xlim(0.6, mm[-1] + 0.4)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11.5)
    salvar(fig, "fig_mdist_slide")


# --------------------------------------------------------------- operadores
def fig_ops_slide():
    import csv
    linhas = list(csv.DictReader(
        open(os.path.join(RES, "op_ablation.csv"), encoding="utf-8")))
    t1 = {r["op"]: r for r in linhas if r["model"] == "type1"}
    nomes = ["cut+splice", "cut", "splice"]
    rotulos = ["corte + emenda\n(como no artigo)", "só corte", "só emenda"]
    vals = [float(t1[n]["sig_final"]) for n in nomes]
    errs = [float(t1[n]["sig_sem"]) for n in nomes]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    cores = [AZUL, VINHO, "#8a8a8a"]
    barras = ax.bar(range(3), vals, yerr=errs, color=cores, width=0.62,
                    capsize=5, error_kw=dict(lw=1.2))
    barras[1].set_hatch("//")
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.012, ("%.3f" % v).replace(".", ","), ha="center",
                fontsize=12.5, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(rotulos, fontsize=12)
    ax.set_ylabel(r"$\sigma^2/N$ no fim  (menor = melhor)")
    ax.set_ylim(0, max(vals) * 1.32)
    ax.axhline(0.173, color="0.55", ls="--", lw=1.3)
    ax.text(-0.42, 0.181, "sem evolução (0,173)", ha="left", fontsize=11.5,
            color="0.45")
    salvar(fig, "fig_ops_slide")


# ------------------------------------------------- comparacao com o MG (slide)
def fig_comparacao_slide():
    """Versao para o slide: so os NOSSOS dados, sem as linhas do artigo."""
    mg = np.loadtxt(os.path.join(RES, "mg_curve.csv"), skiprows=1)
    zl1 = np.load(os.path.join(RES, "long_type1.npz"))
    zl2 = np.load(os.path.join(RES, "long_type2.npz"))

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.plot(mg[:, 0], mg[:, 1], "o-", color=CINZA, ms=6, lw=2.0,
            label="MG básico (variação de $m$)")
    for z, lab, cor, ls in [(zl1, "EMG Tipo-I", AZUL, "--"),
                            (zl2, "EMG Tipo-II", VERDE, "-.")]:
        i = int(np.argmin(np.abs(z["ps"] - 0.05)))
        v = z["sigma2"][i][:, -100:].mean()
        ax.axhline(v, color=cor, ls=ls, lw=2.2,
                   label=lab + r" ($p=0{,}05$)")

    ax.axhline(0.25, color="0.65", ls=":", lw=1.3)
    ax.text(12.4, 0.27, "escolha aleatória", fontsize=11.5, color="0.45",
            ha="right", va="bottom")
    ax.set_xlabel("memória $m$ do MG básico")
    ax.set_ylabel(r"$\sigma^2/N$  (menor = melhor)")
    ax.set_xlim(0.5, 12.8)
    ax.set_ylim(0.010, 3.5)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    salvar(fig, "fig_comparacao_slide")


if __name__ == "__main__":
    print("gerando figuras dos slides...")
    fig_ciclo()
    fig_mg_slide()
    fig_mdist_slide()
    fig_ops_slide()
    fig_comparacao_slide()
