"""Figura esquematica dos operadores de corte e emenda (slides)."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "artigo", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "font.size": 10, "savefig.bbox": "tight"})

W, H = 0.3, 0.42
DARK, LIGHT = "#1b3a6b", "#e8edf7"


def row(ax, x0, y, bits, label=None, fs=9.5, lw=0.5):
    for k, b in enumerate(bits):
        ax.add_patch(Rectangle((x0 + k * W, y), W, H,
                               facecolor=DARK if b else LIGHT,
                               edgecolor="#5a6b85", lw=lw))
    if label:
        ax.text(x0 - 0.18, y + H / 2, label, ha="right", va="center",
                fontsize=fs)


def fig_operadores(fname="fig_operadores", big=False):
    rng = np.random.default_rng(7)
    b16 = rng.integers(0, 2, 16)
    b8a = b16[:8]
    rnd = rng.integers(0, 2, 8)
    fs = 11 if big else 10.2

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.15),
                             gridspec_kw=dict(wspace=0.34))
    # ---------------- (a) corte
    ax = axes[0]
    ax.set_title("(a) corte: $m \\rightarrow m-1$", fontsize=fs + 0.5, pad=6)
    row(ax, 0.95, 1.35, b16, "$2^m$ bits", fs=fs)
    ax.plot([0.95, 0.95 + 8 * W], [1.28, 1.28], color="#1b3a6b", lw=1.0)
    ax.text(0.95 + 4 * W, 1.10, "metade 1", ha="center", fontsize=fs - 1.5,
            color="#1b3a6b")
    ax.plot([0.95 + 8 * W, 0.95 + 16 * W], [1.28, 1.28], color="#b8461c", lw=1.0)
    ax.text(0.95 + 12 * W, 1.10, "metade 2", ha="center", fontsize=fs - 1.5,
            color="#b8461c")
    ax.add_patch(FancyArrowPatch((0.95 + 4 * W, 1.00), (0.95 + 4 * W, 0.62),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color="#444444"))
    row(ax, 0.95, 0.20, b8a, "metade\nmantida", fs=fs)
    ax.text(0.95 + 8 * W + 0.15, 0.41, "sorteio independente\npor estratégia",
            fontsize=fs - 1.5, va="center", ha="left", color="#444444")

    # ---------------- (b) emenda
    ax = axes[1]
    ax.set_title("(b) emenda: $m \\rightarrow m+1$", fontsize=fs + 0.5, pad=6)
    row(ax, 0.95, 1.62, b8a, "$2^m$ bits", fs=fs)
    row(ax, 0.95, 1.02, rnd, "aleatório", fs=fs)
    ax.add_patch(FancyArrowPatch((0.95 + 4 * W, 0.86), (0.95 + 4 * W, 0.66),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color="#444444"))
    ax.text(0.95 + 4.3 * W, 0.76, "concatena (esquerda ou direita,\npor sorteio)",
            fontsize=fs - 1.5, va="center", ha="left", color="#444444")
    row(ax, 0.95, 0.10, np.concatenate([b8a, rnd]),
        "nova\n$2^{m+1}$ bits", fs=fs)

    for ax in axes:
        ax.set_xlim(0, 6.6)
        ax.set_ylim(0, 2.15)
        ax.axis("off")
    for ext, dpi in (("pdf", None), ("png", 200)):
        fig.savefig(os.path.join(FIG, f"{fname}.{ext}"), dpi=dpi)
    plt.close(fig)
    print("fig:", fname)


if __name__ == "__main__":
    fig_operadores()
