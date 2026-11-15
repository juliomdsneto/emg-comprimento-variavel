"""
Animação (GIF) da evolução: a cada geração você vê a distribuição de memórias se
movendo e a variância caindo. É a versão "time-lapse" das figuras do artigo.

Qual código cada tipo executa:
    --tipo I   ->  emg.emg1  via  emg.run_type1
    --tipo II  ->  emg.emg2  via  emg.run_type2

Exemplos
--------
    python animar.py                          # Tipo-I, padrão
    python animar.py --tipo II --geracoes 120
    python animar.py --tipo I --p 0.1 --mmax 10 --saida anim_p01
"""

import argparse
import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import uteis


def main():
    ap = argparse.ArgumentParser(
        description="Gera um GIF mostrando a evolução geração a geração.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tipo", default="I", help="I ou II")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--geracoes", type=int, default=150,
                    help="número de quadros (um por geração)")
    ap.add_argument("--passos", type=int, default=3000,
                    help="passos por geração; menor = mais rápido")
    ap.add_argument("--N", type=int, default=101)
    ap.add_argument("--mmax", type=int, default=15)
    ap.add_argument("--operador", default="misto",
                    choices=["misto", "corte", "emenda", "redesenho",
                             "crossover", "mutacao"])
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--saida", default=None,
                    help="nome base do GIF (padrão: animacao_tipo<X>)")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    tipo, _ = uteis.modelo(args.tipo)
    S = 2 if tipo == "I" else 4
    op = uteis.opcode(args.operador)

    params = dict(N=args.N, S=S, p=args.p, ngen=args.geracoes, T=args.passos,
                  mmin=1, mmax=args.mmax, opmode=op, snap_every=1)

    print(f"\nSimulando Tipo-{tipo}: {args.geracoes} gerações x "
          f"{args.passos} passos (p={args.p}, m_max={args.mmax}, "
          f"operador={args.operador})...")
    saida = uteis.rodar(tipo, params, [args.semente], workers=1, verboso=False)[0]

    sig = np.asarray(saida["sigma2"], dtype=float) / args.N
    snaps = np.asarray(saida["snapshots"], dtype=float)
    # a última coluna de `snapshots` é o número da geração: descartamos
    ger = snaps[:, -1]
    dados = snaps[:, :-1]          # (gerações, m = 0..mmax)

    # o laço de gerações para em ngen-1; algumas linhas ficam vazias
    validas = [g for g in range(dados.shape[0]) if dados[g].sum() > 0.5]
    dados = dados[validas]
    sig = sig[validas]
    ger = ger[validas]
    n = len(validas)
    mm = np.arange(dados.shape[1])
    print(f"  {n} quadros gerados; sigma2/N final = {sig[-1]:.4f}")

    plt = uteis.estilos()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.ticker import MaxNLocator

    cor = uteis.CORES[tipo]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.7),
                                   gridspec_kw=dict(width_ratios=[1, 1.25],
                                                    wspace=0.26))

    ymax = dados.max() * 1.16 if dados.max() > 0 else 1.0
    barras = axL.bar(mm, dados[0], color=cor, width=0.75)
    axL.set_xlabel(r"memória $m$")
    axL.set_ylabel("nº de agentes/estratégias")
    axL.set_xticks(mm)
    axL.xaxis.set_major_locator(MaxNLocator(integer=True))
    axL.set_xlim(-0.7, mm[-1] + 0.7)
    axL.set_ylim(0, ymax)
    axL.yaxis.set_major_locator(MaxNLocator(integer=True))
    tituloL = axL.set_title("")

    ytop = max(0.27, float(sig.max()) * 1.12)
    axR.plot(np.arange(1, n + 1), sig, color=cor, lw=1.8, alpha=0.9)
    ponto, = axR.plot([1], [sig[0]], "o", ms=7, color="#b8451c", zorder=5)
    axR.axhline(0.25, color="0.65", ls=":", lw=1.0)
    axR.text(n, 0.252, "escolha aleatória", fontsize=8, color="0.45",
             ha="right", va="bottom")
    axR.set_xlabel("geração")
    axR.set_ylabel(r"$\sigma^2/N$")
    axR.set_xlim(1, n)
    axR.set_ylim(0, ytop)
    axR.set_title("variância ao longo do tempo", fontsize=10)

    subtitulo = fig.text(0.5, -0.05, "", ha="center", fontsize=9, color="0.25")

    def quadro(k):
        for barra, valor in zip(barras, dados[k]):
            barra.set_height(valor)
        tituloL.set_text(f"geração {int(ger[k])} — distribuição de memórias")
        ponto.set_data([k + 1], [sig[k]])
        m_bar = (dados[k] * mm).sum() / max(dados[k].sum(), 1)
        subtitulo.set_text(
            f"Tipo-{tipo}   p={args.p}   operador={args.operador}   "
            f"sigma²/N = {sig[k]:.4f}   m médio = {m_bar:.2f}")
        return list(barras) + [ponto, tituloL, subtitulo]

    anim = FuncAnimation(fig, quadro, frames=n, interval=1000.0 / args.fps,
                         blit=False, repeat_delay=800)

    nome = args.saida or f"animacao_tipo{tipo}"
    pasta = os.path.join(AQUI, "..", "results", nome)
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, nome + ".gif")
    anim.save(caminho, writer=PillowWriter(fps=args.fps))

    # dois quadros soltos, para usar em slide
    for alvo, rotulo in ((0, "inicio"), (n - 1, "fim")):
        quadro(alvo)
        fig.savefig(os.path.join(pasta, f"{nome}_{rotulo}.png"), dpi=200)

    plt.close(fig)
    print(f"  GIF salvo em: {caminho}")
    print(f"  quadros soltos em: {pasta} (*_inicio.png, *_fim.png)\n")


if __name__ == "__main__":
    main()
