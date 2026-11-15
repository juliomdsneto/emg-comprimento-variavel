"""
Roda UM teste do Tipo-I ou do Tipo-II, com os parâmetros que você escolher, e
gera uma figura + CSV para inspecionar o resultado.

Qual código cada tipo executa:
    --tipo I   ->  emg.emg1  via  emg.run_type1   (S = 2, mesmo comprimento)
    --tipo II  ->  emg.emg2  via  emg.run_type2   (S = 4, comprimentos distintos)

Exemplos
--------
    # o cenário do artigo, Tipo-I
    python rodar_teste.py --tipo I --p 0.05

    # Tipo-II com memória máxima 10 e 6 repetições
    python rodar_teste.py --tipo II --p 0.1 --mmax 10 --repeticoes 6

    # só corte, N = 201, 400 gerações
    python rodar_teste.py --tipo I --operador corte --N 201 --geracoes 400

    # ver o efeito de m_max
    python rodar_teste.py --tipo I --mmax 5 --saida mmax5
"""

import argparse
import csv
import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import uteis


def main():
    ap = argparse.ArgumentParser(
        description="Roda um teste do Tipo-I ou Tipo-II e gera figura + CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--tipo", default="I", help="I ou II")
    ap.add_argument("--p", type=float, default=0.05,
                    help="fração dos mais pobres que ajustam estratégias")
    ap.add_argument("--geracoes", type=int, default=300)
    ap.add_argument("--passos", type=int, default=10000,
                    help="passos por geração (T)")
    ap.add_argument("--N", type=int, default=101, help="número de agentes")
    ap.add_argument("--S", type=int, default=None,
                    help="estratégias por agente (padrão: 2 no Tipo-I, 4 no Tipo-II)")
    ap.add_argument("--mmin", type=int, default=1)
    ap.add_argument("--mmax", type=int, default=15)
    ap.add_argument("--operador", default="misto",
                    choices=["misto", "corte", "emenda", "redesenho",
                             "crossover", "mutacao"])
    ap.add_argument("--repeticoes", type=int, default=4, help="número de sementes")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--saida", default="teste", help="nome da pasta/arquivo de saída")
    ap.add_argument("--ultimas", type=int, default=30,
                    help="quantas gerações finais entram no patamar")
    args = ap.parse_args()

    tipo, _ = uteis.modelo(args.tipo)
    S = args.S if args.S is not None else (2 if tipo == "I" else 4)

    params = dict(N=args.N, S=S, p=args.p, ngen=args.geracoes, T=args.passos,
                  mmin=args.mmin, mmax=args.mmax,
                  opmode=uteis.opcode(args.operador), snap_every=2)
    sementes = list(range(args.repeticoes))

    print(f"\nRodando Tipo-{tipo}: N={args.N}, S={S}, p={args.p}, "
          f"gerações={args.geracoes}, T={args.passos}, "
          f"m em [{args.mmin},{args.mmax}], operador={args.operador}")
    print(f"Sementes: {sementes}\n")

    saidas = uteis.rodar(tipo, params, sementes, workers=args.workers)

    sig = uteis.empilhar(saidas, "sigma2", args.N)
    mserie = uteis.empilhar(saidas, "meanm")
    m_final = float(mserie[:, -args.ultimas:].mean())
    pat, sem = uteis.imprimir_resumo(
        f"Tipo-{tipo} | p={args.p} | m_max={args.mmax} | {args.operador}",
        sig, m_final, args.N)

    # ---------------------------- saída em disco
    pasta = os.path.join(AQUI, "..", "results", args.saida)
    os.makedirs(pasta, exist_ok=True)

    caminho_csv = os.path.join(pasta, "serie.csv")
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["geracao", "sigma2_sobre_N_media", "sigma2_sobre_N_sem",
                    "memoria_media"])
        media = sig.mean(axis=0)
        if sig.shape[0] > 1:
            erro = sig.std(axis=0, ddof=1) / np.sqrt(sig.shape[0])
        else:
            erro = np.zeros_like(media)
        for g in range(sig.shape[1]):
            w.writerow([g + 1, f"{media[g]:.6f}", f"{erro[g]:.6f}",
                        f"{mserie[:, g].mean():.4f}"])
    print(f"  série salva em: {caminho_csv}")

    dist = uteis.distribuicao_final(saidas, args.N)
    caminho_dist = os.path.join(pasta, "distribuicao_m.csv")
    with open(caminho_dist, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["m", "fracao"])
        for m, v in enumerate(dist):
            w.writerow([m, f"{v:.6f}"])
    print(f"  distribuição salva em: {caminho_dist}")

    # ---------------------------- figura
    plt = uteis.estilos()
    cor = uteis.CORES[tipo]
    x = np.arange(1, sig.shape[1] + 1)
    media = sig.mean(axis=0)
    erro = sig.std(axis=0, ddof=1) / np.sqrt(sig.shape[0]) if sig.shape[0] > 1 \
        else np.zeros_like(media)

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.0),
                           gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.28))

    ax[0].plot(x, media, color=cor, lw=1.8)
    ax[0].fill_between(x, media - erro, media + erro, color=cor, alpha=0.22, lw=0)
    ax[0].axhline(0.25, color="0.65", ls=":", lw=1.0)
    ax[0].text(x[-1], 0.246, "aleatório", fontsize=8, color="0.45",
               ha="right", va="top")
    ax[0].set_xlabel("geração")
    ax[0].set_ylabel(r"$\sigma^2/N$")
    ax[0].set_title(f"Tipo-{tipo} — evolução da variância", fontsize=10)
    ax[0].set_xlim(1, x[-1])

    ms = np.arange(len(dist))
    ax[1].bar(ms, dist, color=cor, width=0.75, alpha=0.9)
    ax[1].set_xlabel(r"memória $m$")
    ax[1].set_ylabel("fração dos agentes")
    ax[1].set_title("distribuição final de memórias", fontsize=10)
    ax[1].set_xticks(ms)
    ax[1].set_xlim(-0.7, len(dist) - 0.3)

    subt = (f"$\\sigma^2/N$ final = {pat:.4f} $\\pm$ {sem:.4f}   |   "
            f"$\\bar m$ final = {m_final:.2f}   |   "
            f"{args.repeticoes} execuções, $p={args.p}$, "
            f"$m_{{\\max}}={args.mmax}$, operador = {args.operador}")
    fig.text(0.5, -0.04, subt, ha="center", fontsize=8.5, color="0.25")

    base = os.path.join(pasta, "resultado")
    uteis.salvar_figura(fig, base)
    plt.close(fig)

    print(f"\nResumo: patamar {pat:.4f} +/- {sem:.4f};  m medio final {m_final:.2f}")
    print(f"Abra a figura: {base}.png\n")


if __name__ == "__main__":
    main()
