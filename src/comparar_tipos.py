"""
Roda o Tipo-I e o Tipo-II com os MESMOS parâmetros e coloca os resultados lado a
lado, para você ver as diferenças.

Qual código cada tipo executa:
    Tipo-I  ->  emg.emg1  via  emg.run_type1
    Tipo-II ->  emg.emg2  via  emg.run_type2

Exemplos
--------
    python comparar_tipos.py                       # cenário do artigo
    python comparar_tipos.py --p 0.1 --geracoes 200
    python comparar_tipos.py --operador corte --mmax 8
    python comparar_tipos.py --saida comparacao_final
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
        description="Compara Tipo-I e Tipo-II com os mesmos parâmetros.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--geracoes", type=int, default=300)
    ap.add_argument("--passos", type=int, default=10000)
    ap.add_argument("--N", type=int, default=101)
    ap.add_argument("--mmin", type=int, default=1)
    ap.add_argument("--mmax", type=int, default=15)
    ap.add_argument("--operador", default="misto",
                    choices=["misto", "corte", "emenda", "redesenho",
                             "crossover", "mutacao"])
    ap.add_argument("--repeticoes", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--saida", default="comparacao")
    ap.add_argument("--ultimas", type=int, default=30)
    args = ap.parse_args()

    base = dict(N=args.N, p=args.p, ngen=args.geracoes, T=args.passos,
                mmin=args.mmin, mmax=args.mmax,
                opmode=uteis.opcode(args.operador), snap_every=2)
    sementes = list(range(args.repeticoes))

    resultados = {}
    for tipo, S in (("I", 2), ("II", 4)):
        params = dict(base, S=S)
        print(f"\n>>> Tipo-{tipo}: S={S}, N={args.N}, p={args.p}, "
              f"gerações={args.geracoes}, T={args.passos}, m≤{args.mmax}, "
              f"operador={args.operador}")
        saidas = uteis.rodar(tipo, params, sementes, workers=args.workers)
        sig = uteis.empilhar(saidas, "sigma2", args.N)
        mserie = uteis.empilhar(saidas, "meanm")
        pat, sem = uteis.patamar(sig, args.ultimas)
        resultados[tipo] = dict(
            sig=sig, mserie=mserie, pat=pat, sem=sem,
            m_final=float(mserie[:, -args.ultimas:].mean()),
            dist=uteis.distribuicao_final(saidas, args.N),
            S=S)

    # ---------------------------- tabela no console
    i, ii = resultados["I"], resultados["II"]
    print("\n" + "=" * 62)
    print(f"  COMPARAÇÃO  (p={args.p}, m_max={args.mmax}, "
          f"{args.repeticoes} execuções, operador={args.operador})")
    print("=" * 62)
    print(f"  {'':<22}{'Tipo-I':>16}{'Tipo-II':>16}")
    print(f"  {'S (estratégias)':<22}{i['S']:>16}{ii['S']:>16}")
    print(f"  {'patamar σ²/N':<22}{i['pat']:>11.4f} ±{i['sem']:>3.3f}"
          f"{ii['pat']:>11.4f} ±{ii['sem']:>3.3f}")
    print(f"  {'memória média final':<22}{i['m_final']:>16.2f}{ii['m_final']:>16.2f}")
    razao = ii["pat"] / i["pat"] if i["pat"] > 0 else float("nan")
    print(f"  {'razão II / I':<22}{razao:>16.2f}")
    melhor = "Tipo-I" if i["pat"] < ii["pat"] else "Tipo-II"
    print(f"\n  Menor variância (mais cooperação): {melhor}")
    print("=" * 62 + "\n")

    # ---------------------------- CSV
    pasta = os.path.join(AQUI, "..", "results", args.saida)
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, "comparacao.csv")
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tipo", "geracao", "sigma2_sobre_N_media",
                    "sigma2_sobre_N_sem", "memoria_media"])
        for tipo in ("I", "II"):
            r = resultados[tipo]
            media = r["sig"].mean(axis=0)
            erro = (r["sig"].std(axis=0, ddof=1) / np.sqrt(r["sig"].shape[0])
                    if r["sig"].shape[0] > 1 else np.zeros_like(media))
            for g in range(r["sig"].shape[1]):
                w.writerow([tipo, g + 1, f"{media[g]:.6f}", f"{erro[g]:.6f}",
                            f"{r['mserie'][:, g].mean():.4f}"])
    print(f"  dados salvos em: {caminho}")

    # ---------------------------- figura
    plt = uteis.estilos()
    fig = plt.figure(figsize=(9.0, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    for tipo in ("I", "II"):
        r = resultados[tipo]
        x = np.arange(1, r["sig"].shape[1] + 1)
        media = r["sig"].mean(axis=0)
        erro = (r["sig"].std(axis=0, ddof=1) / np.sqrt(r["sig"].shape[0])
                if r["sig"].shape[0] > 1 else np.zeros_like(media))
        ax0.plot(x, media, color=uteis.CORES[tipo], lw=1.9,
                 label=f"{uteis.ROTULOS[tipo]} ($S={r['S']}$)")
        ax0.fill_between(x, media - erro, media + erro,
                         color=uteis.CORES[tipo], alpha=0.20, lw=0)
    ax0.axhline(0.25, color="0.65", ls=":", lw=1.0)
    ax0.text(1, 0.246, "escolha aleatória (0,25)", fontsize=8, color="0.45",
             ha="left", va="top")
    ax0.set_xlabel("geração")
    ax0.set_ylabel(r"$\sigma^2/N$")
    ax0.set_title(f"Evolução da variância — $p={args.p}$, "
                  f"$m_{{\\max}}={args.mmax}$, operador {args.operador}",
                  fontsize=10.5)
    ax0.legend(loc="upper right")
    ax0.set_xlim(1, resultados["I"]["sig"].shape[1])

    largura = 0.38
    for desloc, tipo in ((-largura / 2, "I"), (largura / 2, "II")):
        d = resultados[tipo]["dist"]
        ms = np.arange(len(d))
        ax1.bar(ms + desloc, d, width=largura, color=uteis.CORES[tipo],
                label=uteis.ROTULOS[tipo])
    ax1.set_xlabel(r"memória $m$")
    ax1.set_ylabel("fração final")
    ax1.set_title("Distribuição final de memórias", fontsize=10)
    ax1.set_xticks(np.arange(len(resultados["I"]["dist"])))
    ax1.legend(fontsize=8)

    tipos = ["I", "II"]
    pats = [resultados[t]["pat"] for t in tipos]
    sems = [resultados[t]["sem"] for t in tipos]
    ax2.bar([0, 1], pats, yerr=sems, color=[uteis.CORES[t] for t in tipos],
            width=0.6, capsize=4)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels([uteis.ROTULOS[t] for t in tipos])
    ax2.set_ylabel(r"$\sigma^2/N$ final")
    ax2.set_title("Patamar final", fontsize=10)
    ax2.set_ylim(0, max(pats) * 1.35)
    for k, (valor, erro) in enumerate(zip(pats, sems)):
        ax2.text(k, valor + erro + max(pats) * 0.05, f"{valor:.4f}",
                 ha="center", fontsize=9)

    base_saida = os.path.join(pasta, "comparacao")
    uteis.salvar_figura(fig, base_saida)
    plt.close(fig)
    print()


if __name__ == "__main__":
    main()
