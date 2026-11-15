"""
Funções compartilhadas pelos scripts de uso direto (rodar_teste, comparar_tipos,
animar). Concentram a tradução de nomes para os parâmetros dos kernels e a
execução em paralelo.

IMPORTANTE — qual código roda cada modelo do artigo:

    Tipo-I  ->  emg.emg1  (kernel Numba)  <-  emg.run_type1  (wrapper Python)
    Tipo-II ->  emg.emg2  (kernel Numba)  <-  emg.run_type2  (wrapper Python)

Os kernels `emg1`/`emg2` são a implementação de fato: recebem tudo por posição e
devolvem tuplas de arrays. Os wrappers `run_type1`/`run_type2` só ajustam o buffer
L = 2^mmax e devolvem um dicionário — é o que estes scripts usam.
"""

import os
import sys
import numpy as np
from multiprocessing import Pool

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from emg import run_type1, run_type2   # noqa: E402


def _forcar_utf8():
    """Evita UnicodeEncodeError quando a saida e' redirecionada (cp1252)."""
    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_forcar_utf8()

# opmode: a mesma convenção nos dois modelos
OPERADORES = {
    "misto": 0,       # corte e emenda 50/50 (como no artigo)
    "corte": 1,
    "emenda": 2,
    "redesenho": 3,   # sorteia estratégia nova com o mesmo m (Li et al.)
    "crossover": 4,
    "mutacao": 5,
    "mutação": 5,
}


def opcode(nome):
    """Converte o nome do operador no código `opmode` usado pelos kernels."""
    chave = str(nome).strip().lower()
    if chave not in OPERADORES:
        opcoes = ", ".join(sorted(set(OPERADORES) - {"mutação"}))
        raise SystemExit(f"operador inválido: {nome!r}. Opções: {opcoes}")
    return OPERADORES[chave]


def modelo(tipo):
    """'I'/'i' -> run_type1 ; 'II'/'ii' -> run_type2."""
    t = str(tipo).strip().upper().replace("TIPO", "").replace("-", "").strip()
    if t in ("I", "1"):
        return "I", run_type1
    if t in ("II", "2"):
        return "II", run_type2
    raise SystemExit(f"tipo inválido: {tipo!r}. Use I ou II.")


def _trabalho(argumento):
    tipo, params, seed = argumento
    _, f = modelo(tipo)
    return f(seed=seed, **params)


def rodar(tipo, params, sementes, workers=4, verboso=True):
    """Executa o modelo em várias sementes e devolve a lista de saídas."""
    trabalhos = [(tipo, params, s) for s in sementes]
    if workers <= 1 or len(trabalhos) == 1:
        saidas = []
        for t in trabalhos:
            saidas.append(_trabalho(t))
            if verboso:
                print(f"   semente {t[2]} concluída", flush=True)
        return saidas
    with Pool(min(workers, len(trabalhos))) as pool:
        saidas = pool.map(_trabalho, trabalhos)
    if verboso:
        print(f"   {len(saidas)} execuções concluídas", flush=True)
    return saidas


def empilhar(saidas, chave, N=None):
    """Empilha uma chave das saídas; se N for dado, normaliza sigma2 por N."""
    arr = np.array([o[chave] for o in saidas], dtype=float)
    if chave == "sigma2" and N:
        arr = arr / N
    return arr


def patamar(sigma2_por_geracao, ultimas=30):
    """Média do patamar final e o erro padrão entre execuções.

    `sigma2_por_geracao` tem forma (execuções, gerações).
    """
    final = sigma2_por_geracao[:, -ultimas:].mean(axis=1)
    sem = final.std(ddof=1) / np.sqrt(len(final)) if len(final) > 1 else 0.0
    return float(final.mean()), float(sem)


def distribuicao_final(saidas, N):
    """Distribuição de memórias na última 'foto', como fração (0..1)."""
    snaps = np.array([o["snapshots"] for o in saidas], dtype=float)
    # snaps: (execuções, fotos, m+1); a última coluna guarda a geração.
    # A última foto pode ficar vazia (o laço de gerações para em ngen-1), então
    # procuramos de trás para frente a última foto realmente preenchida.
    ultima = 0
    for k in range(snaps.shape[1] - 1, -1, -1):
        if snaps[:, k, :-1].sum() > 0.5:
            ultima = k
            break
    d = snaps[:, ultima, :]
    total = d[:, :-1].sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return (d[:, :-1] / total).mean(axis=0)


def imprimir_resumo(nome, sig, m_final, N, extra=""):
    pat, sem = patamar(sig)
    barra = "=" * 58
    print(barra)
    print(f"  {nome}")
    print(barra)
    print(f"  execuções          : {sig.shape[0]}")
    print(f"  gerações           : {sig.shape[1]}")
    print(f"  N (agentes)        : {N}")
    print(f"  patamar final sigma2/N : {pat:.4f} +/- {sem:.4f}")
    print(f"  memoria media final  : {m_final:.2f}")
    if extra:
        print(f"  {extra}")
    print(barra)
    return pat, sem


def salvar_figura(fig, destino_base):
    """Salva em .png (300 dpi) e .pdf ao lado, criando a pasta se preciso."""
    pasta = os.path.dirname(os.path.abspath(destino_base))
    os.makedirs(pasta, exist_ok=True)
    fig.savefig(destino_base + ".png", dpi=300)
    fig.savefig(destino_base + ".pdf")
    print(f"  figura salva em: {destino_base}.png e .pdf")


def estilos():
    """Estilo comum das figuras (o mesmo do artigo)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })
    return plt


CORES = {"I": "#12457a", "II": "#1f7a45"}
ROTULOS = {"I": "Tipo-I", "II": "Tipo-II"}
