"""Estatística das séries longas: média, erro padrão, IC95% e testes.

Usa as 20 execuções independentes por configuração (sementes 500-519).
Saída: tabela pronta para o artigo.
"""

import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(AQUI, "..", "results")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from scipy import stats
    TEM_SCIPY = True
except Exception:
    TEM_SCIPY = False


def ic95(x):
    """Média, erro padrão e meia-largura do IC 95% (t de Student)."""
    n = len(x)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    if TEM_SCIPY:
        t = stats.t.ppf(0.975, n - 1)
    else:                       # aproximação normal, conservadora
        t = 1.96
    return m, sem, t * sem


def welch(a, b):
    """Teste t de Welch (bilateral). Devolve t, gl e p."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se
    gl = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * stats.t.sf(abs(t), gl) if TEM_SCIPY else float("nan")
    return t, gl, p, se


def carregar(tag, sufixo="_20"):
    z = np.load(os.path.join(RES, f"long_{tag}{sufixo}.npz"))
    sig, ps = z["sigma2"], z["ps"]
    mm = z["meanm"]
    return sig, mm, ps


def main():
    print("scipy disponível:", TEM_SCIPY)
    dados = {}
    for tag in ("type1", "type2"):
        sig, mm, ps = carregar(tag)
        print(f"\n=== {tag}  ({sig.shape[1]} execuções) ===")
        print(f"{'p':>5} {'média':>9} {'ep':>8} {'IC95':>16} {'m':>6}")
        for j, p in enumerate(ps):
            x = sig[j][:, -100:].mean(axis=1)
            m, sem, h = ic95(x)
            dados[(tag, float(p))] = x
            print(f"{p:5.2f} {m:9.5f} {sem:8.5f}  [{m-h:.5f}, {m+h:.5f}]"
                  f" {mm[j][:, -100:].mean():6.2f}")

    print("\n=== diferenças (t de Welch, bilateral) ===")
    comparacoes = [
        ("Tipo-I p=0,05 x Tipo-II p=0,05", ("type1", 0.05), ("type2", 0.05)),
        ("Tipo-I p=0,05 x p=0,10", ("type1", 0.05), ("type1", 0.10)),
        ("Tipo-I p=0,10 x p=0,20", ("type1", 0.10), ("type1", 0.20)),
        ("Tipo-I p=0,05 x p=0,20", ("type1", 0.05), ("type1", 0.20)),
        ("Tipo-II p=0,05 x p=0,20", ("type2", 0.05), ("type2", 0.20)),
    ]
    for nome, ka, kb in comparacoes:
        a, b = dados[ka], dados[kb]
        t, gl, p, se = welch(a, b)
        d = a.mean() - b.mean()
        ic = (d - 1.96 * se, d + 1.96 * se)
        print(f"{nome:34s} dif={d:+.5f}  IC95=[{ic[0]:+.5f},{ic[1]:+.5f}]"
              f"  t={t:+6.2f} gl={gl:4.1f} p={p:.2e}")


if __name__ == "__main__":
    main()
