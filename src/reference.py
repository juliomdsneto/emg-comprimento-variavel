"""
Implementacao de referencia (lenta, pura Python) dos modelos de Yang et al. (2004).

Objetivo: (i) servir de segunda implementacao independente para validar os kernels
Numba; (ii) verificar ESTRUTURALMENTE os operadores geneticos (o corte deve produzir
exatamente uma das metades da tabela antiga; a emenda deve conter a tabela antiga como
prefixo ou sufixo, com o restante aleatorio).

Este arquivo e intencionalmente literal em relacao ao texto do artigo.
"""

import random
from statistics import variance


class AgentTypeI:
    """Agente do modelo Type-I: S estrategias de MESMO comprimento (mesmo m)."""

    def __init__(self, m, S, rng):
        self.m = m
        self.S = S
        n = 1 << m
        self.strats = [[rng.randint(0, 1) for _ in range(n)] for _ in range(S)]
        self.score = [0.0] * S

    def reset(self):
        self.score = [0.0] * self.S

    def cut(self, rng):
        """Corta cada estrategia em duas metades e guarda uma delas (sorteada)."""
        assert self.m > 1
        half = 1 << (self.m - 1)
        new = []
        for s in self.strats:
            off = 0 if rng.random() < 0.5 else half
            new.append(s[off:off + half])
        self.strats = new
        self.m -= 1
        self.reset()

    def splice(self, rng, mmax):
        """Acrescenta 2^m bits aleatorios a esquerda ou a direita."""
        assert self.m < mmax
        n = 1 << self.m
        new = []
        for s in self.strats:
            pad = [rng.randint(0, 1) for _ in range(n)]
            new.append(pad + s if rng.random() < 0.5 else s + pad)
        self.strats = new
        self.m += 1
        self.reset()


class AgentTypeII:
    """Agente do modelo Type-II: S estrategias com comprimentos independentes."""

    def __init__(self, mmin, mmax, S, rng):
        self.S = S
        self.strats = []
        self.ms = []
        for _ in range(S):
            m = rng.randint(mmin, mmax)
            self.ms.append(m)
            self.strats.append([rng.randint(0, 1) for _ in range(1 << m)])
        self.score = [0.0] * S

    def highest(self, rng):
        best, bv = 0, -1e18
        for s in range(self.S):
            v = self.score[s] + rng.random()
            if v > bv:
                bv, best = v, s
        return best

    def two_highest(self, rng):
        ks = [self.score[s] + rng.random() for s in range(self.S)]
        order = sorted(range(self.S), key=lambda s: -ks[s])
        return order[0], order[1]

    def lowest(self, rng):
        ks = [self.score[s] + rng.random() for s in range(self.S)]
        return min(range(self.S), key=lambda s: ks[s])

    def two_lowest(self, rng):
        ks = [self.score[s] + rng.random() for s in range(self.S)]
        order = sorted(range(self.S), key=lambda s: ks[s])
        return order[0], order[1]

    def cut(self, rng, mmin):
        """Corta a melhor estrategia em duas metades e substitui as duas piores."""
        s_hi = self.highest(rng)
        if self.ms[s_hi] <= mmin:
            return False
        half = 1 << (self.ms[s_hi] - 1)
        src = self.strats[s_hi]
        a = src[0:half]
        b = src[half:2 * half]
        t1, t2 = self.two_lowest(rng)
        if rng.random() < 0.5:
            a, b = b, a
        self.strats[t1], self.ms[t1] = a[:], self.ms[s_hi] - 1
        self.strats[t2], self.ms[t2] = b[:], self.ms[s_hi] - 1
        self.score[t1] = self.score[t2] = 0.0
        return True

    def splice(self, rng, mmax):
        """Concatena as duas melhores (ordem sorteada) e substitui a pior."""
        s1, s2 = self.two_highest(rng)
        m1, m2 = self.ms[s1], self.ms[s2]
        if max(m1, m2) >= mmax:
            return False
        a, b = (s1, s2) if m1 >= m2 else (s2, s1)
        ma, mb = max(m1, m2), min(m1, m2)
        first, second = self.strats[a], self.strats[b]
        if rng.random() < 0.5:
            first, second = second, first
        new = first + second
        new += [rng.randint(0, 1) for _ in range((1 << ma) - (1 << mb))]
        assert len(new) == 1 << (ma + 1), (len(new), ma, mb)
        t = self.lowest(rng)
        self.strats[t] = new
        self.ms[t] = ma + 1
        self.score[t] = 0.0
        return True


def run_type1(N=51, S=2, p=0.1, ngen=100, T=1000, mmin=1, mmax=6, seed=1,
              check=True):
    """Retorna (sigma2_por_geracao, media_m_por_geracao)."""
    rng = random.Random(seed)
    agents = [AgentTypeI(rng.randint(mmin, mmax), S, rng) for _ in range(N)]
    hist = [rng.randint(0, 1) for _ in range(15)]
    sig, mm = [], []
    for g in range(ngen):
        wealth = [0] * N
        A = []
        for _ in range(T):
            idx = 0
            for b in range(15):  # bit 0 = mais recente
                idx |= hist[b] << b
            pred, best = [], []
            for i, ag in enumerate(agents):
                bs, bv = 0, -1e18
                for s in range(S):
                    v = ag.score[s] + rng.random()
                    if v > bv:
                        bv, bs = v, s
                best.append(bs)
                pred.append(ag.strats[bs][idx & ((1 << ag.m) - 1)])
            a = sum(pred)
            W = 1 if 2 * a < N else 0
            for i, ag in enumerate(agents):
                for s in range(S):
                    if ag.strats[s][idx & ((1 << ag.m) - 1)] == W:
                        ag.score[s] += 1.0
                if pred[i] == W:
                    wealth[i] += 1
            A.append(a)
            hist = [W] + hist[:-1]
        sig.append(variance(A))
        mm.append(sum(a.m for a in agents) / N)

        k = max(1, int(p * N + 0.5))
        key = [(wealth[i], rng.random()) for i in range(N)]
        order = [i for i, _ in sorted(enumerate(key), key=lambda t: t[1])]
        for i in order[:k]:
            ag = agents[i]
            old = [s[:] for s in ag.strats]
            oldm = ag.m
            op = 0 if rng.random() < 0.5 else 1
            if op == 0 and oldm <= mmin:
                continue
            if op == 1 and oldm >= mmax:
                continue
            if op == 0:
                ag.cut(rng)
                if check:
                    assert ag.m == oldm - 1
                    half = 1 << ag.m
                    for s_new, s_old in zip(ag.strats, old):
                        assert s_new == s_old[:half] or s_new == s_old[half:]
            else:
                ag.splice(rng, mmax)
                if check:
                    assert ag.m == oldm + 1
                    n = 1 << oldm
                    for s_new, s_old in zip(ag.strats, old):
                        assert (s_new[:n] == s_old and len(s_new) == 2 * n) or \
                               (s_new[n:] == s_old and len(s_new) == 2 * n)
    return sig, mm


def run_type2(N=51, S=4, p=0.1, ngen=100, T=1000, mmin=1, mmax=6, seed=1,
              check=True):
    rng = random.Random(seed)
    agents = [AgentTypeII(mmin, mmax, S, rng) for _ in range(N)]
    hist = [rng.randint(0, 1) for _ in range(15)]
    sig, mm = [], []
    for g in range(ngen):
        wealth = [0] * N
        A = []
        for _ in range(T):
            idx = 0
            for b in range(15):
                idx |= hist[b] << b
            pred = []
            for ag in agents:
                s, bv = 0, -1e18
                for q in range(S):
                    v = ag.score[q] + rng.random()
                    if v > bv:
                        bv, s = v, q
                pred.append(ag.strats[s][idx & ((1 << ag.ms[s]) - 1)])
            a = sum(pred)
            W = 1 if 2 * a < N else 0
            for i, ag in enumerate(agents):
                for s in range(S):
                    if ag.strats[s][idx & ((1 << ag.ms[s]) - 1)] == W:
                        ag.score[s] += 1.0
                if pred[i] == W:
                    wealth[i] += 1
            A.append(a)
            hist = [W] + hist[:-1]
        sig.append(variance(A))
        mm.append(sum(sum(a.ms) for a in agents) / (N * S))

        k = max(1, int(p * N + 0.5))
        order = sorted(range(N), key=lambda i: (wealth[i], rng.random()))
        for i in order[:k]:
            ag = agents[i]
            op = 0 if rng.random() < 0.5 else 1
            if op == 0:
                ag.cut(rng, mmin)
            else:
                ag.splice(rng, mmax)
            if check:
                for m_, s_ in zip(ag.ms, ag.strats):
                    assert mmin <= m_ <= mmax
                    assert len(s_) == 1 << m_
    return sig, mm
