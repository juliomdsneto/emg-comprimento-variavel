"""
Simuladores do Jogo da Minoria (Minority Game, MG) e do Jogo da Minoria
Evolucionario (EMG) com estrategias de comprimento variavel, conforme
Yang, Wang, Wu e Xie, Physica A 339 (2004) 583-590.

Convencoes
----------
* N agentes, dois lados rotulados 0 e 1. Quem esta no lado MINORITARIO ganha 1 ponto.
* Estrategia com memoria m: tabela de 2^m bits (uma escolha por historia possivel).
* O historico global e um inteiro `hist` cujo BIT 0 e o resultado mais recente
  (lado minoritario do passo anterior). Logo, a linha consultada por um agente de
  memoria m e `hist & (2^m - 1)`.
* Estrategias ficam em buffers de tamanho fixo L = 2^m_max por agente/estrategia;
  apenas as primeiras 2^m posicoes sao validas (o resto e lixo nunca lido).

Kernels compilados com Numba (nopython) e RNG proprio (splitmix64) para
reprodutibilidade bit-a-bit e paralelismo por processos.

Notas de interpretacao (ambiguidades do artigo, verem docs/notacao.md):
  I1. Atualizacao das pontuacoes virtuais: o texto do artigo diz que cada agente
      "adds a virtual point to those strategies predicting winning side correctly",
      isto e, TODAS as S estrategias sao atualizadas (update_all=1).  A variante
      classica de Challet-Zhang atualiza apenas a estrategia jogada (update_all=0).
      Ambas sao implementadas.
  I2. Operador invalido na fronteira (cut com m=1, splice com m=m_max): o agente
      NAO muda nada (no-op).  Variante "espelho" (aplicar o outro operador) pode ser
      testada com op_fallback=1.
  I3. Selecao por riqueza: k = round(p*N) agentes mais pobres; empates quebrados
      aleatoriamente.
  I4. Type-II, splice com m1 > m2: nova estrategia tem m = m1+1, resultante de
      S1 (2^m1 bits) || S2 (2^m2 bits) || preenchimento aleatorio de (2^m1 - 2^m2)
      bits, com a ordem de S1/S2 sorteada.
"""

import numpy as np
from numba import njit

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_INV53 = 1.1102230246251565e-16  # 2^-53


# --------------------------------------------------------------------- RNG
@njit(inline="always", cache=True)
def _next(state):
    """splitmix64: avanca `state` (array uint64 de 1 posicao) e devolve 64 bits."""
    x = state[0] + _GOLD
    state[0] = x
    z = x
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    z = z ^ (z >> np.uint64(31))
    return z


@njit(inline="always", cache=True)
def _rnd(state):
    """Uniforme em [0,1) com 53 bits."""
    return float(_next(state) >> np.uint64(11)) * _INV53


@njit(inline="always", cache=True)
def _rndint(state, n):
    """Inteiro uniforme em [0,n) (int64 explicito, evita promocao p/ float)."""
    return np.int64(_next(state) >> np.uint64(1)) % n


@njit(inline="always", cache=True)
def _seed_state(seed):
    st = np.zeros(1, dtype=np.uint64)
    st[0] = np.uint64(seed) * np.uint64(2654435761) + np.uint64(88172645463325252)
    return st


# -------------------------------------------------------- nucleo de um passo
@njit(cache=True)
def _decide(N, S, strat, mask, hist, score, state, best_s, pred):
    """Cada agente joga a estrategia de maior pontuacao virtual (empate aleatorio).

    Devolve A = numero de agentes que escolheram o lado 1.
    """
    A = 0
    for i in range(N):
        idx = hist & mask[i]
        bs = 0
        bv = -1.0e18
        for s in range(S):
            v = score[i, s] + _rnd(state)
            if v > bv:
                bv = v
                bs = s
        best_s[i] = bs
        p = strat[i, bs, idx]
        pred[i] = p
        A += int(p)
    return A


@njit(cache=True)
def _update(N, S, strat, mask, hist, score, pred, best_s, W, update_all):
    for i in range(N):
        idx = hist & mask[i]
        if update_all == 1:
            for s in range(S):
                if int(strat[i, s, idx]) == W:
                    score[i, s] += 1.0
        else:
            if int(pred[i]) == W:
                score[i, best_s[i]] += 1.0


@njit(inline="always", cache=True)
def _gini(w, N):
    tot = 0.0
    for i in range(N):
        tot += w[i]
    if tot <= 0.0:
        return 0.0
    d = 0.0
    for i in range(N):
        for j in range(N):
            d += abs(w[i] - w[j])
    return d / (2.0 * N * tot)


# =========================================================================
#  Modelo com memoria por agente (Type-I; e tambem o MG basico com p=0)
# =========================================================================
@njit(cache=True, boundscheck=False)
def emg1(N, S, p, ngen, T, mmin, mmax, L, seed, opmode=0, op_fallback=0,
         update_all=1, reset_scores=1, warm=0, snap_every=0, track_wealth=0,
         hbits=0):
    """EMG Type-I: cada agente tem S estrategias de MESMO comprimento m_i.

    opmode: 0 = cut+splice 50/50 (artigo)
            1 = apenas cut
            2 = apenas splice
            3 = redesenho aleatorio completo (modelo de Li et al., 2000)
            4 = crossover de um ponto (modelo de Marko et al., 2003)
            5 = mutacao pontual (bit-flip) mantendo m
    Retorna dicionario com series por geracao.
    """
    state = _seed_state(seed)
    strat = np.zeros((N, S, L), dtype=np.int8)
    score = np.zeros((N, S), dtype=np.float64)
    mask = np.zeros(N, dtype=np.int64)
    mcur = np.zeros(N, dtype=np.int64)
    best_s = np.zeros(N, dtype=np.int64)
    pred = np.zeros(N, dtype=np.int8)
    wealth = np.zeros(N, dtype=np.float64)
    key = np.zeros(N, dtype=np.float64)
    tmp = np.zeros(L, dtype=np.int8)
    tmp2 = np.zeros(L, dtype=np.int8)

    for i in range(N):
        m = mmin + _rndint(state, mmax - mmin + 1)
        mcur[i] = m
        mask[i] = (1 << m) - 1
        for s in range(S):
            for j in range(1 << m):
                strat[i, s, j] = np.int8(_next(state) & np.uint64(1))

    hist = 0
    for j in range(15):
        hist = (hist << 1) | int(_next(state) & np.uint64(1))

    nsnap = 0
    if snap_every > 0:
        nsnap = ngen // snap_every + 1
    snaps = np.zeros((max(nsnap, 1), mmax + 2), dtype=np.float64)

    sig = np.zeros(ngen)
    meanA = np.zeros(ngen)
    meanm = np.zeros(ngen)
    gini = np.zeros(ngen)
    mw = np.zeros(ngen)
    wsum = np.zeros(mmax + 2)
    wcnt = np.zeros(mmax + 2)
    wby = np.zeros((ngen, mmax + 2))
    mplayed = np.zeros(N, dtype=np.float64)

    hb = 1 << hbits if hbits > 0 else 0
    Hgen = np.zeros(ngen)

    for g in range(ngen):
        for i in range(N):
            wealth[i] = 0.0
            mplayed[i] = 0.0
        sA = 0.0
        sA2 = 0.0
        nA = 0
        hsum = np.zeros(max(hb, 1), dtype=np.float64)
        hcnt = np.zeros(max(hb, 1), dtype=np.float64)
        for t in range(T):
            A = _decide(N, S, strat, mask, hist, score, state, best_s, pred)
            W = 1 if A * 2 < N else 0
            _update(N, S, strat, mask, hist, score, pred, best_s, W, update_all)
            for i in range(N):
                if int(pred[i]) == W:
                    wealth[i] += 1.0
                mplayed[i] += float(mcur[i])
            if t >= warm:
                sA += A
                sA2 += A * A
                nA += 1
                if hb > 0:
                    j = hist & (hb - 1)
                    hsum[j] += A
                    hcnt[j] += 1.0
            hist = ((hist << 1) | W) & 32767
        mu = sA / nA
        sig[g] = sA2 / nA - mu * mu
        meanA[g] = mu
        if hb > 0:
            H = 0.0
            for j in range(hb):
                if hcnt[j] > 0:
                    H += ((hsum[j] / hcnt[j] - mu) / N) ** 2
            Hgen[g] = H / hb
        # distribuicao de m (sobre agentes) e riqueza media por m
        mtot = 0.0
        for i in range(N):
            mtot += float(mcur[i])
        meanm[g] = mtot / N
        gini[g] = _gini(wealth, N)
        mt = 0.0
        for i in range(N):
            mt += wealth[i]
        mw[g] = mt / N
        if track_wealth == 1 and g >= ngen // 2:
            for i in range(N):
                j = mcur[i]
                wsum[j] += wealth[i]
                wcnt[j] += 1.0
                wby[g, j] += 1.0
        if snap_every > 0 and (g % snap_every == 0):
            k = g // snap_every
            for i in range(N):
                snaps[k, mcur[i]] += 1.0
            snaps[k, mmax + 1] = float(g)

        # ------------- fim da geracao: os p% mais pobres ajustam estrategias
        if p <= 0.0:
            k = 0
        else:
            k = int(p * N + 0.5)
            if k < 1:
                k = 1
        if k > N:
            k = N
        for i in range(N):
            key[i] = wealth[i] + _rnd(state) * 0.999
        order = np.argsort(key)
        for oi in range(k):
            i = order[oi]
            m = mcur[i]
            # opmode -> codigo do operador (0=cut, 1=splice, 2=mutacao,
            #                            3=redesenho aleatorio, 4=crossover)
            if opmode == 0:
                op = 0 if _rnd(state) < 0.5 else 1
            elif opmode == 1:
                op = 0
            elif opmode == 2:
                op = 1
            elif opmode == 5:
                op = 2
            else:
                op = opmode
            # validade do operador
            if op == 0 and m <= mmin:
                if op_fallback == 1:
                    op = 1
                else:
                    continue
            if op == 1 and m >= mmax:
                if op_fallback == 1:
                    op = 0
                else:
                    continue
            if op == 0:  # cut: cada estrategia vira uma das metades
                half = 1 << (m - 1)
                for s in range(S):
                    off = 0 if _rnd(state) < 0.5 else half
                    for j in range(half):
                        tmp[j] = strat[i, s, off + j]
                    for j in range(half):
                        strat[i, s, j] = tmp[j]
                mcur[i] = m - 1
                mask[i] = half - 1
            elif op == 1:  # splice: acrescenta 2^m bits aleatorios a esq/dir
                ln = 1 << m
                left = _rnd(state) < 0.5
                for s in range(S):
                    for j in range(ln):
                        tmp2[j] = strat[i, s, j]
                    for j in range(ln):
                        tmp[j] = np.int8(_next(state) & np.uint64(1))
                    if left:
                        for j in range(ln):
                            strat[i, s, j] = tmp[j]
                            strat[i, s, ln + j] = tmp2[j]
                    else:
                        for j in range(ln):
                            strat[i, s, j] = tmp2[j]
                            strat[i, s, ln + j] = tmp[j]
                mcur[i] = m + 1
                mask[i] = (1 << (m + 1)) - 1
            elif op == 2:  # mutacao pontual: inverte UM bit de cada estrategia
                for s in range(S):
                    j = _rndint(state, 1 << m)
                    strat[i, s, j] = np.int8(1 - strat[i, s, j])
            elif op == 3:  # redesenho aleatorio completo (Li et al., 2000)
                for s in range(S):
                    for j in range(1 << m):
                        strat[i, s, j] = np.int8(_next(state) & np.uint64(1))
            elif op == 4:  # crossover de um ponto (Marko et al., 2003)
                if S >= 2:
                    a = _rndint(state, S)
                    b = _rndint(state, S)
                    if b == a:
                        b = (a + 1) % S
                    cp = _rndint(state, 1 << m)
                    for j in range(1 << m):
                        tmp[j] = strat[i, a, j]
                        tmp2[j] = strat[i, b, j]
                    keep = _rnd(state) < 0.5
                    for j in range(1 << m):
                        if j < cp:
                            strat[i, a, j] = tmp[j] if keep else tmp2[j]
                        else:
                            strat[i, a, j] = tmp2[j] if keep else tmp[j]
            if reset_scores == 1:
                for s in range(S):
                    score[i, s] = 0.0
    return (sig, meanA, meanm, gini, mw, snaps, wsum, wcnt, wby, Hgen)


def run_type1(N=101, S=2, p=0.05, ngen=500, T=10000, mmin=1, mmax=15,
              seed=0, opmode=0, op_fallback=0, update_all=1, reset_scores=1,
              warm=0, snap_every=0, track_wealth=0, hbits=0):
    """Wrapper Python do kernel emg1 -> devolve dicionario."""
    L = 1 << mmax
    (sig, meanA, meanm, gini, mw, snaps, wsum, wcnt, wby,
     H) = emg1(N, S, p, ngen, T, mmin, mmax, L, seed, opmode, op_fallback,
               update_all, reset_scores, warm, snap_every, track_wealth, hbits)
    return dict(sigma2=sig, meanA=meanA, meanm=meanm, gini=gini, meanwealth=mw,
                snapshots=snaps, wealth_by_m_sum=wsum, wealth_by_m_cnt=wcnt,
                mdist_by_gen=wby, H=H)


# =========================================================================
#  Type-II: cada uma das S estrategias do agente tem seu proprio m
# =========================================================================
@njit(cache=True, boundscheck=False)
def emg2(N, S, p, ngen, T, mmin, mmax, L, seed, opmode=0, op_fallback=0,
         update_all=1, reset_scores=1, warm=0, snap_every=0, track_wealth=0,
         static=0):
    """EMG Type-II (S=4 no artigo): estrategias com comprimentos diferentes.

    Operadores (opmode=0):
      cut    -> pega a estrategia de MAIOR pontuacao, corta em duas metades e
                substitui as DUAS de MENOR pontuacao.
      splice -> pega as DUAS de MAIOR pontuacao, concatena em ordem sorteada e
                substitui a de MENOR pontuacao; se max(m1,m2)==mmax, nao faz nada.
      Nova memoria no splice: m1+1 com m1 = max(m1,m2) e preenchimento aleatorio
      de (2^m1 - 2^m2) bits.
    opmode=3 -> redesenho aleatorio da estrategia de menor pontuacao (mesmo m).
    """
    state = _seed_state(seed)
    strat = np.zeros((N, S, L), dtype=np.int8)
    score = np.zeros((N, S), dtype=np.float64)
    ms = np.zeros((N, S), dtype=np.int64)
    best_s = np.zeros(N, dtype=np.int64)
    pred = np.zeros(N, dtype=np.int8)
    wealth = np.zeros(N, dtype=np.float64)
    key = np.zeros(N, dtype=np.float64)
    key2 = np.zeros(S, dtype=np.float64)
    tmp = np.zeros(L, dtype=np.int8)
    tmp2 = np.zeros(L, dtype=np.int8)

    for i in range(N):
        for s in range(S):
            m = mmin + _rndint(state, mmax - mmin + 1)
            ms[i, s] = m
            for j in range(1 << m):
                strat[i, s, j] = np.int8(_next(state) & np.uint64(1))

    hist = 0
    for j in range(15):
        hist = (hist << 1) | int(_next(state) & np.uint64(1))

    nsnap = 0
    if snap_every > 0:
        nsnap = ngen // snap_every + 1
    snaps = np.zeros((max(nsnap, 1), mmax + 2), dtype=np.float64)

    sig = np.zeros(ngen)
    meanA = np.zeros(ngen)
    meanm = np.zeros(ngen)
    gini = np.zeros(ngen)
    mw = np.zeros(ngen)
    wsum = np.zeros(mmax + 2)
    wcnt = np.zeros(mmax + 2)

    for g in range(ngen):
        for i in range(N):
            wealth[i] = 0.0
        sA = 0.0
        sA2 = 0.0
        nA = 0
        for t in range(T):
            A = 0
            for i in range(N):
                bs = 0
                bv = -1.0e18
                for s in range(S):
                    v = score[i, s] + _rnd(state)
                    if v > bv:
                        bv = v
                        bs = s
                best_s[i] = bs
                pv = strat[i, bs, hist & ((1 << ms[i, bs]) - 1)]
                pred[i] = pv
                A += int(pv)
            W = 1 if A * 2 < N else 0
            for i in range(N):
                if update_all == 1:
                    for s in range(S):
                        if int(strat[i, s, hist & ((1 << ms[i, s]) - 1)]) == W:
                            score[i, s] += 1.0
                else:
                    if int(pred[i]) == W:
                        score[i, best_s[i]] += 1.0
                if int(pred[i]) == W:
                    wealth[i] += 1.0
            hist = ((hist << 1) | W) & 32767
            if t >= warm:
                sA += A
                sA2 += A * A
                nA += 1
        mu = sA / nA
        sig[g] = sA2 / nA - mu * mu
        meanA[g] = mu
        mt = 0.0
        for i in range(N):
            for s in range(S):
                mt += float(ms[i, s])
        meanm[g] = mt / (N * S)
        gini[g] = _gini(wealth, N)
        mt = 0.0
        for i in range(N):
            mt += wealth[i]
        mw[g] = mt / N
        if snap_every > 0 and (g % snap_every == 0):
            k = g // snap_every
            for i in range(N):
                for s in range(S):
                    snaps[k, ms[i, s]] += 1.0
            snaps[k, mmax + 1] = float(g)

        if static == 1:
            continue
        # ------------- fim da geracao
        k = int(p * N + 0.5)
        if k < 1:
            k = 1
        if k > N:
            k = N
        for i in range(N):
            key[i] = wealth[i] + _rnd(state) * 0.999
        order = np.argsort(key)
        for oi in range(k):
            i = order[oi]
            op = opmode
            # mesma convencao do emg1: 0=misto, 1=corte, 2=emenda,
            #                          3=redesenho, 4=crossover, 5=mutacao
            if opmode == 0:
                op = 0 if _rnd(state) < 0.5 else 1
            elif opmode == 1:
                op = 0
            elif opmode == 2:
                op = 1
            elif opmode == 5:
                op = 2
            # ordena as estrategias do agente por pontuacao (desempate aleatorio)
            for s in range(S):
                key2[s] = score[i, s] + _rnd(state)
            oidx = np.argsort(key2)
            s_lo1 = int(oidx[0])
            s_lo2 = int(oidx[1])
            s_hi = int(oidx[S - 1])
            s_second = int(oidx[S - 2])
            if op == 0:
                if ms[i, s_hi] <= mmin:
                    if op_fallback == 1:
                        op = 1
                    else:
                        continue
            elif op == 1:
                if ms[i, s_hi] >= mmax or ms[i, s_second] >= mmax:
                    if op_fallback == 1:
                        op = 0
                    else:
                        continue
            if op == 0:
                m1 = ms[i, s_hi]
                half = 1 << (m1 - 1)
                # copia a estrategia fonte
                for j in range(1 << m1):
                    tmp[j] = strat[i, s_hi, j]
                off1 = 0 if _rnd(state) < 0.5 else half
                off2 = 0 if _rnd(state) < 0.5 else half
                for j in range(half):
                    strat[i, s_lo1, j] = tmp[off1 + j]
                    strat[i, s_lo2, j] = tmp[off2 + j]
                ms[i, s_lo1] = m1 - 1
                ms[i, s_lo2] = m1 - 1
                if reset_scores == 1:
                    score[i, s_lo1] = 0.0
                    score[i, s_lo2] = 0.0
            elif op == 1:
                m1 = ms[i, s_hi]
                s2 = s_second
                m2 = ms[i, s2]
                a = m1 if m1 > m2 else m2
                b = m2 if m1 > m2 else m1
                sa = s_hi if m1 > m2 else s2
                sb = s2 if m1 > m2 else s_hi
                la = 1 << a
                lb = 1 << b
                for j in range(la):
                    tmp[j] = strat[i, sa, j]
                for j in range(lb):
                    tmp2[j] = strat[i, sb, j]
                first_a = _rnd(state) < 0.5
                pos = 0
                if first_a:
                    for j in range(la):
                        strat[i, s_lo1, pos] = tmp[j]
                        pos += 1
                    for j in range(lb):
                        strat[i, s_lo1, pos] = tmp2[j]
                        pos += 1
                else:
                    for j in range(lb):
                        strat[i, s_lo1, pos] = tmp2[j]
                        pos += 1
                    for j in range(la):
                        strat[i, s_lo1, pos] = tmp[j]
                        pos += 1
                while pos < 2 * la:
                    strat[i, s_lo1, pos] = np.int8(_next(state) & np.uint64(1))
                    pos += 1
                ms[i, s_lo1] = a + 1
                if reset_scores == 1:
                    score[i, s_lo1] = 0.0
            elif op == 3:  # redesenho aleatorio com o mesmo m
                m = ms[i, s_lo1]
                for j in range(1 << m):
                    strat[i, s_lo1, j] = np.int8(_next(state) & np.uint64(1))
                if reset_scores == 1:
                    score[i, s_lo1] = 0.0
            elif op == 2:  # mutacao pontual: inverte UM bit da pior estrategia
                m = ms[i, s_lo1]
                j = _rndint(state, 1 << m)
                strat[i, s_lo1, j] = np.int8(1 - strat[i, s_lo1, j])
                if reset_scores == 1:
                    score[i, s_lo1] = 0.0
            elif op == 4:  # crossover de um ponto entre as duas melhores
                m1 = ms[i, s_hi]
                m2 = ms[i, s_second]
                mn = m1 if m1 < m2 else m2
                if mn < mmin:
                    continue
                ln = 1 << mn
                cp = _rndint(state, ln)
                for j in range(ln):
                    tmp[j] = strat[i, s_hi, j]
                    tmp2[j] = strat[i, s_second, j]
                keep = _rnd(state) < 0.5
                for j in range(ln):
                    if j < cp:
                        strat[i, s_lo1, j] = tmp[j] if keep else tmp2[j]
                    else:
                        strat[i, s_lo1, j] = tmp2[j] if keep else tmp[j]
                ms[i, s_lo1] = mn
                if reset_scores == 1:
                    score[i, s_lo1] = 0.0
    return (sig, meanA, meanm, gini, mw, snaps, wsum, wcnt)


def run_type2(N=101, S=4, p=0.05, ngen=500, T=10000, mmin=1, mmax=15,
              seed=0, opmode=0, op_fallback=0, update_all=1, reset_scores=1,
              warm=0, snap_every=0, track_wealth=0, static=0):
    """Wrapper Python do kernel emg2 -> devolve dicionario."""
    L = 1 << mmax
    (sig, meanA, meanm, gini, mw, snaps, wsum,
     wcnt) = emg2(N, S, p, ngen, T, mmin, mmax, L, seed, opmode, op_fallback,
                  update_all, reset_scores, warm, snap_every, track_wealth,
                  static)
    return dict(sigma2=sig, meanA=meanA, meanm=meanm, gini=gini, meanwealth=mw,
                snapshots=snaps, wealth_by_m_sum=wsum, wealth_by_m_cnt=wcnt)

