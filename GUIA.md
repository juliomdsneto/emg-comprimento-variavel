# Guia rápido — rodar, mudar parâmetros e visualizar

Este guia responde a três perguntas:

1. qual código executa o **Tipo-I** e qual executa o **Tipo-II** do artigo;
2. como **mudar as variáveis** para testar outros cenários;
3. como **plotar e visualizar** os resultados.

---

## 1. Qual código executa cada modelo

Tudo está em `src/emg.py`. Cada modelo tem **dois** níveis:

| Modelo do artigo | Kernel (a simulação de fato) | Wrapper (como você usa) |
|---|---|---|
| **Tipo-I** (S=2, mesmo comprimento por agente) | `emg1()` — linha ~131 | `run_type1()` — linha ~339 |
| **Tipo-II** (S=4, comprimentos independentes) | `emg2()` — linha ~356 | `run_type2()` — linha ~585 |

```
emg.py
├── emg1(...)        ← KERNEL do Tipo-I  (compilado com Numba)
│    └── run_type1(...)   ← wrapper: calcula L = 2^mmax e devolve um dicionário
├── emg2(...)        ← KERNEL do Tipo-II (compilado com Numba)
│    └── run_type2(...)   ← wrapper
├── _decide(), _update(), _next() ...   ← funções internas
```

- Os **kernels** (`emg1`, `emg2`) recebem tudo por posição e devolvem tuplas de
  arrays. São eles que contêm a regra do jogo, os operadores genéticos e o laço
  de gerações.
- Os **wrappers** (`run_type1`, `run_type2`) só preparam o buffer
  `L = 2^mmax` e devolvem um dicionário com nomes — é o que os scripts usam.
- A implementação de referência, em Python puro e escrita a partir do texto do
  artigo, está em `reference.py` (`run_type1`/`run_type2` de lá são outra coisa:
  servem só para validação, e são ~100× mais lentas).

### O que muda entre os dois modelos (no código)

| | Tipo-I (`emg1`) | Tipo-II (`emg2`) |
|---|---|---|
| estruturas | `strat[N, S, L]` + `mcur[N]` | `strat[N, S, L]` + `ms[N, S]` |
| comprimento | **um** `m` por agente (as 2 estratégias iguais) | **um** `m` por estratégia (4 independentes) |
| corte | corta **cada** estratégia e guarda uma metade | corta a **melhor** e substitui as **duas piores** |
| emenda | acrescenta `2^m` bits aleatórios em cada estratégia | concatena as **duas melhores** e substitui a **pior** (`m = max+1`) |
| distribuição de `m` | sobre os `N` agentes | sobre as `4N` estratégias |

---

## 2. Trocar as variáveis

### Pela linha de comando (sem editar código)

```powershell
python rodar_teste.py --tipo I --p 0.10 --geracoes 400
python rodar_teste.py --tipo II --mmax 10 --N 201
python rodar_teste.py --tipo I --operador corte --repeticoes 8
```

| Variável | Opção | Padrão | O que faz |
|---|---|---|---|
| modelo | `--tipo` | `I` | `I` → `emg1`; `II` → `emg2` |
| agentes | `--N` | 101 | deve ser ímpar (evita empate) |
| estratégias | `--S` | 2 (I) / 4 (II) | estratégias por agente |
| fração atualizada | `--p` | 0.05 | % dos mais pobres que mudam de estratégia |
| gerações | `--geracoes` | 300 | cada geração = `T` passos |
| passos por geração | `--passos` | 10000 | o artigo usa 10^4; menos = mais rápido e mais ruidoso |
| memória mínima | `--mmin` | 1 | |
| memória máxima | `--mmax` | 15 | teto de `m`; afeta muito o resultado |
| operador | `--operador` | `misto` | ver tabela abaixo |
| repetições | `--repeticoes` | 4 | nº de sementes independentes |
| últimos N gens | `--ultimas` | 30 | janela usada para o "patamar final" |
| pasta de saída | `--saida` | `teste` | vai para `resultados/<saida>/` |

**Operadores** (`--operador`) — a mesma convenção nos dois modelos:

| nome | código | o que faz | muda `m`? |
|---|---|---|---|
| `misto` | 0 | corte e emenda 50/50 (como no artigo) | sim |
| `corte` | 1 | só corte | sim (−1) |
| `emenda` | 2 | só emenda | sim (+1) |
| `redesenho` | 3 | sorteia estratégia nova com o mesmo `m` (Li et al.) | não |
| `crossover` | 4 | cruza duas tabelas | não |
| `mutacao` | 5 | inverte um bit | não |

### Direto no Python

```python
import sys
sys.path.insert(0, r"C:\Users\Julio\Documents\sistemas evolutivos\src")
import emg

# Tipo-I: 200 gerações, guardando uma "foto" a cada 25 gerações
saida = emg.run_type1(N=101, S=2, p=0.05, ngen=200, T=10000,
                      mmin=1, mmax=15, seed=0, opmode=0, snap_every=25)

# Tipo-II: mesmos parâmetros
saida2 = emg.run_type2(N=101, S=4, p=0.05, ngen=200, T=10000,
                       mmin=1, mmax=15, seed=0, opmode=0, snap_every=25)
```

### Variáveis que só existem na chamada Python

| parâmetro | efeito |
|---|---|
| `seed` | semente (reprodutibilidade) |
| `op_fallback` | 1 = quando o operador é impossível na fronteira (`m=1` no corte, `m=mmax` na emenda), aplica o outro operador em vez de não fazer nada |
| `update_all` | 1 = todas as estratégias ganham ponto quando acertam (leitura do artigo); 0 = só a estratégia jogada |
| `reset_scores` | 1 = zera os escores das estratégias alteradas (como no artigo); 0 = mantém |
| `warm` | passos iniciais descartados no cálculo da variância |
| `snap_every` | de quantas em quantas gerações guardar a distribuição de `m` (0 = não guarda) |
| `static` | só no Tipo-II: 1 = não evolui (controle estático) |

---

## 3. Plotar e visualizar

### Um teste isolado

```powershell
python rodar_teste.py --tipo I --p 0.05 --geracoes 300
```

Gera em `resultados/teste/`:

- `resultado.png` (e `.pdf`) — 2 painéis: σ²/N ao longo das gerações e a
  distribuição final de memórias;
- `serie.csv` — geração, média e erro padrão de σ²/N, memória média;
- `distribuicao_m.csv` — fração de agentes/estratégias em cada `m`.

### Comparar Tipo-I × Tipo-II com os mesmos parâmetros

```powershell
python comparar_tipos.py --p 0.05 --geracoes 300
```

Gera `resultados/comparacao/` com:

- `comparacao.png` — (a) as duas curvas de σ²/N; (b) as duas distribuições
  finais de `m`; (c) as barras do patamar final;
- `comparacao.csv` — tudo em formato tabular;
- e imprime no console a tabela comparativa, incluindo a razão II/I.

### Ver a execução "acontecendo" (GIF)

```powershell
python animar.py --tipo I --geracoes 150 --fps 8
python animar.py --tipo II --geracoes 150
```

Gera `resultados/<saida>/<saida>.gif` — um quadro por geração, mostrando a
distribuição de memórias se deslocando e a curva de σ²/N crescendo. Também
salva dois quadros soltos (`*_inicio.png`, `*_fim.png`) para slides.

### Pelo atalho, sem se preocupar com o Python

```powershell
.\rodar.ps1 tipo1        # exemplo Tipo-I  (300 gerações, 4 sementes)
.\rodar.ps1 tipo2        # exemplo Tipo-II
.\rodar.ps1 comparar     # comparação lado a lado
.\rodar.ps1 animar       # GIF do Tipo-I
```

Todos aceitam argumentos extras: `.\rodar.ps1 tipo1 --p 0.1 --mmax 8`.

---

## 4. O que vem em cada saída do `run_type*`

| chave | forma | significado |
|---|---|---|
| `sigma2` | `(gerações,)` | **variância** de `A(t)` em cada geração (divida por `N` para σ²/N) |
| `meanA` | `(gerações,)` | média de `A(t)` (≈ N/2) |
| `meanm` | `(gerações,)` | memória média (agentes no Tipo-I, estratégias no Tipo-II) |
| `gini` | `(gerações,)` | índice de Gini da riqueza |
| `meanwealth` | `(gerações,)` | riqueza média por agente |
| `snapshots` | `(fotos, mmax+2)` | distribuição de `m`; **a última coluna é o número da geração** |
| `H` | `(gerações,)` | previsibilidade (só Tipo-I, requer `hbits>0`) |
| `wealth_by_m_sum/cnt` | `(mmax+2,)` | riqueza acumulada por memória (só Tipo-I, requer `track_wealth=1`) |

Cuidado com dois detalhes:

1. `sigma2` é a variância **crua**; σ²/N é `sigma2 / N`.
2. A última foto de `snapshots` pode estar vazia (o laço de gerações termina em
   `ngen-1`). Os scripts já procuram a última foto preenchida.

---

## 5. Receitas úteis

**Ver o efeito de `p`** (o artigo usa 0,05 / 0,1 / 0,2):
```powershell
python rodar_teste.py --tipo I --p 0.05 --saida p005
python rodar_teste.py --tipo I --p 0.10 --saida p010
python rodar_teste.py --tipo I --p 0.20 --saida p020
```

**Ver o efeito do teto de memória** (0,0063 com `mmax=3` vs 0,097 com 15):
```powershell
python rodar_teste.py --tipo I --mmax 3  --saida mmax3
python rodar_teste.py --tipo I --mmax 8  --saida mmax8
python rodar_teste.py --tipo I --mmax 15 --saida mmax15
```

**Isolar o papel de cada operador** (o corte é o que funciona):
```powershell
python rodar_teste.py --tipo I --operador corte     --saida op_corte
python rodar_teste.py --tipo I --operador emenda    --saida op_emenda
python rodar_teste.py --tipo I --operador crossover --saida op_crossover
```

**Confirmar que a heterogeneidade não é evolução**: no Python,
`run_type2(..., static=1)` desliga a evolução mantendo a distribuição inicial.

---

## 6. Tempos de referência (Ryzen 5 5600GT)

| comando | tempo |
|---|---|
| `rodar_teste.py` (300 gerações, 4 sementes, Tipo-I) | ~35 s |
| `rodar_teste.py` (300 gerações, 4 sementes, Tipo-II) | ~1,5 min |
| `comparar_tipos.py` (300 gerações, 4 sementes) | ~2 min |
| `animar.py` (150 gerações, T=3000) | ~40 s |
| `test_kernels.py` | ~15 s |

Para experimentar rápido, reduza `--passos` (ex.: `--passos 3000`) e
`--repeticoes 2`.
