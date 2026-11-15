# Jogo da Minoria Evolucionário com estratégias de comprimento variável

Replicação e análise crítica de **Yang, Wang, Wu & Xie, "Searching good strategies in
evolutionary minority game using variable length genetic algorithm", Physica A 339
(2004) 583–590**, com uma seção sobre o que mudou em vinte anos
(Morsky, Zhuang & Zhou, arXiv:2307.11846, 2024).

Implementação independente, em Python + Numba, do Jogo da Minoria (MG) e dos dois
modelos evolucionários do artigo, com uma segunda implementação de referência em
Python puro usada para verificação.

## Resultados principais

| | σ²/N (patamar, 2000 gerações) | memória média final |
|---|---|---|
| MG básico (melhor `m` = 6) | 0,060 | — |
| **Tipo-I** (p = 0,05) | **0,0157** ± 0,0006 | 2,87 |
| Tipo-II (p = 0,05) | 0,0981 ± 0,0022 | 7,80 |
| controle estático Tipo-I | 0,173 | 7,94 |

- O Tipo-I **reproduz** o artigo: a memória encolhe e o sistema bate o melhor MG de
  memória fixa por um fator de 4. O valor coincide com o publicado (≈0,016).
- O Tipo-II **não** reproduz: fica acima do mínimo do MG básico e ~6× pior que o
  Tipo-I. A distribuição de memórias não se move (m̄: 8,0 → 7,8).
- A **ablação de operadores** mostra que o motor do mecanismo é o **corte**
  (0,042 sozinho, contra 0,097 da mistura 50/50 do artigo). Emenda isolada é
  prejudicial (0,168). Crossover, mutação e redesenho ficam perto do controle
  (0,145–0,159): renovar estratégias sem mudar o comprimento quase não coordena.

## Estrutura

```
emg.py            núcleo: kernels Numba
                    emg1 -> Tipo-I     (S=2, um m por agente)
                    emg2 -> Tipo-II    (S=4, um m por estratégia)
                    run_type1 / run_type2 -> wrappers que devolvem dicionários
reference.py      segunda implementação, em Python puro, escrita do texto do artigo,
                    com asserções estruturais dos operadores (verificação)
test_kernels.py   valida kernels contra a referência e contra limites conhecidos do MG
validate.py       comparação estatística entre as duas implementações
experiments.py    experimentos rep / long / ext (E1–E8)
rerun_ops.py      ablação de operadores -> results/op_ablation.csv
plots.py          figuras do artigo
make_figs.py      gera todas as figuras do artigo
rodar_teste.py    roda UM cenário com parâmetros próprios + figura + CSV
comparar_tipos.py compara Tipo-I x Tipo-II lado a lado
animar.py         GIF da evolução, um quadro por geração
uteis.py          funções compartilhadas dos três scripts acima
results/          dados: .csv (curvas e resumos) e .npz (séries por geração)
GUIA.md           guia de uso: parâmetros, receitas e como plotar
```

## Como rodar

Requisitos: Python 3.10+ com `numpy`, `numba` e `matplotlib`.

```bash
pip install -r requirements.txt
cd src            # os scripts assumem que são executados daqui
python test_kernels.py          # validação (~15 s)
python experiments.py rep       # Figs. do artigo (~1 min)
python experiments.py long      # séries de 2000 gerações (~8 min)
python experiments.py ext       # experimentos estendidos E1–E8 (~20 min)
python rerun_ops.py             # ablação de operadores (~2 min)
python make_figs.py             # figuras
```

Testes com parâmetros próprios:

```bash
python rodar_teste.py --tipo I --p 0.05 --geracoes 300
python comparar_tipos.py --p 0.05 --geracoes 300
python animar.py --tipo I --geracoes 150
```

No Windows, há um atalho que dispensa configuração de PATH:
`.\rodar.ps1 tipo1`, `.\rodar.ps1 comparar`, `.\rodar.ps1 animar`.
Detalhes em `GUIA.md`.

## Reprodutibilidade

- Todas as sementes são fixas: mesmos parâmetros produzem exatamente os mesmos
  resultados.
- 8 sementes independentes por ponto; os gráficos mostram média ± erro padrão.
- Gerador próprio (`splitmix64`) em vez do gerador do NumPy, para que os resultados
  não dependam da versão da biblioteca.
- Parâmetros do artigo: N = 101, S = 2 (Tipo-I) e 4 (Tipo-II), m ∈ {1..15},
  p ∈ {0,05; 0,10; 0,20}, T = 10⁴ passos por geração.

## Dados

`results/` tem duas categorias:

- `.csv` — curvas e resumos (MG básico, ablação, síntese dos experimentos, riqueza
  por memória);
- `.npz` — séries por geração de cada experimento (chaves: `sigma2`, `meanm`,
  `snapshots`, `ps`).

As figuras podem ser regeradas a partir desses arquivos sem reexecutar as simulações.

## Trabalho relacionado

- Challet, D.; Zhang, Y.-C. Emergence of cooperation and organization in an
  evolutionary game. *Physica A* 246 (1997).
- Li, Y.; Riolo, R.; Savit, R. Evolution in minority games. *Physica A* 276 (2000).
- Morsky, B.; Zhuang, F.; Zhou, Z. Social and individual learning in the Minority
  Game. arXiv:2307.11846 (2024).

## Autores

Julio Machado da Silveira Neto e Igor Pereira Goularte — Universidade Federal de
Pelotas (UFPel).
