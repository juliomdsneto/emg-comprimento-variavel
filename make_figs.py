"""Gera todas as figuras disponiveis (executar depois dos experimentos)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plots as P

P.fig_series("type1", 500, "fig_type1_series")
P.fig_series("type2", 500, "fig_type2_series")
P.fig_series2()
P.fig_mdist("type1", "fig_type1_mdist", gc=2500, title="Tipo-I")
P.fig_mdist("type2", "fig_type2_mdist", gc=2500, title="Tipo-II")
P.fig_mg_so()
P.fig_comparacao()
P.fig_contraste()
for fn in (P.fig_p_sweep, P.fig_mmax_N, P.fig_ops, P.fig_riqueza):
    try:
        fn()
    except Exception as e:
        print("  (pulado)", fn.__name__, "->", type(e).__name__, e)
print("ok")
