"""Pruebas de la biblioteca astcc. Ejecutar desde Implementaciones/:  python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc.cortes import (corte_filas_2medias, mascaras_biparticion, mejor_corte_columnas,  # noqa: E402
                          mejor_corte_filas, mejor_corte_filas_exacto)
from astcc.estadisticos import G2, esperados, info_mutua, pvalor_simulado  # noqa: E402
from astcc.nulas import (replicas_poisson, replicas_semirreales,  # noqa: E402
                         replicas_taylor, varianza_agregada)


def tabla(rng, m, n, escala=50.0):
    return rng.poisson(escala * rng.lognormal(0, 1, (m, n))).astype(float) + 1


def test_mascaras():
    M = mascaras_biparticion(5)
    assert M.shape == (15, 5)
    assert np.all(M[:, 0] == 0)
    assert len({tuple(f) for f in M}) == 15
    assert np.all((M.sum(1) > 0) & (M.sum(1) < 5))


def test_G2_lote_y_info_mutua():
    rng = np.random.default_rng(0)
    T = np.stack([tabla(rng, 4, 6) for _ in range(3)])
    assert np.allclose(G2(T), [G2(t) for t in T])
    assert np.allclose(info_mutua(T), G2(T) / (2 * T.sum((1, 2))))
    indep = np.outer([1, 2, 3], [4, 5]) * 10.0
    assert abs(G2(indep)) < 1e-9


def test_corte_columnas_es_el_maximo():
    rng = np.random.default_rng(1)
    T = tabla(rng, 5, 6)[None]
    g, mascara = mejor_corte_columnas(T)
    todas = [G2(np.stack([T[0][:, m == 0].sum(1), T[0][:, m == 1].sum(1)], 1))
             for m in mascaras_biparticion(6)]
    assert np.isclose(g[0], max(todas))


def test_heuristica_filas_nunca_supera_al_optimo():
    """El 2-medias KL es una cota inferior del óptimo y lo alcanza casi siempre
    (con R_ini = 10, en torno al 92 %; por eso hasta 14 filas se enumera)."""
    rng = np.random.default_rng(2)
    T = np.stack([tabla(rng, 8, 5) for _ in range(40)])
    g_heur, _ = corte_filas_2medias(T, rng, R_ini=10)
    g_exac, _ = mejor_corte_filas_exacto(T)
    assert np.all(g_heur <= g_exac + 1e-8)
    assert np.mean(np.isclose(g_heur, g_exac)) >= 0.85
    g, _ = mejor_corte_filas(T, rng)
    assert np.allclose(g, g_exac)


def test_pvalor():
    assert pvalor_simulado(5.0, np.array([1.0, 2.0, 5.0, 7.0])) == 3 / 5
    assert pvalor_simulado(100.0, np.zeros(99)) == 1 / 100


def test_patefield_conserva_marginales():
    rng = np.random.default_rng(3)
    T = tabla(rng, 4, 5)
    R = replicas_poisson(T, 50, rng)
    assert np.allclose(R.sum(2), T.sum(1)) and np.allclose(R.sum(1), T.sum(0))


def test_taylor_media_y_varianza():
    rng = np.random.default_rng(4)
    T = tabla(rng, 3, 4, escala=200)
    a, b, D = 0.8, 1.5, 12
    R = replicas_taylor(T, 40000, a, b, D, rng)
    E = esperados(T)
    assert np.allclose(R.mean(0), E, rtol=0.02)
    assert np.allclose(R.var(0), varianza_agregada(E, a, b, D), rtol=0.06)
    # a = b = 1 es Poisson
    assert np.allclose(varianza_agregada(E, 1, 1, D), E)


def test_semirreal_conserva_independencia_con_efecto_de_dia():
    """Si los días solo difieren por un factor común, las réplicas son exactamente
    proporcionales a la media de independencia (G2 = 0 salvo el redondeo)."""
    rng = np.random.default_rng(5)
    base = np.outer(rng.uniform(50, 500, 6), rng.uniform(1, 3, 4))   # independiente
    gamma = rng.uniform(0.6, 1.4, 12)
    Xb = gamma[:, None, None] * base[None]
    R = replicas_semirreales(Xb, 200, rng)
    assert np.max(info_mutua(R)) < 1e-5


def test_semirreal_sin_sesgo_en_celdas_escasas():
    """Con pocos viajes por celda y día, la media de las réplicas diarias agregadas
    debe seguir siendo la de independencia (el redondeo no puede sesgarla)."""
    rng = np.random.default_rng(6)
    base = np.outer(rng.uniform(0.2, 3, 7), rng.uniform(0.5, 2, 3))   # ~1 viaje/celda/día
    Xb = rng.poisson(np.broadcast_to(base, (12, 7, 3))).astype(float)
    R = replicas_semirreales(Xb, 20000, rng, diarias=True).sum(1)
    assert np.allclose(R.mean(0), esperados(Xb.sum(0)), rtol=0.03, atol=0.05)


def test_suma_G2_hijos_explicita():
    from astcc.cortes import _suma_G2_hijos
    rng = np.random.default_rng(10)
    T = tabla(rng, 6, 5)
    a, b = np.array([0, 1, 0, 1, 1, 0]), np.array([1, 0, 0, 1, 0])
    explicita = sum(G2(T[np.ix_(a == r, b == c)]) for r in (0, 1) for c in (0, 1))
    assert np.isclose(_suma_G2_hijos(T[None], a[None], b[None])[0], explicita)


def test_corte_conjunto_domina_a_los_unidimensionales_y_es_optimo():
    """El modelo conjunto contiene a los de filas y columnas (Cap_3, 'El corte conjunto
    como recurso'), así que su ganancia es al menos la del mejor corte 1D. Se compara
    además con la enumeración de todos los pares (corte de filas, corte de columnas)."""
    from astcc.cortes import _suma_G2_hijos, mejor_corte
    rng = np.random.default_rng(11)
    T = np.stack([tabla(rng, 6, 5) for _ in range(30)])
    g_conj, (a, b) = mejor_corte(T, "conj", rng)
    g_fil, _ = mejor_corte(T, "fil", rng)
    g_col, _ = mejor_corte(T, "col", rng)
    assert np.all(g_conj >= np.maximum(g_fil, g_col) - 1e-8)
    Mf, Mc = mascaras_biparticion(6).astype(int), mascaras_biparticion(5).astype(int)
    optimos = []
    for t in T:
        pares = [(x, y) for x in Mf for y in Mc]
        A = np.array([p[0] for p in pares]); Bm = np.array([p[1] for p in pares])
        optimos.append(G2(t) - _suma_G2_hijos(np.repeat(t[None], len(pares), 0), A, Bm).min())
    assert np.all(g_conj <= np.array(optimos) + 1e-8)
    assert np.allclose(g_conj, optimos)          # búsqueda exacta con |J| <= 6
