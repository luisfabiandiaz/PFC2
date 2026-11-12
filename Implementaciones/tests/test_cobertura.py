"""Pruebas de ramas poco cubiertas de astcc. Ejecutar desde Implementaciones/:  python -m pytest tests/test_cobertura.py"""

import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc.cortes import (mascaras_biparticion, mejor_corte, mejor_corte_columnas,  # noqa: E402
                          mejor_corte_conjunto)
from astcc.estabilidad import (concordancia, divergencia_fuera_de_muestra,  # noqa: E402
                               frecuencia_de_soporte, hay_hoja_similar, jaccard)
from astcc.estadisticos import G2  # noqa: E402
from astcc.nulas import (ajuste_sin_interaccion_triple, ajustar_taylor, phi_constante,  # noqa: E402
                         replicas_poisson, replicas_taylor, varianza_agregada)
from astcc.sinteticos import Patron, generar  # noqa: E402


def g2_hijos(T, a, b):
    """Suma de G2 de los cuatro hijos de un corte conjunto, a mano."""
    a, b = np.asarray(a), np.asarray(b)
    return sum(float(G2(T[np.ix_(a == r, b == c)])) for r in (0, 1) for c in (0, 1)
               if np.any(a == r) and np.any(b == c))


def igual_salvo_intercambio(et, plantada):
    et, plantada = np.asarray(et), np.asarray(plantada)
    return np.array_equal(et, plantada) or np.array_equal(1 - et, plantada)


def tabla_2x2(rng, m, n, escala=400.0):
    """Tabla m x n con patrón 2 x 2 plantado fuerte y sin ruido, de modo que cada hijo
    cumple la independencia exacta; devuelve (T, filas, cols)."""
    a = (np.arange(m) >= m // 2).astype(int)
    b = (np.arange(n) >= n // 2).astype(int)
    lam = np.array([[1.0, 8.0], [8.0, 1.0]])
    efecto = np.outer(rng.uniform(1, 3, m).round(1), rng.uniform(1, 3, n).round(1))
    return escala * lam[a][:, b] * efecto, a, b


# ---------------------------------------------------------------------------
# 1. Corte de filas por 2-medias
# ---------------------------------------------------------------------------

def test_corte_filas_2medias_recupera_biparticion_plantada():
    """Con m = 20 > exacto_hasta se usa 2-medias y recupera la bipartición plantada."""
    rng = np.random.default_rng(10)
    m, n = 20, 6
    plantada = np.zeros(m, int)
    plantada[[1, 4, 5, 8, 9, 12, 15, 16, 17, 19]] = 1
    perfiles = np.array([[1, 1, 1, 8, 8, 8], [8, 8, 8, 1, 1, 1]], float)
    T = rng.poisson(100 * perfiles[plantada]).astype(float)
    g, (A, Bl) = mejor_corte(T[None], "fil", rng, exacto_hasta=14)
    assert igual_salvo_intercambio(A[0], plantada)
    assert np.all(Bl == 0)
    colapsada = np.stack([T[plantada == 0].sum(0), T[plantada == 1].sum(0)])
    assert np.isclose(g[0], G2(colapsada))


# ---------------------------------------------------------------------------
# 2. Cortes de columnas ordenados
# ---------------------------------------------------------------------------

def test_corte_columnas_ordenado():
    """Las máscaras ordenadas son prefijo de 0 y sufijo de 1, y la ganancia es la de fuerza bruta."""
    rng = np.random.default_rng(11)
    T = rng.poisson(30 * rng.lognormal(0, 1, (6, 5, 7))).astype(float) + 1
    g_o, M = mejor_corte_columnas(T, ordenado=True)
    g_l, _ = mejor_corte_columnas(T)
    assert np.all(g_o <= g_l + 1e-9)
    for m in M:
        s = int(m.sum())
        assert 0 < s < 7
        assert np.array_equal(m, np.r_[np.zeros(7 - s), np.ones(s)])
    for t, g in zip(T, g_o):
        brutos = [G2(np.stack([t[:, :c].sum(1), t[:, c:].sum(1)], 1)) for c in range(1, 7)]
        assert np.isclose(g, max(brutos))


# ---------------------------------------------------------------------------
# 3. Corte conjunto
# ---------------------------------------------------------------------------

def test_corte_conjunto_alternante_recupera_patron():
    """Ramas alternantes (n > 6; y m > 14 con filas fijas): ganancia = G2 - suma de hijos."""
    rng = np.random.default_rng(12)
    for m, n in [(10, 8), (16, 8)]:
        T, a, b = tabla_2x2(rng, m, n)
        g, (A, Bl) = mejor_corte(T[None], "conj", rng)
        assert g[0] >= 0
        assert np.isclose(g[0], G2(T) - g2_hijos(T, A[0], Bl[0]))
        # el patrón se recupera: los cuatro hijos cumplen la independencia exacta, así que
        # la ganancia es todo el G2 (la bipartición que lo logra no tiene por qué ser única:
        # dado b, cualquier corte de filas deja hijos independientes en esta tabla sin ruido)
        assert g2_hijos(T, A[0], Bl[0]) < 1e-6 * G2(T)
        assert np.isclose(g[0], G2(T))


def test_corte_conjunto_exacto_igual_a_fuerza_bruta():
    """En 5 x 4 (rama exacta) la ganancia es el máximo sobre todos los pares de biparticiones."""
    rng = np.random.default_rng(13)
    for _ in range(3):
        T = rng.poisson(20 * rng.lognormal(0, 1, (5, 4))).astype(float) + 1
        g, (A, Bl) = mejor_corte(T[None], "conj", rng)
        brutos = [G2(T) - g2_hijos(T, a, b)
                  for a, b in product(mascaras_biparticion(5), mascaras_biparticion(4))]
        assert np.isclose(g[0], max(brutos))
        assert np.isclose(g[0], G2(T) - g2_hijos(T, A[0], Bl[0]))


def test_corte_conjunto_acepta_semillas_a0_b0():
    """mejor_corte_conjunto en la rama alternante nunca devuelve una ganancia negativa."""
    rng = np.random.default_rng(14)
    T = np.stack([tabla_2x2(rng, 8, 8)[0] for _ in range(3)])
    a0 = np.zeros((3, 8), int)
    a0[:, 4:] = 1
    b0 = np.zeros((3, 8), int)
    b0[:, ::2] = 1
    g, a, b = mejor_corte_conjunto(T, a0, b0)
    assert np.all(g >= -1e-9)
    for t, gi, ai, bi in zip(T, g, a, b):
        assert np.isclose(gi, G2(t) - g2_hijos(t, ai, bi))


# ---------------------------------------------------------------------------
# 4. Coherencia por lotes
# ---------------------------------------------------------------------------

def test_lote_igual_a_tabla_a_tabla():
    """mejor_corte sobre un lote coincide con el cálculo tabla a tabla ('col' y 'fil')."""
    rng = np.random.default_rng(15)
    T = rng.poisson(25 * rng.lognormal(0, 1, (7, 6, 5))).astype(float) + 1
    for tipo in ("col", "fil"):
        g, (A, Bl) = mejor_corte(T, tipo, rng)
        for i in range(len(T)):
            gi, (Ai, Bi) = mejor_corte(T[i:i + 1], tipo, rng)
            assert np.isclose(g[i], gi[0])
            assert np.array_equal(A[i], Ai[0]) and np.array_equal(Bl[i], Bi[0])


def test_troceo_no_cambia_el_resultado():
    """Un max_elementos pequeño fuerza varios trozos y da lo mismo que sin trocear."""
    rng = np.random.default_rng(16)
    T = rng.poisson(25 * rng.lognormal(0, 1, (9, 6, 5))).astype(float) + 1
    for tipo in ("col", "fil", "conj"):
        g0, (A0, B0) = mejor_corte(T, tipo, rng)
        g1, (A1, B1) = mejor_corte(T, tipo, rng, max_elementos=800)
        assert np.allclose(g0, g1)
        assert np.array_equal(A0, A1) and np.array_equal(B0, B1)


def test_tipo_de_corte_desconocido():
    """Un tipo de corte inválido lanza ValueError."""
    import pytest
    with pytest.raises(ValueError):
        mejor_corte(np.ones((1, 3, 3)), "diag", np.random.default_rng(0))


# ---------------------------------------------------------------------------
# 5. Máscaras
# ---------------------------------------------------------------------------

def test_mascaras_biparticion_propiedades():
    """Forma, filas únicas, columna 0 en 0 y ninguna fila constante, para varios k."""
    for k in (2, 3, 4, 7):
        M = mascaras_biparticion(k)
        assert M.shape == (2 ** (k - 1) - 1, k)
        assert len({tuple(f) for f in M}) == len(M)
        assert np.all(M[:, 0] == 0)
        assert np.all(M.sum(1) > 0)


# ---------------------------------------------------------------------------
# 6. Sintéticos
# ---------------------------------------------------------------------------

def test_sinteticos_sin_patrones_bloques_base_rango_1():
    """Sin patrones, mu es de rango 1 en cada bloque base (rho, tau): G2 ~ 0."""
    s = generar(np.random.default_rng(20), m=40, n=12, KR=4, KT=3)
    assert s.activas == []
    for g, h in product(range(4), range(3)):
        bloque = s.mu[np.ix_(s.rho == g, s.tau == h)]
        assert abs(G2(bloque)) < 1e-6 * bloque.sum()


def test_sinteticos_forma_y_agregado_cerca_de_mu():
    """X es (D, m, n), entera y no negativa; con escala grande X.sum(0) ~ mu por bloque."""
    s = generar(np.random.default_rng(21), m=24, n=9, D=6, escala=1e6, k=np.inf,
                var_dia=0.0, var_dia_fila=0.0, var_dia_col=0.0)
    assert s.X.shape == (6, 24, 9)
    assert np.all(s.X >= 0) and np.all(s.X == np.round(s.X))
    O = s.X.sum(0)
    for g, h in product(range(4), range(3)):
        idx = np.ix_(s.rho == g, s.tau == h)
        assert abs(O[idx].sum() - s.mu[idx].sum()) / s.mu[idx].sum() < 0.01


def test_sinteticos_patron_activo_dentro_del_bloque_y_rompe_rango_1():
    """Las celdas activas caen en el bloque (g, h) y mu allí deja de ser de rango 1."""
    pt = Patron(g=1, h=2, frac_filas=0.5, frac_cols=0.5, factor=6.0)
    s = generar(np.random.default_rng(22), m=40, n=12, patrones=[pt])
    assert len(s.activas) == 1
    fi, fj = s.activas[0]
    assert np.all(s.rho[fi] == 1) and np.all(s.tau[fj] == 2)
    bloque = s.mu[np.ix_(s.rho == 1, s.tau == 2)]
    assert G2(bloque) > 1e-3 * bloque.sum()


def test_sinteticos_modificar_se_anade_a_activas():
    """El gancho `modificar` puede alterar mu y su descripción se añade a `activas`."""
    def mod(mu, rho, tau, rng):
        mu[0, 0] *= 10
        return ["marca"]
    s = generar(np.random.default_rng(23), m=10, n=6, modificar=mod)
    assert s.activas == ["marca"]


# ---------------------------------------------------------------------------
# 7. Nulas
# ---------------------------------------------------------------------------

def test_ipf_triple_conserva_marginales():
    """El ajuste [DZ][DT][ZT] conserva las tres marginales bidimensionales."""
    X = np.random.default_rng(30).gamma(2.0, 10.0, (5, 6, 4))
    E = ajuste_sin_interaccion_triple(X)
    for eje in (0, 1, 2):
        assert np.allclose(E.sum(eje), X.sum(eje), rtol=1e-6)


def test_ajustar_taylor_recupera_parametros():
    """Con var = a mu^b exacta se recuperan (a, b)."""
    mu = np.linspace(6, 500, 60)
    a, b = ajustar_taylor(mu, 2.5 * mu ** 1.3)
    assert np.isclose(a, 2.5) and np.isclose(b, 1.3)


def test_phi_constante_mediana_en_activas():
    """phi es la mediana de var/media en las celdas con media >= media_min."""
    media = np.array([1.0, 10.0, 20.0, 40.0])
    var = np.array([100.0, 20.0, 80.0, 40.0])      # razones 100, 2, 4, 1; la primera no es activa
    assert phi_constante(media, var) == 2.0


def test_varianza_agregada():
    """V_D(mu) >= mu, y con a = b = 1 es igual a mu (también con mu = 0)."""
    mu = np.array([0.0, 0.5, 3.0, 50.0, 1000.0])
    for a, b, D in [(0.3, 0.8, 10), (2.0, 1.5, 12), (5.0, 1.0, 7)]:
        assert np.all(varianza_agregada(mu, a, b, D) >= mu)
    assert np.allclose(varianza_agregada(mu, 1.0, 1.0, 12), mu)


def test_replicas_poisson_conserva_marginales():
    """Todas las réplicas de Patefield tienen las marginales de T."""
    rng = np.random.default_rng(31)
    T = rng.poisson(15, (5, 4)).astype(float)
    Rep = replicas_poisson(T, 50, rng)
    assert Rep.shape == (50, 5, 4)
    assert np.all(Rep.sum(2) == T.sum(1)) and np.all(Rep.sum(1) == T.sum(0))
    assert np.all(replicas_poisson(np.zeros((3, 3)), 4, rng) == 0)


def test_replicas_taylor_poisson_y_sobredispersion():
    """Con a = b = 1 media y varianza ~ E (Poisson); con a > 1 la varianza es mayor."""
    rng = np.random.default_rng(32)
    T = rng.poisson(30, (3, 3)).astype(float)
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    R = 20000
    P = replicas_taylor(T, R, 1.0, 1.0, 12, rng)
    assert np.allclose(P.mean(0), E, rtol=0.05)
    assert np.allclose(P.var(0), E, rtol=0.1)
    S = replicas_taylor(T, R, 3.0, 1.5, 4, rng)
    assert np.allclose(S.mean(0), E, rtol=0.1)
    assert np.all(S.var(0) > 1.5 * E)


# ---------------------------------------------------------------------------
# 8. Estabilidad
# ---------------------------------------------------------------------------

def hoja(filas, cols):
    return SimpleNamespace(filas=np.array(filas), cols=np.array(cols))


def test_jaccard_casos_conocidos():
    """Idénticas = 1, disjuntas = 0, y un caso parcial calculado a mano."""
    assert jaccard([0, 1], [0, 1, 2], [0, 1], [0, 1, 2]) == 1.0
    assert jaccard([0, 1], [0, 1], [2, 3], [0, 1]) == 0.0
    assert jaccard([0, 1], [0, 1], [0, 1], [1, 2]) == 2 / 6   # inter 2, unión 4 + 4 - 2


def test_frecuencia_de_soporte_y_concordancia():
    """Frecuencia 1 si todas las ejecuciones repiten las hojas; 0.5 si la mitad las pierde."""
    ref = [hoja([0, 1], [0, 1]), hoja([2, 3], [2, 3])]
    iguales = [ref, ref, ref]
    assert np.all(frecuencia_de_soporte(ref, iguales) == 1.0)
    mitad = [ref, [hoja([5, 6], [5, 6])]]
    assert np.allclose(frecuencia_de_soporte(ref, mitad), [0.5, 0.5])
    assert concordancia(ref, ref) == 1.0
    assert concordancia(ref, [hoja([5], [5])]) == 0.0
    assert hay_hoja_similar(ref[0], [hoja([0, 1], [0, 1, 2])], kappa=0.5)     # 4/6
    assert not hay_hoja_similar(ref[0], [hoja([0, 1], [0, 1, 2])], kappa=0.7)


def test_divergencia_fuera_de_muestra():
    """Sin suavizado y con predicción proporcional a la prueba la divergencia es 0; si no, > 0."""
    O = np.array([[10.0, 0.0, 5.0], [3.0, 8.0, 2.0]])
    assert abs(divergencia_fuera_de_muestra(O, 2 * O, suavizado=0.0)) < 1e-9
    d = divergencia_fuera_de_muestra(O, np.ones((2, 3)))
    assert np.isfinite(d) and d > 0


def test_enumeracion_de_cortes_coincide_con_G2_directo_y_no_depende_de_los_hilos():
    """El núcleo optimizado (fórmula sin esperados, trozos, hilos y búferes) da la
    misma ganancia y la misma máscara que evaluar G^2 de cada tabla colapsada."""
    import astcc.cortes as cortes
    from astcc.estadisticos import G2 as G2_directo

    rng = np.random.default_rng(31)
    T = rng.poisson(rng.uniform(0.2, 40, size=(1, 5, 7)), size=(60, 5, 7)).astype(float)
    T[0] = 0.0                                                # tabla vacía
    M = cortes.mascaras_biparticion(7)
    uno = np.einsum("bmn,kn->bkm", T, M)
    directo = G2_directo(np.stack([T.sum(-1)[:, None, :] - uno, uno], axis=-1))
    hilos = cortes.HILOS
    try:
        for cortes.HILOS in (1, 4):
            g, mascara = cortes.mejor_corte_columnas(T, elementos_por_trozo=2_000)
            assert np.allclose(g, directo.max(1), rtol=1e-10, atol=1e-8)
            assert np.array_equal(mascara[1:], M[directo[1:].argmax(1)])
    finally:
        cortes.HILOS = hilos
