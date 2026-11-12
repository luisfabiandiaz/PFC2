"""Calibración de la nula de bootstrap de días con datos que cumplen H0 por
construcción (Poisson independiente), es decir, generados por un mecanismo distinto
del propio bootstrap. Ejecutar desde Implementaciones/:  python -m pytest tests"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc.cortes import mejor_corte  # noqa: E402
from astcc.estadisticos import G2, esperados, pvalor_simulado  # noqa: E402
from astcc.nulas import replicas_semirreales  # noqa: E402

ALFA, B = 0.05, 199


def media_independiente(m, n, escala, rango_filas=30.0, rango_cols=5.0):
    mu = np.outer(np.geomspace(1, rango_filas, m), np.geomspace(1, rango_cols, n))
    return mu / mu.mean() * escala


def tasa_de_rechazo(rng, mu, D, reps, estadistico=G2):
    rechazos = 0
    for _ in range(reps):
        X = rng.poisson(mu, size=(D,) + mu.shape).astype(float)
        nulos = estadistico(replicas_semirreales(X, B, rng, inflar=True))
        rechazos += pvalor_simulado(float(estadistico(X.sum(0)[None])[0]), nulos) <= ALFA
    return rechazos / reps


def test_bloque_con_celdas_escasas_no_es_liberal():
    """Con la mitad de las celdas por debajo de media_min, fijar F = 1 sin ruido
    dejaba esas celdas casi deterministas y el error de tipo I subía a ~0.8."""
    rng = np.random.default_rng(20)
    mu = media_independiente(10, 6, 1.0)
    assert 0.3 < np.mean(mu < 0.5) < 0.7
    assert tasa_de_rechazo(rng, mu, D=12, reps=200) <= 0.10


def test_celdas_escasas_tienen_al_menos_varianza_de_poisson():
    rng = np.random.default_rng(21)
    mu = media_independiente(8, 4, 0.6)
    X = rng.poisson(mu, size=(12,) + mu.shape).astype(float)
    escasa = X.mean(0) < 0.5
    assert escasa.sum() >= 5
    R = replicas_semirreales(X, 20000, rng, inflar=True)
    E = esperados(X.sum(0))
    con_masa = escasa & (E > 0.5)
    assert np.all(R.var(0)[con_masa] >= 0.8 * E[con_masa])
    assert np.allclose(R.mean(0)[escasa], E[escasa], rtol=0.05, atol=0.05)


def test_la_correccion_de_muestra_pequena_no_sesga_la_media():
    """Inflar las fluctuaciones y recortarlas en 0 subía la media un 2-3 % en celdas
    con pocos viajes; tras reescalar, la media de las réplicas vuelve a ser E^B."""
    rng = np.random.default_rng(22)
    mu = media_independiente(7, 3, 2.0, rango_filas=6.0, rango_cols=2.0)   # ~1 a 5 viajes/celda/día
    X = rng.poisson(mu, size=(9,) + mu.shape).astype(float)
    assert np.all(X.mean(0) >= 0.5)                                     # todas activas
    assert np.any(X == 0)                                               # hay recorte
    R = replicas_semirreales(X, 40000, rng, inflar=True)
    assert np.allclose(R.mean(0), esperados(X.sum(0)), rtol=0.01)


def test_nivel_en_bloques_pequenos():
    """Con menos grados de libertad que días, el nivel real queda cerca del nominal."""
    rng = np.random.default_rng(23)
    tasa = tasa_de_rechazo(rng, media_independiente(2, 6, 100.0), D=12, reps=400)
    assert 0.02 <= tasa <= 0.10


def test_bloques_grandes_son_conservadores():
    """Limitación conocida: las réplicas de D días varían en a lo sumo D - 1
    direcciones, así que con (m - 1)(n - 1) >> D la cola nula de G^2 es más pesada
    que la real y la prueba pierde potencia, pero no nivel."""
    rng = np.random.default_rng(24)
    assert tasa_de_rechazo(rng, media_independiente(10, 6, 100.0), D=12, reps=200) <= ALFA


def test_minP_con_bootstrap_no_es_liberal():
    """El ajuste min-P (repetir la búsqueda del mejor corte en cada réplica) también
    mantiene el nivel con la nula de bootstrap, no solo con la de Poisson."""
    rng = np.random.default_rng(25)
    mu = media_independiente(4, 4, 100.0)
    for tipo in ("fil", "col"):
        tasa = tasa_de_rechazo(rng, mu, D=12, reps=200,
                               estadistico=lambda T, t=tipo: mejor_corte(T, t, rng)[0])
        assert tasa <= 0.10
