"""Pruebas de la segunda etapa de eventos (astcc.eventos)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc import eventos as ev  # noqa: E402

M, N_INT = 6, 4                                   # regiones e intervalos del cubo de prueba


@pytest.fixture(scope="module")
def cubo():
    """Veinte semanas de conteos de Poisson con un perfil fijo por celda."""
    rng = np.random.default_rng(0)
    fechas = pd.date_range("2017-01-02", periods=140)
    media = rng.uniform(200, 800, size=(M, N_INT))
    return rng.poisson(media, size=(len(fechas), M, N_INT)), fechas, media


def cadena(m):
    """Regiones en fila: cada una colinda con la siguiente."""
    v = np.zeros((m, m), bool)
    i = np.arange(m - 1)
    v[i, i + 1] = v[i + 1, i] = True
    return v


# ----------------------------------------------------------------------------
# Puntuación por celda y por unidad
# ----------------------------------------------------------------------------

def test_sin_eventos_los_z_son_moderados(cubo):
    Y, fechas, _ = cubo
    Z = ev.puntuar_celdas(Y, fechas)
    assert Z.shape == Y.shape
    assert abs(np.median(Z)) < 0.1
    assert np.abs(Z).max() < 9


def test_un_pico_plantado_es_la_celda_mas_atipica(cubo):
    Y, fechas, media = cubo
    Y = Y.copy()
    Y[70, 2, 1] += int(4 * media[2, 1])
    Z = ev.puntuar_celdas(Y, fechas)
    assert np.unravel_index(np.abs(Z).argmax(), Z.shape) == (70, 2, 1)
    assert Z[70, 2, 1] > 15


def test_el_nivel_del_dia_no_cambia_la_puntuacion(cubo):
    """Un día con la mitad de llegadas en todas las celdas reparte igual: mismo z de cuota."""
    Y, fechas, _ = cubo
    Y = Y.astype(float)
    Y2 = Y.copy()
    Y2[70] *= 0.5
    Z, Z2 = ev.puntuar_celdas(Y, fechas), ev.puntuar_celdas(Y2, fechas)
    otros = np.arange(len(fechas)) != 70
    assert np.allclose(Z[otros], Z2[otros])
    # en el propio día solo cambia el suelo de Poisson, que depende del total
    assert np.abs(Z2[70]).max() < 9


def test_solo_cuentan_los_dias_del_mismo_dia_de_la_semana(cubo):
    Y, fechas, _ = cubo
    Y2 = Y.copy()
    martes = fechas.dayofweek == 1
    Y2[martes] = Y2[martes][:, ::-1]                       # se altera el reparto de todos los martes
    lunes = fechas.dayofweek == 0
    assert np.array_equal(ev.puntuar_celdas(Y, fechas)[lunes].shape, ev.puntuar_celdas(Y2, fechas)[lunes].shape)
    N, N2 = Y.sum((1, 2)), Y2.sum((1, 2))
    assert np.array_equal(N, N2)
    assert np.allclose(ev.puntuar_celdas(Y, fechas)[lunes], ev.puntuar_celdas(Y2, fechas)[lunes])


def test_la_ventana_limita_la_linea_base(cubo):
    Y, fechas, _ = cubo
    Y2 = Y.copy()
    Y2[:21] = Y2[:21][:, :, ::-1]                          # cambian las tres primeras semanas
    d = 100                                                # a más de 28 días de ellas
    assert np.allclose(ev.puntuar_celdas(Y, fechas, ventana=28)[d], ev.puntuar_celdas(Y2, fechas, ventana=28)[d])
    assert not np.allclose(ev.puntuar_celdas(Y, fechas, ventana=120)[d], ev.puntuar_celdas(Y2, fechas, ventana=120)[d])


def test_puntuar_con_celdas_coincide_con_puntuar_celdas(cubo):
    Y, fechas, _ = cubo
    us = [(np.array([i]), np.array([j])) for i in range(M) for j in range(N_INT)]
    assert np.allclose(ev.puntuar(Y, fechas, us).reshape(Y.shape), ev.puntuar_celdas(Y, fechas))


def test_residuo_ignora_un_cambio_de_toda_la_unidad(cubo):
    Y, fechas, _ = cubo
    us = [(np.arange(0, 3), np.arange(N_INT)), (np.arange(3, M), np.arange(N_INT))]
    Y2 = Y.astype(float).copy()
    Y2[70, :3] *= 3                                        # la primera unidad triplica sus llegadas
    R, R2 = ev.residuo(Y.astype(float), fechas, us), ev.residuo(Y2, fechas, us)
    assert np.abs(R2[70]).max() < 9                        # dentro de la unidad nada destaca
    assert np.abs(ev.puntuar(Y2, fechas, us)[70]).max() > 20   # la cuota de la unidad sí
    assert np.allclose(R[:60], R2[:60])


def test_amplitud_distingue_dia_difuso_de_evento_local(cubo):
    Y, fechas, media = cubo
    us = [(np.array([i]), np.arange(N_INT)) for i in range(M)]
    Y2 = Y.copy()
    Y2[70] = Y2[70][::-1]                                  # día difuso: cambia el reparto entre todas las regiones
    # evento local: una celda. Es moderado porque en un cubo de 24 celdas un pico grande
    # cambia el total del día y, con él, la cuota de todas las demás regiones.
    Y2[90, 2, 1] += int(0.5 * media[2, 1])
    a = ev.amplitud(Y2, fechas, us)
    assert a[70] >= 0.5
    assert a[90] <= 2 / M
    assert np.median(a) == 0


# ----------------------------------------------------------------------------
# Ganancia del corte de un día
# ----------------------------------------------------------------------------

def test_ganancia_excluye_unidades_de_una_celda(cubo):
    Y, fechas, _ = cubo
    us = [(np.array([0]), np.array([0])), (np.arange(1, 4), np.arange(N_INT)), (np.arange(4, M), np.arange(2))]
    Z, validas = ev.ganancia(Y, fechas, us)
    assert validas == [1, 2]
    assert Z.shape == (len(fechas), 2)


def test_ganancia_detecta_un_cambio_de_reparto_dentro_de_la_unidad(cubo):
    Y, fechas, _ = cubo
    us = [(np.arange(0, 3), np.arange(N_INT)), (np.arange(3, M), np.arange(N_INT))]
    Y2 = Y.copy()
    Y2[70, :3] = Y2[70, :3][:, ::-1]                       # el día 70 la unidad 0 invierte su perfil horario
    Z, _ = ev.ganancia(Y2, fechas, us)
    assert Z[:, 0].argmax() == 70
    assert Z[70, 0] > 5
    assert abs(Z[70, 1]) < 5                               # la otra unidad no cambia


# ----------------------------------------------------------------------------
# Esperados, sobredispersión y simulación
# ----------------------------------------------------------------------------

def test_esperados_recupera_la_media(cubo):
    Y, fechas, media = cubo
    E = ev.esperados(Y, fechas)
    assert np.abs(E.mean(0) / media - 1).max() < 0.03
    assert np.allclose(E.sum((1, 2)) / Y.sum((1, 2)), 1, atol=0.02)


@pytest.mark.parametrize("phi", [1.0, 4.0])
def test_dispersion_y_simular_son_coherentes(phi):
    rng = np.random.default_rng(1)
    E = np.full((400, 3, 2), 300.0)
    Y = ev.simular(E, np.full((3, 2), phi), rng)
    assert abs(Y.mean() / 300 - 1) < 0.02
    assert abs(Y.var(0).mean() / 300 / phi - 1) < 0.15
    est = ev.dispersion(Y, E, np.ones(400, bool))
    assert est.shape == (3, 2)
    assert np.abs(est / phi - 1).max() < 0.35


def test_dispersion_no_baja_de_poisson():
    E = np.full((50, 2, 2), 100.0)
    assert np.array_equal(ev.dispersion(E.copy(), E, np.ones(50, bool)), np.ones((2, 2)))


# ----------------------------------------------------------------------------
# Umbral
# ----------------------------------------------------------------------------

def test_umbral_fdr_con_ruido_puro_casi_no_marca():
    z = np.random.default_rng(2).normal(size=100_000)
    t, s0 = ev.umbral_fdr(z, q=0.01)
    assert abs(s0 - 1) < 0.02
    assert (np.abs(z) >= t).sum() <= 5


def test_umbral_fdr_baja_cuando_hay_señal():
    rng = np.random.default_rng(3)
    z = rng.normal(size=100_000)
    t0, _ = ev.umbral_fdr(z, q=0.01)
    z[:500] += 8
    t1, _ = ev.umbral_fdr(z, q=0.01)
    assert t1 < t0
    assert (z[:500] >= t1).mean() > 0.95
    t_cola, _ = ev.umbral_fdr(z, q=0.01, una_cola=True)
    assert t_cola <= t1


# ----------------------------------------------------------------------------
# Agrupación en eventos
# ----------------------------------------------------------------------------

def marcar(*celdas, forma=(4, 5, 4)):
    Mk = np.zeros(forma, bool)
    for c in celdas:
        Mk[c] = True
    return Mk


FECHAS4 = pd.date_range("2017-03-01", periods=4)


def n_eventos(Mk, signo=None, fechas=FECHAS4):
    signo = np.ones(Mk.shape) if signo is None else signo
    nodos, etiq = ev.componentes(Mk, signo, fechas, cadena(Mk.shape[1]))
    return len(np.unique(etiq)), nodos, etiq


def test_componentes_sin_celdas_marcadas():
    n, nodos, etiq = n_eventos(marcar())
    assert n == 0 and len(nodos) == 0 and len(etiq) == 0


@pytest.mark.parametrize("a, b, juntas", [
    ((0, 1, 1), (0, 2, 1), True),      # regiones colindantes, mismo día e intervalo
    ((0, 1, 1), (0, 3, 1), False),     # regiones no colindantes
    ((0, 1, 1), (0, 1, 2), True),      # intervalos contiguos
    ((0, 1, 1), (0, 1, 3), False),     # intervalos separados
    ((0, 1, 1), (1, 1, 1), True),      # misma celda en días consecutivos
    ((0, 1, 1), (2, 1, 1), False),     # con un día en medio
    ((0, 1, 3), (1, 1, 0), True),      # medianoche: último intervalo y primero del día siguiente
    ((0, 1, 1), (0, 2, 2), False),     # en diagonal (otra región y otro intervalo)
])
def test_componentes_reglas_de_vecindad(a, b, juntas):
    n, _, _ = n_eventos(marcar(a, b))
    assert n == (1 if juntas else 2)


def test_componentes_no_une_signos_opuestos():
    Mk = marcar((0, 1, 1), (0, 2, 1))
    signo = np.ones(Mk.shape)
    signo[0, 2, 1] = -1
    assert n_eventos(Mk, signo)[0] == 2


def test_componentes_no_une_dias_no_consecutivos_del_calendario():
    """Dos filas seguidas del cubo solo se unen si sus fechas son consecutivas."""
    fechas = pd.DatetimeIndex(["2017-03-02", "2017-03-07", "2017-03-08", "2017-03-09"])
    assert n_eventos(marcar((0, 1, 1), (1, 1, 1)), fechas=fechas)[0] == 2
    assert n_eventos(marcar((1, 1, 1), (2, 1, 1)), fechas=fechas)[0] == 1


def test_componentes_encadena_una_huella_de_varios_dias():
    Mk = marcar((0, 0, 1), (0, 1, 1), (0, 1, 2), (1, 1, 2), (2, 1, 2), (3, 4, 0))
    n, nodos, etiq = n_eventos(Mk)
    assert n == 2
    assert np.bincount(etiq).tolist() in ([5, 1], [1, 5])
    assert len(nodos) == 6


def test_picos_da_el_maximo_de_cada_evento():
    Z = np.zeros((4, 5, 4))
    Z[0, 1, 1], Z[0, 2, 1], Z[3, 4, 0] = 6.0, 9.0, -7.0
    p, nodos, etiq = ev.picos(Z, np.abs(Z) >= 5, FECHAS4, cadena(5))
    assert sorted(p.tolist()) == [7.0, 9.0]
    assert len(nodos) == 3 and len(etiq) == 3


def test_picos_nulos_es_reproducible_y_raro_por_encima_del_umbral(cubo):
    Y, fechas, _ = cubo
    E = ev.esperados(Y, fechas)
    phi = ev.dispersion(Y, E, np.ones(len(fechas), bool))
    a, colas = ev.picos_nulos(E, phi, 3.5, fechas, cadena(M), np.random.default_rng(5), R=3)
    b, _ = ev.picos_nulos(E, phi, 3.5, fechas, cadena(M), np.random.default_rng(5), R=3)
    assert np.array_equal(a, b)
    assert (a >= 3.5).all()
    assert (a >= 9).mean() < 0.05
    assert colas[0] > colas[1] >= colas[2] >= colas[3]
