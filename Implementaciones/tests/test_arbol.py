"""Pruebas del algoritmo AST-CC (astcc.arbol)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.sinteticos import Patron, generar  # noqa: E402


def ajustar(datos, **kw):
    params = Parametros(**kw)
    return ASTCC(params, np.random.default_rng(0)).ajustar(datos.X, datos.rho, datos.tau)


@pytest.fixture(scope="module")
def con_patron():
    rng = np.random.default_rng(1)
    return generar(rng, m=24, n=8, KR=2, KT=2, D=10, escala=3000,
                   patrones=[Patron(0, 1, 0.5, 0.5, 2.5)])


@pytest.mark.parametrize("nula", ["poisson", "bootstrap"])
def test_hojas_forman_una_particion(con_patron, nula):
    r = ajustar(con_patron, nula=nula, delta=0.002)
    cubierta = np.zeros(con_patron.X.shape[1:], int)
    for h in r.hojas():
        cubierta[np.ix_(h.filas, h.cols)] += 1
    assert np.all(cubierta == 1)


@pytest.mark.parametrize("nula", ["poisson", "bootstrap"])
def test_corolario_de_descomposicion(con_patron, nula):
    r = ajustar(con_patron, nula=nula, delta=0.002)
    d = r.descomposicion(con_patron.X.sum(0))
    assert abs(d["error"]) < 1e-6 * d["total"]
    assert d["explicada"] > 0


@pytest.mark.parametrize("nula", ["poisson", "bootstrap"])
def test_refina_solo_el_bloque_con_patron(con_patron, nula):
    r = ajustar(con_patron, nula=nula, delta=0.002)
    filas, cols = con_patron.activas[0]
    refinados = [raiz for raiz in r.raices if raiz.hijos]
    assert len(refinados) == 1
    raiz = refinados[0]
    assert set(filas) <= set(raiz.filas) and set(cols) <= set(raiz.cols)
    # El patrón (unas filas activas en unas columnas) se recupera con un corte de
    # filas que separa exactamente las filas activas: dentro de cada hijo todas las
    # filas comparten perfil, así que ambos hijos son homogéneos (Cap_3, "un corte
    # por nivel": la heterogeneidad era espacial).
    assert raiz.corte == "fil"
    assert any(set(h.filas) == set(filas) for h in raiz.hijos)
    assert all(h.es_hoja and h.estado == "homogenea" for h in raiz.hijos)


def test_sin_patron_no_refina():
    rng = np.random.default_rng(2)
    datos = generar(rng, m=24, n=8, KR=2, KT=2, D=10, escala=3000)
    r = ajustar(datos, nula="poisson", delta=0.0)
    assert all(not raiz.hijos for raiz in r.raices)


def test_delta_detiene_patrones_irrelevantes(con_patron):
    """Con un umbral de efecto mayor que la interacción plantada, no se refina nada."""
    r = ajustar(con_patron, nula="poisson", delta=0.5)
    assert all(not raiz.hijos for raiz in r.raices)
    assert any(raiz.estado == "homogenea" for raiz in r.raices)
