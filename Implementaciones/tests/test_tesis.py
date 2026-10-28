"""Pruebas de que astcc reproduce lo que afirma el Cap. 3 (Cap_3.tex): los ejemplos
numéricos del texto y las reglas de decisión y de parada del Algoritmo 1."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.cortes import mejor_corte  # noqa: E402
from astcc.estadisticos import G2, info_mutua  # noqa: E402
from astcc.nulas import replicas_semirreales  # noqa: E402
from astcc.sinteticos import Patron, generar  # noqa: E402


# ----------------------------------------------------------------------------
# Ejemplos numéricos del texto
# ----------------------------------------------------------------------------

def test_ejemplo_de_la_proposicion_de_ganancia():
    """sec:prop-descomp: bloque de cuatro regiones y dos horarios."""
    T = np.array([[30, 10], [28, 12], [10, 30], [12, 28]], float)
    assert G2(T) == pytest.approx(34.10, abs=0.01)
    assert info_mutua(T) == pytest.approx(0.107, abs=0.001)
    corte = np.array([[58, 22], [22, 58]], float)
    assert G2(corte) == pytest.approx(33.59, abs=0.01)
    hijos = [G2(T[:2]), G2(T[2:])]
    assert hijos == [pytest.approx(0.25, abs=0.01)] * 2
    assert G2(corte) + sum(hijos) == pytest.approx(G2(T))
    assert G2(corte) / G2(T) == pytest.approx(0.985, abs=0.001)
    g, (a, _) = mejor_corte(T[None], "fil", np.random.default_rng(0))
    assert g[0] == pytest.approx(33.59, abs=0.01) and set(a[0][:2]) != set(a[0][2:])
    g_col, _ = mejor_corte(T[None], "col", np.random.default_rng(0))
    assert g_col[0] == pytest.approx(G2(T))           # el corte de columnas explica todo


def test_nulo_de_constancia_frente_a_independencia():
    """sec:prop-estadisticos, 'Elección del modelo nulo'."""
    C = np.array([[80, 20], [40, 10]], float)
    assert 2 * np.sum(C * np.log(C / C.mean())) == pytest.approx(74.8, abs=0.1)
    assert G2(C) == pytest.approx(0.0, abs=1e-9)


def test_patron_en_tablero_lo_captura_un_corte_de_filas():
    X = np.array([[30, 10], [10, 30], [30, 10], [10, 30]], float)
    assert np.allclose(X.sum(1), 40) and np.allclose(X.sum(0), 80)
    g, (a, _) = mejor_corte(X[None], "fil", np.random.default_rng(0))
    assert g[0] == pytest.approx(41.9, abs=0.05)
    assert a[0][0] == a[0][2] != a[0][1] == a[0][3]


def test_desempate_por_z_del_ejemplo():
    z = lambda g, nu: (g - nu) / math.sqrt(2 * nu)
    assert z(60, 5) == pytest.approx(17.4, abs=0.05)
    assert z(45, 9) == pytest.approx(8.5, abs=0.05)


def test_bootstrap_aplica_la_fluctuacion_corregida():
    """eq:prop-fluctuacion: con D = 4 días de conteos (80, 100, 90, 130) por celda, las
    réplicas diarias valen E/D * F~_d, con F~ = 1 + sqrt(D/(D-1)) (F - 1)."""
    rng = np.random.default_rng(0)
    cuenta = np.array([8000.0, 10000.0, 9000.0, 13000.0])
    Xb = cuenta[:, None, None] * np.ones((4, 2, 2))
    R = replicas_semirreales(Xb, 2000, rng, diarias=True, inflar=True)
    F = cuenta / cuenta.mean()
    esperado = 10000.0 * np.maximum(0.0, 1 + math.sqrt(4 / 3) * (F - 1))
    assert np.allclose(esperado / 10000.0, [0.77, 1.00, 0.88, 1.35], atol=0.005)
    distancia = np.abs(R[..., None] - esperado).min(-1)
    assert distancia.max() < 1.0 + 1e-9                  # solo el redondeo estocástico
    for v in esperado:                                    # los cuatro días aparecen
        assert np.any(np.abs(R - v) < 1.0)
    # La réplica del ejemplo: días (4, 1, 4, 2) con E = 400 dan 446
    Ft = np.maximum(0.0, 1 + math.sqrt(4 / 3) * (np.array([80, 100, 90, 130]) / 100 - 1))
    assert 400 / 4 * (Ft[3] + Ft[0] + Ft[3] + Ft[1]) == pytest.approx(446, abs=1)


def test_lift_del_ejemplo():
    assert 360 / (800 * 3000 / 10000) == pytest.approx(1.5)


def test_jaccard_del_ejemplo_de_frecuencia_de_soporte():
    """sec:prop-representacion: una hoja de 3 x 2 celdas y otra de 2 x 2 contenida en ella
    tienen índice 4/6; no son la misma hoja con kappa = 0.7 y sí con kappa = 0.5."""
    from astcc.estabilidad import frecuencia_de_soporte, jaccard

    class Hoja:
        def __init__(self, filas, cols):
            self.filas, self.cols = np.array(filas), np.array(cols)

    grande, chica = Hoja([0, 1, 2], [0, 1]), Hoja([0, 1], [0, 1])
    assert jaccard(grande.filas, grande.cols, chica.filas, chica.cols) == pytest.approx(4 / 6)
    assert frecuencia_de_soporte([grande], [[chica]], kappa=0.7)[0] == 0.0
    assert frecuencia_de_soporte([grande], [[chica]], kappa=0.5)[0] == 1.0
    ajena = Hoja([5, 6], [3, 4])
    assert jaccard(grande.filas, grande.cols, ajena.filas, ajena.cols) == 0.0
    assert frecuencia_de_soporte([grande], [[chica], [ajena], [grande]])[0] == pytest.approx(2 / 3)


def test_divergencia_fuera_de_muestra():
    from astcc.estabilidad import divergencia_fuera_de_muestra as dfm
    rng = np.random.default_rng(0)
    O = rng.poisson(100, (6, 4)).astype(float)
    assert dfm(O, O * 7.0, suavizado=0.0) == pytest.approx(0.0, abs=1e-9)   # misma forma, otra escala
    ruido = rng.poisson(100, (6, 4)).astype(float)
    assert dfm(O, ruido) > 0
    assert np.isfinite(dfm(O, np.zeros((6, 4))))                              # el suavizado evita el infinito


# ----------------------------------------------------------------------------
# Algoritmo 1: nivel por bloque, réplicas y reglas de parada
# ----------------------------------------------------------------------------

def cuatro_bloques():
    """N = 10 000 viajes y cuatro bloques base independientes de 4000, 3000, 2000 y
    1000 (sec:prop-multiplicidad). Cada bloque es una tabla 2 x 2 exactamente
    independiente, así que ninguno puede rechazarse."""
    tablas = [np.array([[0.3, 0.3], [0.2, 0.2]]) * n for n in (4000, 3000, 2000, 1000)]
    X = np.vstack(tablas)[None]
    rho = np.repeat(np.arange(4), 2)
    return X, rho, np.zeros(2, int)


def ajustar(X, rho, tau, **kw):
    return ASTCC(Parametros(nula="poisson", **kw), np.random.default_rng(0)).ajustar(X, rho, tau)


def test_nivel_por_masa_y_numero_de_replicas():
    X, rho, tau = cuatro_bloques()
    r = ajustar(X, rho, tau)
    assert [n.alfa for n in r.raices] == pytest.approx([0.020, 0.015, 0.010, 0.005])
    assert sum(n.alfa for n in r.raices) == pytest.approx(0.05)
    # B = max{199, ceil(2 / alfa_B)}: 199, 199, 200, 400
    assert [n.B_usadas for n in r.raices] == [199, 199, 200, 400]
    assert all(n.estado == "homogenea" and n.p_omn == 1.0 for n in r.raices)


def test_replicas_acotadas_por_B_max_y_hoja_inalcanzable():
    X, rho, tau = cuatro_bloques()
    r = ajustar(X, rho, tau, B_max=40)       # 1 / 41 = 0.0244 supera todos los niveles
    assert all(n.estado == "inalcanzable" and n.B_usadas == 0 for n in r.raices)
    r = ajustar(X, rho, tau, B_max=100)      # 1 / 101 = 0.0099 solo supera al de 0.005
    assert [n.estado for n in r.raices] == ["homogenea"] * 3 + ["inalcanzable"]


def test_parada_secuencial_se_evalua_por_lotes():
    """Con B <= lote la regla no se activa; con B > lote, un bloque claramente no
    significativo se abandona tras el primer lote (Cap_3, sec:prop-multiplicidad)."""
    X, rho, tau = cuatro_bloques()
    r = ajustar(X, rho, tau, B_min=2000, lote=500)
    assert all(n.B_usadas == 500 and n.p_omn == 1.0 for n in r.raices)
    r = ajustar(X, rho, tau, B_min=2000, lote=5000)
    assert all(n.B_usadas == 2000 for n in r.raices)


def test_reglas_triviales():
    X, rho, tau = cuatro_bloques()
    r = ajustar(X, rho, tau, N_min=3500)               # masa mínima
    assert [n.estado for n in r.raices] == ["homogenea", "trivial", "trivial", "trivial"]
    assert all(n.estado == "trivial" for n in ajustar(X, rho, tau, d_max=0).raices)
    una_fila = np.array([[300.0, 300.0], [200.0, 200.0], [200.0, 200.0]])[None]
    r = ajustar(una_fila, np.array([0, 1, 1]), np.zeros(2, int))   # |I| < 2
    assert r.raices[0].estado == "trivial" and r.raices[1].estado == "homogenea"


def test_I_corregida_descuenta_el_ruido_del_nulo():
    """eq:prop-efecto-corregido: la información mutua corregida no supera a la cruda."""
    rng = np.random.default_rng(3)
    O = rng.poisson(40, (8, 5)).astype(float)
    r = ajustar(O[None], np.zeros(8, int), np.zeros(5, int))
    nodo = r.raices[0]
    assert 0.0 <= nodo.I_corr < info_mutua(O)


# ----------------------------------------------------------------------------
# Regla de decisión (Figura de la regla de decisión), con las pruebas dirigidas simuladas
# ----------------------------------------------------------------------------

M, N_ = 10, 6                      # nu_fil = N_ - 1 = 5 y nu_col = M - 1 = 9, como en el ejemplo


def bloque_heterogeneo():
    O = np.full((M, N_), 100.0)
    O[:5, :3] = O[5:, 3:] = 1000.0
    return O[None], np.zeros(M, int), np.zeros(N_, int)


def falsas(monkeypatch, fil, col, conj=(30.0, 0.5, 0.0)):
    """Sustituye las pruebas dirigidas por resultados dados (G2, p, media nula) y
    registra cuáles se evalúan."""
    llamadas = []
    a, b = np.repeat([0, 1], M // 2), np.repeat([0, 1], N_ // 2)
    etiquetas = {"fil": (a, np.zeros(N_, int)), "col": (np.zeros(M, int), b), "conj": (a, b)}
    datos = {"fil": fil, "col": col, "conj": conj}

    def prueba(self, T, replicas, tipo):
        llamadas.append(tipo)
        g, p, t = datos[tipo]
        return g, p, t, etiquetas[tipo]

    monkeypatch.setattr(ASTCC, "_prueba_dirigida", prueba)
    return llamadas


def raiz(**kw):
    X, rho, tau = bloque_heterogeneo()
    return ajustar(X, rho, tau, d_max=1, **kw).raices[0]


def test_gana_el_menor_p_valor_y_los_empates_se_resuelven_por_z(monkeypatch):
    falsas(monkeypatch, fil=(60.0, 0.001, 0.0), col=(45.0, 0.001, 0.0))
    assert raiz(delta=0.0).corte == "fil"                 # z = 17.4 frente a 8.5
    falsas(monkeypatch, fil=(20.0, 0.001, 0.0), col=(60.0, 0.001, 0.0))
    assert raiz(delta=0.0).corte == "col"                 # z = 4.7 frente a 12.0
    falsas(monkeypatch, fil=(60.0, 0.004, 0.0), col=(20.0, 0.001, 0.0))
    assert raiz(delta=0.0).corte == "col"                 # menor p-valor, aunque menor z


def test_un_solo_corte_unidimensional_significativo(monkeypatch):
    llamadas = falsas(monkeypatch, fil=(60.0, 0.3, 0.0), col=(45.0, 0.004, 0.0))
    nodo = raiz(delta=0.0)
    assert nodo.estado == "interno" and nodo.corte == "col" and len(nodo.hijos) == 2
    assert "conj" not in llamadas                         # el conjunto es el último recurso
    # eq:prop-alpha: cada hijo recibe alfa_B N_C / N_B, y los hijos suman lo del padre
    assert sum(h.N for h in nodo.hijos) == pytest.approx(nodo.N)
    for h in nodo.hijos:
        assert h.alfa == pytest.approx(nodo.alfa * h.N / nodo.N) and h.profundidad == 1
    assert sum(h.alfa for h in nodo.hijos) == pytest.approx(nodo.alfa)


def test_corte_conjunto_solo_si_ningun_corte_1D_es_significativo(monkeypatch):
    llamadas = falsas(monkeypatch, fil=(30.0, 0.5, 0.0), col=(30.0, 0.5, 0.0),
                      conj=(80.0, 0.001, 0.0))
    nodo = raiz(delta=0.0)
    assert llamadas == ["fil", "col", "conj"]
    assert nodo.corte == "conj" and len(nodo.hijos) == 4
    falsas(monkeypatch, fil=(30.0, 0.5, 0.0), col=(30.0, 0.5, 0.0), conj=(80.0, 0.5, 0.0))
    nodo = raiz(delta=0.0)
    assert nodo.estado == "no_localizada" and nodo.es_hoja


def test_ganancia_minima_corregida_por_el_nulo_del_corte(monkeypatch):
    """Un corte significativo solo se acepta si (G2_s - media nula) / (2 N) >= delta."""
    falsas(monkeypatch, fil=(60.0, 0.001, 0.0), col=(30.0, 0.5, 0.0))
    assert raiz(delta=1e-4).estado == "interno"           # 60 / 66 000 = 9.1e-4
    falsas(monkeypatch, fil=(60.0, 0.001, 59.99), col=(30.0, 0.5, 0.0))
    assert raiz(delta=1e-4).estado == "no_localizada"


# ----------------------------------------------------------------------------
# Corolario sobre un árbol con varios cortes
# ----------------------------------------------------------------------------

def test_corolario_con_varios_patrones_en_el_mismo_bloque():
    rng = np.random.default_rng(7)
    datos = generar(rng, m=30, n=8, KR=2, KT=2, D=10, escala=3000,
                    patrones=[Patron(0, 1, 0.5, 0.5, 2.5), Patron(0, 1, 0.5, 0.5, 2.5)])
    r = ajustar(datos.X, datos.rho, datos.tau, delta=0.001)
    cubierta = np.zeros(datos.X.shape[1:], int)
    for h in r.hojas():
        cubierta[np.ix_(h.filas, h.cols)] += 1
    assert np.all(cubierta == 1)
    d = r.descomposicion(datos.X.sum(0))
    assert abs(d["error"]) < 1e-6 * d["total"]
