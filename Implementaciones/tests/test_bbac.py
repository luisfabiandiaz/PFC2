"""Pruebas de astcc.bbac (BBAC_I con bases C2 y C5)."""

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astcc.bbac import (_costos_filas, aproximacion, bbac, d_I,  # noqa: E402
                        divergencia)
from astcc.estadisticos import info_mutua  # noqa: E402


def plantada(rng, base, m=30, n=12, KR=3, KT=3, contraste=3.0):
    """Matriz de Poisson con un tablero plantado. En C2 los bloques son constantes;
    en C5 se multiplican además por efectos de fila y columna desiguales."""
    rho, tau = np.arange(m) % KR, np.arange(n) % KT
    rng.shuffle(rho)
    rng.shuffle(tau)
    lam = rng.uniform(1, contraste, (KR, KT)) ** 2
    mu = lam[rho][:, tau] * 50
    if base == "C5":
        mu = mu * np.outer(rng.lognormal(0, 0.8, m), rng.lognormal(0, 0.5, n))
    return rng.poisson(mu).astype(float), rho, tau


def test_C5_es_perdida_de_informacion_mutua():
    rng = np.random.default_rng(0)
    O = rng.poisson(20, (15, 8)).astype(float)
    rho, tau = rng.integers(0, 3, 15), rng.integers(0, 2, 8)
    R, C = np.eye(3)[rho], np.eye(2)[tau]
    esperado = O.sum() * (info_mutua(O) - info_mutua(R.T @ O @ C))
    assert np.isclose(divergencia(O, rho, tau, "C5", 3, 2), esperado)


@pytest.mark.parametrize("base", ["C2", "C5"])
def test_divergencia_no_aumenta(base):
    rng = np.random.default_rng(1)
    O, _, _ = plantada(rng, base)
    r = bbac(O, 3, 3, base=base, nruns=5, rng=rng)
    h = np.array(r.historia)
    assert np.all(np.diff(h) <= 1e-8 * h[0])


@pytest.mark.parametrize("base", ["C2", "C5"])
def test_solucion_es_minimo_local(base):
    """Ningún paso de filas ni de columnas mejora la solución final (Banerjee et al.,
    Teorema 6): cada fila y cada columna ya está en su grupo de menor costo."""
    rng = np.random.default_rng(2)
    O, _, _ = plantada(rng, base)
    r = bbac(O, 3, 3, base=base, nruns=5, rng=rng)
    cf = _costos_filas(O, r.rho, r.tau, 3, 3, base)
    cc = _costos_filas(O.T, r.tau, r.rho, 3, 3, base)
    assert np.allclose(cf[np.arange(len(r.rho)), r.rho], cf.min(1))
    assert np.allclose(cc[np.arange(len(r.tau)), r.tau], cc.min(1))


@pytest.mark.parametrize("base", ["C2", "C5"])
def test_recupera_tablero_plantado(base):
    rng = np.random.default_rng(3)
    O, rho, tau = plantada(rng, base)
    r = bbac(O, 3, 3, base=base, nruns=20, rng=rng)
    assert adjusted_rand_score(rho, r.rho) > 0.95
    assert adjusted_rand_score(tau, r.tau) > 0.95


def test_C5_ignora_el_volumen_y_C2_no():
    """Con efectos de fila fuertes y sin interacción por bloques, a C5 solo le queda
    el ruido de Poisson, mientras que C2 agrupa las filas por volumen y deja una
    divergencia mucho mayor."""
    rng = np.random.default_rng(4)
    mu = np.outer(rng.lognormal(0, 1.5, 40), rng.lognormal(0, 0.3, 10)) * 100
    O = rng.poisson(mu).astype(float)
    r2 = bbac(O, 4, 2, base="C2", nruns=10, rng=rng)
    volumen = np.log(O.sum(1))
    # En C2 los grupos de filas quedan ordenados por volumen (intervalos disjuntos)
    rangos = sorted((volumen[r2.rho == g].min(), volumen[r2.rho == g].max()) for g in range(4))
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(rangos, rangos[1:]))
    r5 = bbac(O, 4, 2, base="C5", nruns=10, rng=rng)
    # Nivel de ruido: E[D_I] ~ gl / 2, con gl = celdas - parámetros de C5 (tab:parametros)
    m, n, KR, KT = 40, 10, 4, 2
    ruido = 0.5 * (m * n - (KR * KT + (m - KR) + (n - KT)))
    assert r5.divergencia < 1.2 * ruido
    assert r2.divergencia > 20 * r5.divergencia


def test_C2_coincide_con_cgc():
    """La divergencia que reporta cgc (constante + error) es la de la base C2 para
    su propia partición, y nuestra BBAC C2 alcanza un óptimo igual o mejor."""
    cgc = pytest.importorskip("cgc.coclustering")
    logging.disable(logging.WARNING)
    rng = np.random.default_rng(5)
    O, _, _ = plantada(rng, "C2", m=40, n=12, KR=4, KT=3, contraste=1.5)
    np.random.seed(5)
    res = cgc.Coclustering(O, 4, 3, max_iterations=100, conv_threshold=1e-9, nruns=20).run_with_threads(nthreads=2)
    rho, tau = np.asarray(res.row_clusters), np.asarray(res.col_clusters)
    nz = O > 0
    constante = float(np.sum(O[nz] * np.log(O[nz]) - O[nz]))
    assert np.isclose(constante + res.error, divergencia(O, rho, tau, "C2", 4, 3), rtol=1e-6)
    nuestro = bbac(O, 4, 3, base="C2", nruns=20, rng=rng)
    assert nuestro.divergencia <= (constante + res.error) * (1 + 1e-6)


def test_aproximaciones_conservan_totales():
    rng = np.random.default_rng(6)
    O = rng.poisson(10, (12, 6)).astype(float)
    rho, tau = rng.integers(0, 3, 12), rng.integers(0, 2, 6)
    R, C = np.eye(3)[rho], np.eye(2)[tau]
    for base in ("C2", "C5"):
        A = aproximacion(O, rho, tau, base, 3, 2)
        assert np.allclose(R.T @ A @ C, R.T @ O @ C)          # totales de bloque
        assert d_I(O, A) >= 0
    A5 = aproximacion(O, rho, tau, "C5", 3, 2)
    assert np.allclose(A5.sum(1), O.sum(1)) and np.allclose(A5.sum(0), O.sum(0))


@pytest.mark.parametrize("base", ["C2", "C5"])
def test_ajuste_por_bloques_es_exacto(base):
    """La búsqueda local evalúa D_I solo con los totales de bloque: la diferencia de
    divergencia entre dos particiones debe ser exactamente la de _ajuste_bloques."""
    from astcc.bbac import _ajuste_bloques, indicadora
    rng = np.random.default_rng(7)
    O = rng.poisson(rng.lognormal(2, 1, (20, 9))).astype(float)
    valores = []
    for _ in range(6):
        rho, tau = rng.permutation(np.arange(20) % 4), rng.permutation(np.arange(9) % 3)
        Ngh = indicadora(rho, 4).T @ O @ indicadora(tau, 3)
        G = _ajuste_bloques(Ngh, np.bincount(rho, minlength=4), np.bincount(tau, minlength=3), base)
        valores.append((divergencia(O, rho, tau, base, 4, 3), G))
    d, g = np.array(valores).T
    assert np.allclose(d - d[0], -(g - g[0]))


@pytest.mark.parametrize("base", ["C2", "C5"])
def test_busqueda_local_no_empeora(base):
    rng = np.random.default_rng(8)
    O, _, _ = plantada(rng, base, contraste=1.3)
    sin = bbac(O, 3, 3, base=base, nruns=10, rng=np.random.default_rng(9), local=False)
    con = bbac(O, 3, 3, base=base, nruns=10, rng=np.random.default_rng(9), local=True)
    assert con.divergencia <= sin.divergencia + 1e-9
    assert np.isclose(con.divergencia, divergencia(O, con.rho, con.tau, base, 3, 3))


def test_optimizar_c6_no_empeora_y_conserva_los_grupos():
    from astcc.bbac import optimizar_c6
    rng = np.random.default_rng(10)
    O, _, _ = plantada(rng, "C5", m=30, n=12, KR=3, KT=3)
    b = bbac(O, 4, 3, base="C5", nruns=5, rng=rng)
    antes = divergencia(O, b.rho, b.tau, "C6", 4, 3)
    rho, tau, d = optimizar_c6(O, b.rho, b.tau, 4, 3)
    assert d <= antes + 1e-9
    assert np.isclose(d, divergencia(O, rho, tau, "C6", 4, 3))
    assert len(np.unique(rho)) == 4 and len(np.unique(tau)) == 3
    # no modifica la partición recibida
    assert divergencia(O, b.rho, b.tau, "C6", 4, 3) == pytest.approx(antes)
