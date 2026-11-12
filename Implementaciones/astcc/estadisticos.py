"""
Estadísticos de independencia dentro de un bloque (Cap_3.tex, eq:prop-EB y eq:prop-G2).

Todas las funciones aceptan una tabla (m, n) o un lote de tablas (..., m, n).
"""

import numpy as np
from scipy.special import xlogy


def div(a, b):
    """a / b con 0 donde b == 0."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def esperados(T):
    """E^B_ij = O_i. O_.j / N_B: conteos esperados bajo independencia en el bloque."""
    T = np.asarray(T, float)
    filas = T.sum(-1, keepdims=True)
    cols = T.sum(-2, keepdims=True)
    N = T.sum((-2, -1), keepdims=True)
    return div(filas * cols, N)


def G2(T):
    """G^2 = 2 sum O log(O / E^B), con 0 log 0 = 0."""
    T = np.asarray(T, float)
    filas, cols = T.sum(-1), T.sum(-2)
    N = filas.sum(-1)
    # G^2 / 2 = sum xlx(O) - sum xlx(O_i.) - sum xlx(O_.j) + xlx(N), sin formar E^B
    return 2.0 * (xlogy(T, T).sum((-2, -1)) - xlogy(filas, filas).sum(-1)
                  - xlogy(cols, cols).sum(-1) + xlogy(N, N))


def info_mutua(T):
    """I^_B = G^2 / (2 N_B), en nats por observación."""
    T = np.asarray(T, float)
    return div(G2(T), 2.0 * T.sum((-2, -1)))


def grados_libertad(m, n):
    return (m - 1) * (n - 1)


def pvalor_simulado(t_obs, t_nulos):
    """p = (1 + #{t_b >= t_obs}) / (1 + B)  (eq:prop-pval, Phipson y Smyth 2010).
    Se tolera un error relativo de 1e-9 para no romper empates exactos."""
    t_nulos = np.asarray(t_nulos)
    umbral = t_obs - 1e-9 * max(1.0, abs(t_obs))
    return (1 + np.sum(t_nulos >= umbral, axis=-1)) / (1 + t_nulos.shape[-1])
