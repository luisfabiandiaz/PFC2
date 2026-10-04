"""
Datos sintéticos con estructura conocida para evaluar AST-CC (Cap_1, objetivo 3).

La media de la celda (i, j) es
    mu_ij = escala * a_i * b_j * lambda_{rho(i) tau(j)} * prod_k f_k(i, j),
con efectos de fila y columna desiguales (como en movilidad), un tablero base
(rho, tau) y patrones locales f_k: dentro del bloque (g, h), un subconjunto de
filas y uno de columnas se multiplican por un factor. Fuera de los patrones, cada
bloque cumple la independencia, así que todo rechazo en un bloque sin patrón es
un falso descubrimiento.

Los D días se generan como
    X_dij ~ Poisson(mu_ij / D * gamma_d * u_di * v_dj * G_dij),
con un efecto de día gamma_d común a toda la matriz, efectos día-región u_di y
día-intervalo v_dj (lognormales de media 1) y ruido gamma por celda y día G_dij
con forma k (k = inf: sin ruido de celda). Como todos los efectos tienen media 1
y son independientes, E[O_ij] = mu_ij y la independencia dentro de los bloques
sin patrón se cumple en esperanza; pero la combinación u_di v_dj produce, en cada
realización, desviaciones correlacionadas entre celdas como las de los datos
reales (Cap_3, sec:prop-sobredispersion).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Patron:
    g: int                 # grupo de filas del bloque base
    h: int                 # grupo de columnas
    frac_filas: float      # fracción de las filas del bloque que se activan
    frac_cols: float       # fracción de las columnas
    factor: float          # multiplicador de la media en la submatriz activada


@dataclass
class Sintetico:
    X: np.ndarray          # (D, m, n)
    rho: np.ndarray
    tau: np.ndarray
    mu: np.ndarray         # media total (m, n)
    activas: list = field(default_factory=list)   # [(filas, cols)] de cada patrón


def _lognormal_media_1(rng, sd, forma):
    return rng.lognormal(-sd ** 2 / 2, sd, forma) if sd > 0 else np.ones(forma)


def generar(rng, m=40, n=12, KR=4, KT=3, D=12, escala=2000.0, patrones=(),
            k=np.inf, var_dia=0.1, var_dia_fila=0.0, var_dia_col=0.0, modificar=None):
    """modificar(mu, rho, tau, rng) -> lista: patrones arbitrarios. Recibe la media
    (la modifica en el lugar) y devuelve una descripción de la verdad que se añade a
    `activas`."""
    rho = rng.permutation(np.arange(m) % KR)
    tau = rng.permutation(np.arange(n) % KT)
    a = rng.lognormal(0, 1.0, m)
    b = rng.lognormal(0, 0.5, n)
    lam = rng.lognormal(0, 0.7, (KR, KT))
    mu = escala * np.outer(a, b) * lam[rho][:, tau]
    activas = []
    for pt in patrones:
        I, J = np.where(rho == pt.g)[0], np.where(tau == pt.h)[0]
        fi = rng.choice(I, max(1, round(pt.frac_filas * len(I))), replace=False)
        fj = rng.choice(J, max(1, round(pt.frac_cols * len(J))), replace=False)
        mu[np.ix_(fi, fj)] *= pt.factor
        activas.append((np.sort(fi), np.sort(fj)))
    if modificar is not None:
        activas += modificar(mu, rho, tau, rng)
    gamma = _lognormal_media_1(rng, var_dia, D)
    u = _lognormal_media_1(rng, var_dia_fila, (D, m))
    v = _lognormal_media_1(rng, var_dia_col, (D, n))
    intensidad = (mu / D)[None] * gamma[:, None, None] * u[:, :, None] * v[:, None, :]
    if np.isfinite(k):
        intensidad = intensidad * rng.gamma(k, 1.0 / k, size=intensidad.shape)
    X = rng.poisson(intensidad).astype(float)
    return Sintetico(X, rho, tau, mu, activas)
