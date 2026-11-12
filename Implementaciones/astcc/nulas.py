"""
Modelos nulos de independencia dentro de un bloque (Cap_3.tex, sec:prop-calibracion
y sec:prop-sobredispersion), y estimación de la función de varianza con las
réplicas diarias.
"""

import numpy as np
from scipy.stats import random_table

from .estadisticos import div, esperados


# ----------------------------------------------------------------------------
# Réplicas nulas de un bloque
# ----------------------------------------------------------------------------

def replicas_poisson(T, R, rng):
    """Nula de Poisson: R tablas de Patefield con las marginales de T."""
    T = np.asarray(T)
    filas, cols = T.sum(1).astype(int), T.sum(0).astype(int)
    if filas.sum() == 0:
        return np.zeros((R,) + T.shape)
    return random_table(filas, cols).rvs(size=R, method="patefield", random_state=rng).astype(float)


def varianza_agregada(mu, a, b, D):
    """V_D(mu) = max{mu, a D^(1-b) mu^b}: varianza agregada de la ley de potencia de
    Taylor. Es la nula Gamma-Poisson que el Cap. 3 compara y descarta
    (tab:prop-calibracion)."""
    mu = np.asarray(mu, float)
    return np.maximum(mu, a * D ** (1 - b) * np.power(mu, b, where=mu > 0, out=np.zeros_like(mu)))


def replicas_taylor(T, R, a, b, D, rng):
    """Nula sobredispersa: O_ij ~ Poisson(Lambda_ij), Lambda_ij gamma con media E^B_ij
    y varianza V_D(E^B_ij) - E^B_ij. Con a = b = 1 se reduce a Poisson independiente."""
    E = esperados(np.asarray(T, float))
    exceso = varianza_agregada(E, a, b, D) - E
    con_gamma = exceso > 1e-12
    forma = div(E ** 2, exceso)
    escala = div(exceso, E)
    lam = np.broadcast_to(E, (R,) + E.shape).copy()
    if np.any(con_gamma):
        g = rng.gamma(np.where(con_gamma, forma, 1.0), np.where(con_gamma, escala, 1.0), size=(R,) + E.shape)
        lam = np.where(con_gamma, g, lam)
    return rng.poisson(lam).astype(float)


# ----------------------------------------------------------------------------
# Función de varianza a partir de las réplicas diarias
# ----------------------------------------------------------------------------

def ajuste_sin_interaccion_triple(X, tol=1e-10, max_iter=5000):
    """IPF del modelo log-lineal [DZ][DT][ZT] sobre el tensor día x región x intervalo."""
    E = np.full_like(X, X.mean(), dtype=float)
    for _ in range(max_iter):
        E *= div(X.sum(0), E.sum(0))[None, :, :]
        E *= div(X.sum(1), E.sum(1))[:, None, :]
        E *= div(X.sum(2), E.sum(2))[:, :, None]
        err = max(np.max(np.abs(E.sum(0) - X.sum(0))), np.max(np.abs(E.sum(1) - X.sum(1))))
        if err < tol * X.sum():
            break
    return E


def varianzas_diarias(X):
    """Media diaria, varianza cruda entre días y varianza residual del modelo
    [DZ][DT][ZT] por celda (escalada por los gl que corresponden a cada celda)."""
    D, Z, T = X.shape
    media = X.mean(0)
    var = X.var(0, ddof=1)
    E2 = ajuste_sin_interaccion_triple(X)
    var_res = np.sum((X - E2) ** 2, axis=0) / ((D - 1) * (Z - 1) * (T - 1) / (Z * T))
    return media, var, var_res


def ajustar_taylor(media, var, media_min=5.0):
    """Ajuste var = a mu^b por mínimos cuadrados en escala log (celdas activas)."""
    activas = (media >= media_min) & (var > 0)
    b, log_a = np.polyfit(np.log(media[activas]), np.log(var[activas]), 1)
    return float(np.exp(log_a)), float(b)


def phi_constante(media, var, media_min=5.0):
    """phi^ de la versión original de la Propuesta: mediana de var/media en celdas activas."""
    activas = media >= media_min
    return float(np.median(var[activas] / media[activas]))


# ----------------------------------------------------------------------------
# Nula semirreal para el experimento de calibración
# ----------------------------------------------------------------------------

def replicas_semirreales(Xb, R, rng, media_min=0.5, diarias=False, inflar=False, estratos=None):
    """Tablas de un bloque que cumplen H0 con la variabilidad real de los días
    (bootstrap multiplicativo de días).

    Xb: conteos diarios del bloque, forma (D, m, n). La media nula es la de
    independencia de la tabla agregada, e = E^B / D por día. Cada réplica remuestrea
    D días con reemplazo y aplica a e su fluctuación multiplicativa real
    F_d = X_d / X_media. Un efecto de día común a todo el bloque, a una fila o a una
    columna multiplica e por un factor de rango 1 y conserva la independencia; la
    combinación de efectos y la variación propia de cada celda se trasladan tal cual.
    En las celdas con media diaria menor que media_min el cociente F_d no es fiable:
    se toma F_d = 1 y la celda se sortea como Poisson con esa media.

    Devuelve las tablas agregadas (R, m, n) o, con diarias=True, los días
    remuestreados (R, D, m, n), cuya suma es la tabla agregada. Con inflar=True se
    aplica la corrección de muestra pequeña: las fluctuaciones F_d - 1 se
    multiplican por sqrt(D / (D - 1)), porque la varianza empírica de D días
    subestima la poblacional en ese factor.

    estratos: etiqueta de cada día (por ejemplo sábado, domingo, festivo). Con
    estratos, cada réplica remuestrea los días dentro de su estrato y conserva la
    composición del periodo; la corrección de muestra pequeña se aplica a las
    desviaciones respecto de la media de cada estrato, y un estrato de un solo día
    entra fijo en todas las réplicas. Como cada estrato conserva su suma esperada,
    la media de las réplicas sigue siendo E^B.
    """
    D = Xb.shape[0]
    media = Xb.mean(0)
    e = esperados(Xb.sum(0)) / D
    escasa = media < media_min
    F = np.where(escasa, 1.0, div(Xb, media))
    estratos = np.zeros(D, int) if estratos is None else np.asarray(estratos)
    dias = np.empty((R, D), int)
    for s in np.unique(estratos):
        idx = np.where(estratos == s)[0]
        if inflar and len(idx) > 1:
            centro = F[idx].mean(0)
            inflado = np.maximum(0.0, centro + np.sqrt(len(idx) / (len(idx) - 1)) * (F[idx] - centro))
            # El recorte en 0 sube la media del estrato; se reescala para conservarla
            F[idx] = inflado * div(centro, inflado.mean(0))
        dias[:, idx] = idx[rng.integers(0, len(idx), size=(R, len(idx)))]
    v = e * F[dias] if diarias else e * F[dias].sum(1)
    replicas = redondeo_estocastico(v, rng)
    if escasa.any():
        # Sin fluctuación diaria fiable, la celda recibe al menos el ruido de Poisson
        # (con F = 1 quedaría casi determinista y la prueba sería liberal)
        replicas[..., escasa] = rng.poisson(v[..., escasa])
    return replicas


def redondeo_estocastico(v, rng):
    """Redondea a entero sin sesgo: floor(v) + Bernoulli(v - floor(v)). Un redondeo
    al entero más cercano distorsiona las proporciones en celdas con pocos viajes."""
    base = np.floor(v)
    return base + (rng.random(v.shape) < (v - base))
