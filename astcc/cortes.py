"""
Búsqueda del mejor corte de un bloque (Cap_3.tex, sec:prop-candidatos).

La ganancia de un corte de filas {I1, I2} es el G^2 de la tabla colapsada 2 x |J|
(eq:prop-G2-colapsada); la de un corte de columnas, el de la tabla |I| x 2.
Las funciones trabajan sobre lotes de tablas (B, m, n) para poder repetir la
búsqueda en todas las réplicas nulas a la vez (ajuste min-P, sec:prop-minP).
"""

import numpy as np

from .estadisticos import G2


def mascaras_biparticion(k):
    """Todas las biparticiones no triviales de k elementos: (2^(k-1) - 1, k) en {0, 1}.
    El elemento 0 queda siempre en el grupo 0 para no contar cada corte dos veces."""
    codigos = np.arange(1, 2 ** (k - 1))
    return ((codigos[:, None] >> np.arange(k - 1)[None, :]) & 1).astype(float) @ np.eye(k - 1, k, 1)


def mejor_corte_columnas(T, ordenado=False):
    """Enumera todos los cortes de columnas (o solo los que respetan el orden si
    ordenado=True). T: (B, m, n). Devuelve (ganancia máxima (B,), máscara (B, n))."""
    T = np.asarray(T, float)
    n = T.shape[-1]
    if ordenado:
        M = (np.arange(n)[None, :] >= np.arange(1, n)[:, None]).astype(float)
    else:
        M = mascaras_biparticion(n)
    uno = np.einsum("bmn,kn->bkm", T, M)               # suma de las columnas del grupo 1
    cero = T.sum(-1)[:, None, :] - uno
    ganancias = G2(np.stack([cero, uno], axis=-1))     # (B, K)
    k = np.argmax(ganancias, axis=1)
    return ganancias[np.arange(len(T)), k], M[k]


def mejor_corte_filas_exacto(T):
    """Enumeración exhaustiva de los cortes de filas (solo para bloques pequeños)."""
    g, M = mejor_corte_columnas(np.swapaxes(np.asarray(T, float), -1, -2))
    return g, M


def mejor_corte_filas(T, rng, R_ini=10, max_iter=50, exacto_hasta=14):
    """Mejor corte de filas. Con |I| <= exacto_hasta se enumeran todos los cortes;
    por encima, 2-medias con divergencia KL ponderada (paso de filas de ITCC con dos
    grupos) y R_ini inicializaciones aleatorias.

    Para cada fila i con perfil p(.|i) se elige el grupo r que maximiza
    sum_j O_ij log q_r(j), equivalente a minimizar O_i. KL(p(.|i) || q_r).
    Devuelve (ganancia máxima (B,), etiquetas (B, m))."""
    T = np.asarray(T, float)
    B, m, n = T.shape
    if m <= exacto_hasta:
        return mejor_corte_filas_exacto(T)
    return corte_filas_2medias(T, rng, R_ini, max_iter)


def corte_filas_2medias(T, rng, R_ini=10, max_iter=50):
    """2-medias KL con R_ini inicializaciones (ver mejor_corte_filas)."""
    B, m, n = T.shape
    Tr = np.repeat(T, R_ini, axis=0)                   # (B R_ini, m, n)
    etiq = rng.integers(0, 2, size=(B * R_ini, m))
    etiq[:, 0] = 0
    etiq[:, 1] = 1                                     # ambos grupos empiezan no vacíos
    for _ in range(max_iter):
        uno = np.einsum("bm,bmn->bn", etiq, Tr)
        cero = Tr.sum(1) - uno
        Q = np.stack([cero, uno], axis=1)              # (BR, 2, n)
        Q = Q / np.maximum(Q.sum(-1, keepdims=True), 1e-300)
        logQ = np.log(np.maximum(Q, 1e-300))
        puntaje = np.einsum("bmn,brn->bmr", Tr, logQ)
        nuevas = np.argmax(puntaje, axis=-1)
        vacio = (nuevas.sum(1) == 0) | (nuevas.sum(1) == m)
        nuevas[vacio] = etiq[vacio]                    # no aceptar un grupo vacío
        if np.array_equal(nuevas, etiq):
            break
        etiq = nuevas
    uno = np.einsum("bm,bmn->bn", etiq, Tr)
    ganancias = G2(np.stack([Tr.sum(1) - uno, uno], axis=1)).reshape(B, R_ini)
    mejor = np.argmax(ganancias, axis=1)
    etiq = etiq.reshape(B, R_ini, m)[np.arange(B), mejor]
    return ganancias[np.arange(B), mejor], etiq
