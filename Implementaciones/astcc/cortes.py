"""
Búsqueda del mejor corte de un bloque (Cap_3.tex, sec:prop-candidatos).

La ganancia de un corte de filas {I1, I2} es el G^2 de la tabla colapsada 2 x |J|
(eq:prop-G2-colapsada); la de un corte de columnas, el de la tabla |I| x 2.
Las funciones trabajan sobre lotes de tablas (B, m, n) para poder repetir la
búsqueda en todas las réplicas nulas a la vez (ajuste min-P, sec:prop-minP).
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import xlogy

from .estadisticos import G2

# Hilos para la enumeración exhaustiva de cortes (variable de entorno ASTCC_HILOS).
# El cálculo está limitado por la memoria: con más de 4 hilos ya no se gana nada.
HILOS = max(1, int(os.environ.get("ASTCC_HILOS", min(4, os.cpu_count() or 1))))
_EJECUTOR = None


def _ejecutor():
    global _EJECUTOR
    if _EJECUTOR is None:
        _EJECUTOR = ThreadPoolExecutor(HILOS)
    return _EJECUTOR


def _xlx(x):
    """x log x elemento a elemento, con 0 log 0 = 0."""
    return xlogy(x, x)


def mascaras_biparticion(k):
    """Todas las biparticiones no triviales de k elementos: (2^(k-1) - 1, k) en {0, 1}.
    El elemento 0 queda siempre en el grupo 0 para no contar cada corte dos veces."""
    codigos = np.arange(1, 2 ** (k - 1))
    return ((codigos[:, None] >> np.arange(k - 1)[None, :]) & 1).astype(float) @ np.eye(k - 1, k, 1)


_LOCAL = threading.local()


def _buferes(b, m, K):
    """Arreglos de trabajo del hilo. Reservar temporales grandes en cada trozo cuesta
    más que el propio cálculo (fallos de página) e impide que los hilos escalen."""
    forma = getattr(_LOCAL, "forma", None)
    if forma is None or forma[0] < b or forma[1:] != (m, K):
        _LOCAL.forma = (b, m, K)
        _LOCAL.grandes = [np.empty((b, m, K)) for _ in range(3)]
        _LOCAL.chicos = [np.empty((b, K)) for _ in range(4)]
    return [x[:b] for x in _LOCAL.grandes], [x[:b] for x in _LOCAL.chicos]


def _xlx_en(x, aux):
    """x log x sobre el propio x (x >= 0), con 0 log 0 = 0; aux es memoria de trabajo."""
    np.maximum(x, 1e-300, out=aux)
    np.log(aux, out=aux)
    x *= aux
    return x


def _ganancias_cortes(T, Mt):
    """G^2 de la tabla colapsada |I| x 2 para cada corte de columnas.
    T: (b, m, n); Mt: (n, K) con las máscaras en columnas. Devuelve (b, K), en un
    búfer del hilo que la siguiente llamada sobrescribe.

    Con las sumas por fila O_i., el total N y los grupos uno/cero de cada fila,
    G^2 / 2 = sum_i [xlx(uno_i) + xlx(cero_i)] - xlx(sum_i uno_i) - xlx(sum_i cero_i)
              - sum_i xlx(O_i.) + xlx(N),
    que evita construir los esperados de cada tabla colapsada."""
    b, m, n = T.shape
    K = Mt.shape[1]
    (uno, cero, aux), (g, tot_uno, tot_cero, aux2) = _buferes(b, m, K)
    filas = T.sum(-1)                                          # (b, m)
    N = filas.sum(-1)                                          # (b,)
    np.matmul(T.reshape(b * m, n), Mt, out=uno.reshape(b * m, K))
    np.subtract(filas[:, :, None], uno, out=cero)
    np.maximum(cero, 0.0, out=cero)                            # redondeo de la resta
    np.sum(uno, axis=1, out=tot_uno)
    np.subtract(N[:, None], tot_uno, out=tot_cero)
    np.maximum(tot_cero, 0.0, out=tot_cero)
    np.sum(_xlx_en(uno, aux), axis=1, out=g)
    np.sum(_xlx_en(cero, aux), axis=1, out=aux2)
    g += aux2
    g -= _xlx_en(tot_uno, aux2)
    g -= _xlx_en(tot_cero, aux2)
    g += (_xlx(N) - _xlx(filas).sum(-1))[:, None]
    g *= 2.0
    return g


def _mejor_en_trozo(T, Mt):
    g = _ganancias_cortes(T, Mt)
    k = np.argmax(g, axis=1)
    return g[np.arange(len(T)), k], k


def mejor_corte_columnas(T, ordenado=False, elementos_por_trozo=50_000):
    """Enumera todos los cortes de columnas (o solo los que respetan el orden si
    ordenado=True). T: (B, m, n). Devuelve (ganancia máxima (B,), máscara (B, n)).

    Las tablas se reparten en trozos que caben en la caché y se procesan en hilos
    (NumPy libera el GIL). El resultado no depende del número de hilos."""
    T = np.ascontiguousarray(T, dtype=float)
    B, m, n = T.shape
    if ordenado:
        M = (np.arange(n)[None, :] >= np.arange(1, n)[:, None]).astype(float)
    else:
        M = mascaras_biparticion(n)
    Mt = np.ascontiguousarray(M.T)
    paso = max(1, elementos_por_trozo // max(1, len(M) * m))
    trozos = [T[s:s + paso] for s in range(0, B, paso)]
    if len(trozos) > 1 and HILOS > 1:
        partes = list(_ejecutor().map(lambda t: _mejor_en_trozo(t, Mt), trozos))
    else:
        partes = [_mejor_en_trozo(t, Mt) for t in trozos]
    ganancias = np.concatenate([p[0] for p in partes])
    k = np.concatenate([p[1] for p in partes])
    return ganancias, M[k]


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


# ----------------------------------------------------------------------------
# Corte conjunto 2 x 2
# ----------------------------------------------------------------------------

def _suma_G2_hijos(T, a, b):
    """sum_C G^2_C de los cuatro hijos {I_r x J_c} de un corte conjunto, para lotes.
    T: (B, m, n); a: (B, m) y b: (B, n) en {0, 1}.
    sum_C G^2_C / 2 = sum xlogy(T) - sum_i sum_c xlogy(O_ic) - sum_r sum_j xlogy(O_rj)
                      + sum_rc xlogy(N_rc)."""
    A = np.stack([1 - a, a], axis=-1).astype(float)          # (B, m, 2)
    Bc = np.stack([1 - b, b], axis=-1).astype(float)         # (B, n, 2)
    fila_c = np.einsum("bmn,bnc->bmc", T, Bc)                # O_ic
    fila_r = np.einsum("bmr,bmn->brn", A, T)                 # O_rj
    N_rc = np.einsum("brn,bnc->brc", fila_r, Bc)
    return 2.0 * (_xlx(T).sum((1, 2)) - _xlx(fila_c).sum((1, 2))
                  - _xlx(fila_r).sum((1, 2)) + _xlx(N_rc).sum((1, 2)))


def _mejor_dimension(T, otra):
    """Con las columnas agrupadas según `otra` (B, n), elige el corte de filas que
    minimiza sum_C G^2_C, es decir, que maximiza
    S = sum_r sum_j xlogy(O_rj) - sum_rc xlogy(N_rc)
    (los demás términos no dependen del corte de filas). Enumera todos los cortes."""
    m = T.shape[1]
    M = mascaras_biparticion(m)                               # (K, m)
    Bc = np.stack([1 - otra, otra], axis=-1).astype(float)    # (B, n, 2)
    uno = np.einsum("km,bmn->bkn", M, T)                      # O_1j para cada máscara
    cero = T.sum(1)[:, None, :] - uno
    S = (_xlx(uno).sum(-1) + _xlx(cero).sum(-1)
         - _xlx(np.einsum("bkn,bnc->bkc", uno, Bc)).sum(-1)
         - _xlx(np.einsum("bkn,bnc->bkc", cero, Bc)).sum(-1))
    return M[np.argmax(S, axis=1)].astype(int)


def mejor_corte_conjunto(T, a0, b0, max_iter=10, exacto_hasta=14, cols_exacto_hasta=6):
    """Corte conjunto 2 x 2 con G^2_{B,s} = G^2_B - sum_C G^2_C (eq:prop-cadena).

    Si |J| <= cols_exacto_hasta y |I| <= exacto_hasta, la búsqueda es exacta: para
    cada corte de columnas se enumeran todos los de filas. Si no, minimización
    alternante (BBAC con base C6 y k = l = 2 restringido al bloque, Cap_3.tex
    sec:prop-candidatos) desde los cortes a0 (filas) y b0 (columnas), que alcanza el
    óptimo global en menos de la mitad de las tablas pequeñas probadas. El paso de
    filas enumera todos los cortes, así que con |I| > exacto_hasta se conserva a0 y
    solo se actualizan las columnas.
    Devuelve (ganancia (B,), a (B, m), b (B, n))."""
    T = np.asarray(T, float)
    B, m, n = T.shape
    if n <= cols_exacto_hasta and m <= exacto_hasta:
        mejor_g, mejor_a, mejor_b = np.full(B, -np.inf), None, None
        for c in mascaras_biparticion(n).astype(int):
            b = np.broadcast_to(c, (B, n))
            a = _mejor_dimension(T, b)
            g = G2(T) - _suma_G2_hijos(T, a, b)
            if mejor_a is None:
                mejor_g, mejor_a, mejor_b = g, a, b.copy()
            else:
                mejora = g > mejor_g
                mejor_g = np.where(mejora, g, mejor_g)
                mejor_a[mejora], mejor_b[mejora] = a[mejora], c
        return mejor_g, mejor_a, mejor_b
    a, b = np.array(a0, int), np.array(b0, int)
    for _ in range(max_iter):
        nueva_a = _mejor_dimension(T, b) if T.shape[1] <= exacto_hasta else a
        nueva_b = _mejor_dimension(np.swapaxes(T, 1, 2), nueva_a) if T.shape[2] <= exacto_hasta else b
        if np.array_equal(nueva_a, a) and np.array_equal(nueva_b, b):
            break
        a, b = nueva_a, nueva_b
    return G2(T) - _suma_G2_hijos(T, a, b), a, b


# ----------------------------------------------------------------------------
# Interfaz común con procesamiento por lotes
# ----------------------------------------------------------------------------

def mejor_corte(T, tipo, rng, R_ini=10, exacto_hasta=14, max_elementos=4_000_000):
    """Máximo de G^2_{B,s} sobre los cortes de un tipo ('fil', 'col', 'conj') para un
    lote de tablas (B, m, n), procesado en trozos para acotar la memoria.
    Devuelve (ganancia (B,), etiquetas): etiquetas es (a, b) con a (B, m) y b (B, n);
    en los cortes de un solo tipo, la otra dimensión queda toda en el grupo 0."""
    T = np.asarray(T, float)
    B, m, n = T.shape
    K = 2 ** (max(min(m, exacto_hasta), n) - 1)
    trozo = max(1, int(max_elementos // max(1, K * max(m, n) * 2)))
    if tipo == "col" or (tipo == "fil" and m <= exacto_hasta):
        trozo = max(B, 1)          # la enumeración exacta ya trocea y usa hilos
    gan, A, Bl = np.empty(B), np.zeros((B, m), int), np.zeros((B, n), int)
    for s in range(0, B, trozo):
        t = T[s:s + trozo]
        if tipo == "fil":
            g, et = mejor_corte_filas(t, rng, R_ini=R_ini, exacto_hasta=exacto_hasta)
            A[s:s + trozo] = et
        elif tipo == "col":
            g, et = mejor_corte_columnas(t)
            Bl[s:s + trozo] = et
        elif tipo == "conj":
            _, a0 = mejor_corte_filas(t, rng, R_ini=R_ini, exacto_hasta=exacto_hasta)
            _, b0 = mejor_corte_columnas(t)
            g, a, b = mejor_corte_conjunto(t, a0, b0, exacto_hasta=exacto_hasta)
            A[s:s + trozo], Bl[s:s + trozo] = a, b
        else:
            raise ValueError(f"tipo de corte desconocido: {tipo}")
        gan[s:s + trozo] = g
    return gan, (A, Bl)
