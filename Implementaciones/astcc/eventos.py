"""
Segunda etapa: eventos de pocos días en el cubo días x regiones x intervalos.

AST-CC describe el patrón que se repite entre días. Estas funciones puntúan cuánto se
aparta cada día de ese patrón y agrupan las desviaciones en eventos:

  puntuar, puntuar_celdas   z robusto de la cuota de una unidad (o de cada celda) en el día
  residuo                   z de la cuota de cada celda dentro de su unidad
  ganancia                  z de la ganancia G^2 del corte «día d frente a su línea base»
  esperados, dispersion     conteo esperado y sobredispersión de cada celda
  umbral_fdr                umbral con la parte central de los z como nula empírica
  componentes, picos        eventos como componentes conexas de celdas anómalas
  picos_nulos               eventos que aparecen por ruido en cubos simulados sin eventos

En todas, Y es un arreglo (días, regiones, intervalos) de conteos y `fechas` un
pandas.DatetimeIndex con un elemento por día. La línea base de un día son los demás días
del mismo día de la semana a menos de `ventana` días.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import norm

from .estadisticos import G2

VENTANA = 91                # días a cada lado para la línea base


def _linea_base(fechas, d, ventana):
    """Máscara de los días de la línea base del día d."""
    dsem, dia = fechas.dayofweek.to_numpy(), fechas.dayofyear.to_numpy()
    return (dsem == dsem[d]) & (np.abs(dia - dia[d]) <= ventana) & (np.arange(len(dia)) != d)


def z_robusto(S, N, fechas, ventana=VENTANA):
    """z robusto de las cuotas S (días x unidades, con denominadores N del mismo tamaño)
    frente a la línea base: (S - mediana) / max(1.4826 MAD, error típico de Poisson)."""
    Z = np.zeros_like(S)
    dsem, dia = fechas.dayofweek.to_numpy(), fechas.dayofyear.to_numpy()
    for w in np.unique(dsem):
        k = np.flatnonzero(dsem == w)
        for a in k:                                                # dejar el día fuera
            resto = k[(k != a) & (np.abs(dia[k] - dia[a]) <= ventana)]
            med = np.median(S[resto], 0)
            mad = 1.4826 * np.median(np.abs(S[resto] - med), 0)
            poisson = np.sqrt(np.maximum(med, 1 / N[a]) / N[a])    # suelo para unidades escasas
            Z[a] = (S[a] - med) / np.maximum(mad, poisson)
    return Z


def puntuar(Y, fechas, us, ventana=VENTANA):
    """z de la cuota de cada unidad en el total del día. `us` es una lista de (filas, cols).
    Dividir entre el total del día quita su nivel; queda el cambio de reparto.
    Devuelve días x unidades."""
    N = Y.sum((1, 2)).astype(float)
    C = np.stack([Y[:, f][:, :, c].sum((1, 2)) for f, c in us], 1)
    return z_robusto(C / N[:, None], np.broadcast_to(N[:, None], C.shape), fechas, ventana)


def puntuar_celdas(Y, fechas, ventana=VENTANA):
    """`puntuar` con cada celda como unidad. Devuelve un arreglo como Y."""
    D = len(Y)
    N = Y.sum((1, 2)).astype(float)
    C = Y.reshape(D, -1)
    return z_robusto(C / N[:, None], np.broadcast_to(N[:, None], C.shape), fechas, ventana).reshape(Y.shape)


def residuo(Y, fechas, us, ventana=VENTANA):
    """z de la cuota de cada celda dentro de su unidad, N(d, celda) / N(d, unidad). Si la
    unidad entera sube o baja ese día, la cuota no cambia. Devuelve días x celdas."""
    Nu = np.ones(Y.shape)
    for f, c in us:
        Nu[np.ix_(range(len(Y)), f, c)] = np.maximum(Y[:, f][:, :, c].sum((1, 2)), 1)[:, None, None]
    D = len(Y)
    return z_robusto((Y / Nu).reshape(D, -1), Nu.reshape(D, -1), fechas, ventana)


def amplitud(Y, fechas, us, z_min=3.0, ventana=VENTANA):
    """Fracción de unidades con |z| >= z_min en cada día. Un día de calendario (festivo,
    nevada) mueve muchas unidades; un evento local, pocas."""
    return (np.abs(puntuar(Y, fechas, us, ventana)) >= z_min).mean(1)


def ganancia(Y, fechas, us, ventana=VENTANA):
    """Ganancia del corte que separa el día d de su línea base dentro de cada unidad:

        g(d, u) = G^2([T_d ; suma de T en la línea base]) / (2 N_{d,u})    nats por llegada del día

    con las celdas de la unidad como la otra dimensión de la tabla. Se devuelve el z robusto
    de log g entre los días del mismo día de la semana, solo para las unidades de más de una
    celda: (días x unidades válidas, índices de esas unidades)."""
    dsem = fechas.dayofweek.to_numpy()
    D = len(fechas)
    validas = [k for k, (f, c) in enumerate(us) if len(f) * len(c) > 1]
    g = np.zeros((D, len(validas)))
    for a, k in enumerate(validas):
        f, c = us[k]
        T = Y[:, f][:, :, c].reshape(D, -1).astype(float)
        for d in range(D):
            base = _linea_base(fechas, d, ventana)
            g[d, a] = G2(np.stack([T[d], T[base].sum(0)])) / (2 * max(T[d].sum(), 1))
    lg = np.log(np.maximum(g, 1e-12))
    Z = np.zeros_like(lg)
    for w in np.unique(dsem):
        k = dsem == w
        med = np.median(lg[k], 0)
        mad = 1.4826 * np.median(np.abs(lg[k] - med), 0)
        Z[k] = (lg[k] - med) / np.maximum(mad, 1e-9)
    return Z, validas


def esperados(Y, fechas, ventana=VENTANA):
    """Conteo esperado de cada celda: mediana de su cuota en la línea base por el total del día."""
    N = Y.sum((1, 2)).astype(float)
    S = Y / N[:, None, None]
    E = np.zeros(Y.shape)
    for d in range(len(fechas)):
        E[d] = np.median(S[_linea_base(fechas, d, ventana)], 0) * N[d]
    return E


def dispersion(Y, E, normales):
    """Sobredispersión de cada celda (varianza / media), robusta: MAD de los residuos de
    Pearson en los días `normales`. 1 = Poisson."""
    r = (Y[normales] - E[normales]) / np.sqrt(np.maximum(E[normales], 0.5))
    med = np.median(r, 0)
    return np.maximum((1.4826 * np.median(np.abs(r - med), 0)) ** 2, 1.0)


def umbral_fdr(z, q=0.01, una_cola=False):
    """Menor t con tasa de falsos descubrimientos estimada <= q, tomando la parte central
    de z como nula N(mediana, s0^2) (Efron, 2004). Devuelve (t, s0)."""
    z = np.asarray(z).ravel()
    med = np.median(z)
    s0 = 1.4826 * np.median(np.abs(z - med))
    a = (z - med) if una_cola else np.abs(z - med)
    for t in np.arange(2.0, 60.0, 0.1):
        nulos = len(z) * norm.sf(t / s0) * (1 if una_cola else 2)
        if nulos / max((a >= t).sum(), 1) <= q:
            return float(t), float(s0)
    return 60.0, float(s0)


def componentes(M, signo, fechas, vecinas):
    """Componentes conexas de las celdas marcadas M (días x regiones x intervalos). Dos
    celdas marcadas del mismo signo se unen si son: regiones colindantes (`vecinas`) en el
    mismo día e intervalo; intervalos contiguos de la misma región y día; la misma región e
    intervalo en días consecutivos; o el último intervalo de un día y el primero del siguiente.
    Devuelve (nodos (n, 3), etiqueta de componente de cada nodo)."""
    nodos = np.argwhere(M)
    if len(nodos) == 0:
        return nodos, np.zeros(0, int)
    idx = -np.ones(M.shape, int)
    idx[tuple(nodos.T)] = np.arange(len(nodos))
    consecutivo = np.r_[(fechas[1:] - fechas[:-1]).days == 1, False]
    n_int = M.shape[2]
    a, b = [], []

    def unir(p, q):
        if idx[q] >= 0 and signo[p] == signo[q]:
            a.append(idx[p])
            b.append(idx[q])

    for d, i, j in map(tuple, nodos):
        p = (d, i, j)
        for k in np.flatnonzero(vecinas[i]):
            unir(p, (d, k, j))                       # región colindante
        if j + 1 < n_int:
            unir(p, (d, i, j + 1))                   # intervalo siguiente
        if consecutivo[d]:
            unir(p, (d + 1, i, j))                   # mismo intervalo al día siguiente
            if j == n_int - 1:
                unir(p, (d + 1, i, 0))               # medianoche
    n = len(nodos)
    _, etiq = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    return nodos, etiq


def picos(Z, M, fechas, vecinas):
    """|z| máximo de cada componente conexa de las celdas marcadas M.
    Devuelve (picos, nodos, etiquetas)."""
    nodos, etiq = componentes(M, np.sign(Z), fechas, vecinas)
    p = np.zeros(etiq.max() + 1 if len(nodos) else 0)
    np.maximum.at(p, etiq, np.abs(Z[tuple(nodos.T)]))
    return p, nodos, etiq


def simular(E, phi, rng):
    """Cubo sin eventos: cada celda con media E y varianza phi E (binomial negativa como
    mezcla gamma-Poisson; Poisson si phi = 1), independiente de las demás."""
    escala = np.maximum(phi - 1, 1e-9)
    return rng.poisson(rng.gamma(np.maximum(E, 1e-9) / escala, escala))


def picos_nulos(E, phi, t, fechas, vecinas, rng, R, ventana=VENTANA):
    """Picos de los eventos que aparecen sin que haya ninguno: R cubos de `simular` pasados
    por la misma puntuación y la misma agrupación (celdas con |z| >= t). Devuelve (picos de
    todas las réplicas, fracción media de celdas con |z| >= 3, 4, 5, 6)."""
    todos, colas = [], []
    for _ in range(R):
        Zs = puntuar_celdas(simular(E, phi, rng), fechas, ventana)
        todos.append(picos(Zs, np.abs(Zs) >= t, fechas, vecinas)[0])
        colas.append([(np.abs(Zs) >= u).mean() for u in (3, 4, 5, 6)])
    return np.concatenate(todos), np.mean(colas, 0)


def ganancias_nulas(E, phi, fechas, us, rng, R, ventana=VENTANA):
    """z de la ganancia de cada unidad en R cubos de `simular` (sin eventos).
    Devuelve (R, días, unidades válidas)."""
    return np.stack([ganancia(simular(E, phi, rng), fechas, us, ventana)[0] for _ in range(R)])


def umbral_nulo(obs, nulos, R, q=0.05, factor=1.0, desde=2.0, hasta=40.0):
    """Menor umbral u (en pasos de 0.1) con

        (nº de valores nulos >= u por réplica) x factor / (nº de observados >= u) <= q,

    es decir, con una fracción estimada de ruido entre lo detectado de q como mucho.
    `nulos` reúne los valores de R réplicas simuladas; `factor` ajusta el tamaño de la
    réplica al de los datos (por ejemplo, la fracción de días que se analizan). Si ningún
    umbral lo cumple mientras queda algo observado, devuelve infinito: lo observado no se
    distingue del ruido."""
    obs, nulos = np.asarray(obs).ravel(), np.asarray(nulos).ravel()
    for u in np.arange(desde, hasta, 0.1):
        n_obs = (obs >= u).sum()
        if n_obs == 0:
            break
        if (nulos >= u).sum() / R * factor / n_obs <= q:
            return float(u)
    return float("inf")


def fusionar(huellas, signos, holgura=0):
    """Une los fragmentos de un mismo evento. Dos huellas (arreglos (n, 3) de día, región,
    intervalo) del mismo signo, cada una de un solo día, se juntan si son del mismo día y
    comparten algún intervalo (o distan `holgura` como mucho), estén donde estén sus
    regiones. Las huellas de varios días no se funden con nada: una feria de cuatro días
    coincide en hora con casi cualquier otro evento de esos días.
    Devuelve la etiqueta de grupo de cada huella."""
    n = len(huellas)
    if n == 0:
        return np.zeros(0, int)
    horas = [{} for _ in range(n)]                    # día -> intervalos de la huella ese día
    for k, h in enumerate(huellas):
        for d, _, j in h:
            horas[k].setdefault(int(d), set()).add(int(j))
    a, b = [], []
    for k in range(n):
        for m in range(k + 1, n):
            if signos[k] != signos[m] or len(horas[k]) > 1 or len(horas[m]) > 1:
                continue
            if any(abs(x - y) <= holgura for d in horas[k].keys() & horas[m].keys()
                   for x in horas[k][d] for y in horas[m][d]):
                a.append(k)
                b.append(m)
    return connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)[1]
