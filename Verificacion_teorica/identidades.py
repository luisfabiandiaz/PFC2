"""
Verificación numérica de las identidades algebraicas del marco teórico
(Cap_2.tex) y de la propuesta (Cap_3.tex).

Todas las afirmaciones verificadas aquí son exactas: si son correctas, se
cumplen en cualquier matriz de conteos y cualquier partición, salvo error de
redondeo (~1e-10). El script genera matrices, particiones y árboles aleatorios,
evalúa ambos lados de cada identidad y reporta el error relativo máximo.
Un FALLO indica un error en el enunciado o en la fórmula; un OK no es una
demostración, pero descarta casi cualquier error algebraico.

Uso (entorno stcc):
    python identidades.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.linalg import qr
from scipy.special import xlogy

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

SEMILLA = 20260928
N_ENSAYOS = 300       # matrices aleatorias para las identidades cerradas
N_ENSAYOS_GLM = 40    # ajustes GLM (más lentos)
TOL = 1e-9            # identidades en forma cerrada
TOL_GLM = 1e-6        # comparación con un ajuste iterativo (IRLS)

rng = np.random.default_rng(SEMILLA)


# ============================================================================
# Utilidades
# ============================================================================

def div(a, b):
    """a / b con 0 donde b == 0."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0)


def d_I(x, y):
    """I-divergencia sum x log(x/y) - x + y, con 0 log 0 = 0 (Tabla tab:bregman)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(xlogy(x, x) - xlogy(x, y) - x + y))


def esperados(T):
    """Conteos esperados bajo independencia dentro de la tabla T (eq:prop-EB)."""
    return div(np.outer(T.sum(1), T.sum(0)), T.sum())


def G2(T):
    """Estadístico G^2 de independencia, escrito literalmente como en eq:c6-g2."""
    return float(2.0 * np.sum(xlogy(T, T) - xlogy(T, esperados(T))))


def info_mutua(T):
    """Información mutua de la distribución empírica T / sum(T) (eq:im)."""
    P = T / T.sum()
    return float(np.sum(xlogy(P, P) - xlogy(P, np.outer(P.sum(1), P.sum(0)))))


def onehot(etiquetas, K=None):
    etiquetas = np.asarray(etiquetas)
    if K is None:
        _, etiquetas = np.unique(etiquetas, return_inverse=True)
        K = etiquetas.max() + 1
    return np.eye(K)[etiquetas]


def rango(X):
    return int(np.linalg.matrix_rank(X))


def columnas_independientes(X):
    """Subconjunto de columnas de X linealmente independientes que genera el mismo espacio."""
    _, R, piv = qr(X, mode="economic", pivoting=True)
    r = rango(X)
    return X[:, np.sort(piv[:r])]


def matriz(m, n, densa=True):
    """Conteos de Poisson con efectos de fila, de columna e interacción,
    con volúmenes desiguales como en los datos de movilidad."""
    a = rng.lognormal(0, 1.0, m)
    b = rng.lognormal(0, 0.7, n)
    interaccion = rng.lognormal(0, 0.5, (m, n))
    escala = 40.0 if densa else 0.4
    O = rng.poisson(escala * np.outer(a, b) * interaccion).astype(float)
    if densa:
        O += 1.0  # sin ceros: los EMV del GLM quedan en el interior
    elif O.sum() == 0:
        O[0, 0] = 1.0
    return O


def particion(m, K):
    etiquetas = np.arange(m) % K
    rng.shuffle(etiquetas)
    return etiquetas


def biparticion(k):
    while True:
        e = rng.integers(0, 2, k)
        if 0 < e.sum() < k:
            return e


def tamanos():
    m, n = int(rng.integers(4, 15)), int(rng.integers(4, 13))
    KR, KT = int(rng.integers(1, min(4, m) + 1)), int(rng.integers(1, min(4, n) + 1))
    return m, n, KR, KT


# ============================================================================
# Registro de resultados
# ============================================================================

resultados = {}


def error_relativo(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    escala = max(1.0, float(np.max(np.abs(a), initial=0)), float(np.max(np.abs(b), initial=0)))
    return float(np.max(np.abs(a - b), initial=0)) / escala


def registrar(afirmacion, referencia, lhs, rhs, tol=TOL):
    r = resultados.setdefault(afirmacion, {"ref": referencia, "err": 0.0, "n": 0, "tol": tol})
    r["err"] = max(r["err"], error_relativo(lhs, rhs))
    r["n"] += 1


def registrar_bool(afirmacion, referencia, cumple):
    registrar(afirmacion, referencia, 0.0 if cumple else 1.0, 0.0)


# ============================================================================
# Cap. 2: bases C2, C5, C6 y modelos log-lineales
# ============================================================================

def bases(O, rho, tau, KR, KT):
    """Aproximaciones eq:c2, eq:c5 y eq:c6 tal como están escritas en la tesis."""
    R, C = onehot(rho, KR), onehot(tau, KT)
    Ngh = R.T @ O @ C
    tam = np.outer(R.sum(0), C.sum(0))
    c2 = div(Ngh, tam)[rho][:, tau]

    Oi, Oj = O.sum(1), O.sum(0)
    Og, Oh = R.T @ Oi, C.T @ Oj
    c5 = np.outer(Oi, Oj) * div(Ngh, np.outer(Og, Oh))[rho][:, tau]

    Oih = (O @ C)[:, tau]       # O_{ih} con h = tau(j)
    Ogj = (R.T @ O)[rho, :]     # O_{gj} con g = rho(i)
    c6 = div(Oih * Ogj, Ngh[rho][:, tau])
    return c2, c5, c6, Ngh


def disenos(m, n, rho, tau, KR, KT):
    """Matrices de diseño de eq:ll-c2, eq:ll-c5, eq:ll-c6 (celdas en orden fila-mayor)."""
    ii, jj = np.divmod(np.arange(m * n), n)
    X2 = onehot(rho[ii] * KT + tau[jj], KR * KT)
    X5 = np.hstack([onehot(ii, m), onehot(jj, n), X2])
    X6 = np.hstack([onehot(ii * KT + tau[jj], m * KT), onehot(rho[ii] * n + jj, KR * n)])
    return X2, X5, X6


def es_emv_birch(X, y, mu):
    """Condiciones de Birch: mu reproduce los estadísticos suficientes X'y y
    log mu pertenece al espacio columna de X. Devuelve el error máximo."""
    err_score = error_relativo(X.T @ mu, X.T @ y)
    coef, *_ = np.linalg.lstsq(X, np.log(mu), rcond=None)
    err_espacio = error_relativo(X @ coef, np.log(mu))
    return max(err_score, err_espacio)


def verificar_cap2(O, rho, tau, KR, KT, con_glm=False):
    m, n = O.shape
    N = O.sum()
    c2, c5, c6, Ngh = bases(O, rho, tau, KR, KT)
    I_g = [np.where(rho == g)[0] for g in range(KR)]
    J_h = [np.where(tau == h)[0] for h in range(KT)]

    # eq:devianza con una media arbitraria (no solo la de máxima verosimilitud)
    mu = O * rng.uniform(0.5, 1.5, O.shape) + rng.uniform(0.1, 1, O.shape)
    dev = sm.families.Poisson().deviance(O.ravel(), mu.ravel())
    registrar("D_I(O||mu) = Dev(mu)/2", "eq:devianza", d_I(O, mu), dev / 2)

    # Homogeneidad y escala
    c = rng.uniform(0.1, 10)
    registrar("d_I(cx,cy) = c d_I(x,y)", "eq:homogeneidad", d_I(c * O, c * mu), c * d_I(O, mu))

    # eq:perdida-im
    registrar("D_I(O||C5) = N[I(R;T) - I(R^;T^)]", "eq:perdida-im",
              d_I(O, c5), N * (info_mutua(O) - info_mutua(Ngh)))

    # eq:c6-g2 y eq:g2-im
    G2_bloques = [G2(O[np.ix_(I, J)]) for I in I_g for J in J_h]
    registrar("D_I(O||C6) = 1/2 sum G2_gh", "eq:c6-g2", d_I(O, c6), 0.5 * sum(G2_bloques))
    for I in I_g:
        for J in J_h:
            B = O[np.ix_(I, J)]
            if B.sum() > 0:
                registrar("G2_gh = 2 N_gh I_gh(R;T)", "eq:g2-im", G2(B), 2 * B.sum() * info_mutua(B))

    # Proposición de descomposición: (i) + (ii) + (iii)
    registrar("Descomposición D_I(O||C2) = (i)+(ii)+(iii)", "prop:descomposicion",
              d_I(O, c2), d_I(c5, c2) + d_I(c6, c5) + 0.5 * sum(G2_bloques))

    # Identidad pitagórica usada en la demostración, par por par
    registrar("Pitágoras C2 ⊂ C5", "prop:descomposicion (dem.)", d_I(O, c2), d_I(O, c5) + d_I(c5, c2))
    registrar("Pitágoras C5 ⊂ C6", "prop:descomposicion (dem.)", d_I(O, c5), d_I(O, c6) + d_I(c6, c5))

    # Número de parámetros (Tabla tab:parametros) y anidamiento C2 ⊂ C5 ⊂ C6
    X2, X5, X6 = disenos(m, n, rho, tau, KR, KT)
    r2, r5, r6 = rango(X2), rango(X5), rango(X6)
    registrar("Parámetros C2 = K_R K_T", "tab:parametros", r2, KR * KT)
    registrar("Parámetros C5 = K_R K_T + (m-K_R) + (n-K_T)", "tab:parametros",
              r5, KR * KT + (m - KR) + (n - KT))
    registrar("Parámetros C6 = K_T m + K_R n - K_R K_T", "tab:parametros", r6, KT * m + KR * n - KR * KT)
    registrar("Saturado - C6 = sum (|I_g|-1)(|J_h|-1)", "tab:parametros (texto)",
              m * n - r6, sum((len(I) - 1) * (len(J) - 1) for I in I_g for J in J_h))
    registrar_bool("Anidamiento C2 ⊂ C5 ⊂ C6", "sec:loglineal",
                   rango(np.hstack([X2, X5])) == r5 and rango(np.hstack([X5, X6])) == r6)

    # Las formas cerradas son los EMV del modelo log-lineal
    y = O.ravel()
    for nombre, X, aprox in (("C2", X2, c2), ("C5", X5, c5), ("C6", X6, c6)):
        if not np.all(aprox > 0):
            continue    # EMV en la frontera (log mu = -inf); se verifica en las matrices densas
        registrar(f"{nombre} cumple las condiciones de Birch (EMV)", f"eq:c{nombre[1]} / sec:loglineal",
                  es_emv_birch(X, y, aprox.ravel()), 0.0)
        if con_glm:
            Xr = columnas_independientes(X)   # mismo espacio columna, rango completo
            inicio, *_ = np.linalg.lstsq(Xr, np.log(y), rcond=None)
            ajuste = sm.GLM(y, Xr, family=sm.families.Poisson()).fit(
                start_params=inicio, tol=1e-14, maxiter=1000)
            registrar(f"{nombre} = ajuste GLM de Poisson (statsmodels)", f"eq:c{nombre[1]} / sec:loglineal",
                      ajuste.fittedvalues, aprox.ravel(), tol=TOL_GLM)

    # Escala: promedios diarios (sec:escala)
    D = int(rng.integers(2, 31))
    registrar("G2(S/D) = G2(S)/D", "sec:escala", G2(O / D), G2(O) / D)
    for nombre, a, b in zip(("C2", "C5", "C6"), bases(c * O, rho, tau, KR, KT)[:3], (c2, c5, c6)):
        registrar(f"Aproximación {nombre} equivariante a la escala", "sec:escala", a, c * b)


# ============================================================================
# Cap. 2: propiedades de las divergencias de Bregman
# ============================================================================

def verificar_bregman():
    k, d = int(rng.integers(3, 12)), int(rng.integers(2, 8))
    X = rng.gamma(2.0, 3.0, (k, d))
    w = rng.dirichlet(np.ones(k))
    media = w @ X
    s = rng.gamma(2.0, 3.0, d)
    lhs = sum(w[u] * d_I(X[u], s) for u in range(k))
    rhs = sum(w[u] * d_I(X[u], media) for u in range(k)) + d_I(media, s)
    registrar("E[d(X,s)] = E[d(X,mu)] + d(mu,s)", "eq:media", lhs, rhs)

    # Información de Bregman con KL sobre condicionales = información mutua
    T = rng.gamma(1.0, 1.0, (k, d))
    pu = T.sum(1) / T.sum()
    cond = T / T.sum(1, keepdims=True)
    py = T.sum(0) / T.sum()
    registrar("I_phi(X) con KL = I(U;Y)", "eq:info-bregman (texto)",
              sum(pu[u] * d_I(cond[u], py) for u in range(k)), info_mutua(T))


# ============================================================================
# Cap. 3: árbol de refinamiento, ganancia de un corte y corolario
# ============================================================================

def cortar(I, J, tipo):
    if tipo == "fil":
        a = biparticion(len(I))
        return [(I[a == 0], J), (I[a == 1], J)]
    if tipo == "col":
        b = biparticion(len(J))
        return [(I, J[b == 0]), (I, J[b == 1])]
    a, b = biparticion(len(I)), biparticion(len(J))
    return [(I[a == x], J[b == y]) for x in (0, 1) for y in (0, 1)]


def tipos_posibles(I, J):
    t = []
    if len(I) >= 2:
        t.append("fil")
    if len(J) >= 2:
        t.append("col")
    if len(I) >= 2 and len(J) >= 2:
        t.append("conj")
    return t


def arbol_aleatorio(rho, tau, KR, KT, prof_max=4, p_corte=0.65):
    hojas, internos = [], []
    pila = [(np.where(rho == g)[0], np.where(tau == h)[0], 0) for g in range(KR) for h in range(KT)]
    while pila:
        I, J, prof = pila.pop()
        tipos = tipos_posibles(I, J)
        if prof < prof_max and tipos and rng.random() < p_corte:
            tipo = str(rng.choice(tipos))
            hijos = cortar(I, J, tipo)
            internos.append((I, J, tipo, hijos))
            pila.extend((Ic, Jc, prof + 1) for Ic, Jc in hijos)
        else:
            hojas.append((I, J))
    return hojas, internos


def aprox_arbol(O, hojas):
    A = np.zeros_like(O)
    for I, J in hojas:
        A[np.ix_(I, J)] = esperados(O[np.ix_(I, J)])
    return A


def G2_corte(O, I, J, hijos):
    """G2_{B,s} = 2 D_I(E^{B,s} || E^B), con E^{B,s} la aproximación del corte."""
    F = np.zeros_like(O)
    for Ic, Jc in hijos:
        F[np.ix_(Ic, Jc)] = esperados(O[np.ix_(Ic, Jc)])
    return 2.0 * d_I(F[np.ix_(I, J)], esperados(O[np.ix_(I, J)]))


def G2_colapsada_literal(O, I, J, hijos, tipo):
    """eq:prop-G2-colapsada escrita término a término (y su simétrica por columnas)."""
    if tipo == "fil":
        T = np.vstack([O[np.ix_(Ic, J)].sum(0) for Ic, _ in hijos])      # 2 x |J|
    else:
        T = np.vstack([O[np.ix_(I, Jc)].sum(1) for _, Jc in hijos])      # 2 x |I|
    NB = T.sum()
    Nr = T.sum(1)
    Ocol = T.sum(0)
    denom = np.outer(Nr, Ocol)
    mask = T > 0
    return float(2 * np.sum(T[mask] * np.log((T * NB)[mask] / denom[mask]))), T


def diseno_local(I, J, hijos=None, parametrizacion=None):
    """Diseño restringido al bloque B = I x J.
    hijos=None -> independencia en B; con hijos -> efectos de fila y columna por hijo.
    parametrizacion='fil' o 'conj' -> las formas escritas en la demostración de prop:ganancia."""
    ii, jj = np.divmod(np.arange(len(I) * len(J)), len(J))
    if hijos is None:
        return np.hstack([onehot(ii, len(I)), onehot(jj, len(J))])
    pos_i = {v: k for k, v in enumerate(I)}
    pos_j = {v: k for k, v in enumerate(J)}
    hijo_fila = np.zeros(len(I), int)   # a(i) / r(i)
    hijo_col = np.zeros(len(J), int)    # b(j)
    hijo_celda = np.zeros((len(I), len(J)), int)
    for c, (Ic, Jc) in enumerate(hijos):
        for i in Ic:
            for j in Jc:
                hijo_celda[pos_i[i], pos_j[j]] = c
    if parametrizacion is None:
        cc = hijo_celda[ii, jj]
        return np.hstack([onehot(cc * len(I) + ii), onehot(cc * len(J) + jj)])
    # Grupos de filas y columnas inducidos por el corte
    filas_hijo = [set(pos_i[i] for i in Ic) for Ic, _ in hijos]
    cols_hijo = [set(pos_j[j] for j in Jc) for _, Jc in hijos]
    grupos_f = sorted({frozenset(s) for s in filas_hijo}, key=min)
    grupos_c = sorted({frozenset(s) for s in cols_hijo}, key=min)
    for a, g in enumerate(grupos_f):
        hijo_fila[list(g)] = a
    for b, g in enumerate(grupos_c):
        hijo_col[list(g)] = b
    if parametrizacion == "fil":       # log mu = alpha_i + beta_{r(i), j}
        return np.hstack([onehot(ii), onehot(hijo_fila[ii] * len(J) + jj)])
    if parametrizacion == "col":       # simétrico: alpha_{i, b(j)} + beta_j
        return np.hstack([onehot(hijo_col[jj] * len(I) + ii), onehot(jj)])
    # conj: log mu = alpha_{i, b(j)} + beta_{a(i), j}
    return np.hstack([onehot(hijo_col[jj] * len(I) + ii), onehot(hijo_fila[ii] * len(J) + jj)])


def verificar_cap3(O, rho, tau, KR, KT, con_rangos=True):
    m, n = O.shape
    c2, c5, c6, _ = bases(O, rho, tau, KR, KT)
    hojas, internos = arbol_aleatorio(rho, tau, KR, KT)
    A = aprox_arbol(O, hojas)

    # Árbol sin refinar = base C6
    base = [(np.where(rho == g)[0], np.where(tau == h)[0]) for g in range(KR) for h in range(KT)]
    registrar("Árbol con hojas = bloques base da C6", "def:aprox-arbol (texto)", aprox_arbol(O, base), c6)

    # (a) Descomposición por hojas
    registrar("(a) D_I(O||O^T) = 1/2 sum_L G2_L", "eq:prop-desc-hojas",
              d_I(O, A), 0.5 * sum(G2(O[np.ix_(I, J)]) for I, J in hojas))

    # (b) Cadena en cada nodo interno, forma colapsada, regla de la cadena y gl
    suma_cortes = 0.0
    for I, J, tipo, hijos in internos:
        B = O[np.ix_(I, J)]
        g2s = G2_corte(O, I, J, hijos)
        suma_cortes += g2s
        registrar("(b) G2_B = G2_{B,s} + sum_C G2_C", "eq:prop-cadena",
                  G2(B), g2s + sum(G2(O[np.ix_(Ic, Jc)]) for Ic, Jc in hijos))
        registrar_bool("(b) G2_{B,s} >= 0", "prop:ganancia (b)", g2s >= -1e-9 * max(1, G2(B)))

        if tipo in ("fil", "col"):
            literal, T = G2_colapsada_literal(O, I, J, hijos, tipo)
            etiqueta = "fil" if tipo == "fil" else "col (simétrico)"
            registrar(f"(b) G2_{{B,s}} = fórmula colapsada, corte {etiqueta}", "eq:prop-G2-colapsada", g2s, literal)
            registrar(f"(b) G2_{{B,s}} = G2 de la tabla colapsada 2x|.|, corte {etiqueta}",
                      "eq:prop-G2-colapsada", g2s, G2(T))
            if B.sum() > 0:
                NB = B.sum()
                rhs = info_mutua(T) + sum(
                    O[np.ix_(Ic, Jc)].sum() / NB * info_mutua(O[np.ix_(Ic, Jc)])
                    for Ic, Jc in hijos if O[np.ix_(Ic, Jc)].sum() > 0)
                registrar("Regla de la cadena de la IM", "eq:prop-regla-cadena", info_mutua(B), rhs)

        if con_rangos and len(I) * len(J) <= 300:
            X0 = diseno_local(I, J)
            Xh = diseno_local(I, J, hijos)
            Xp = diseno_local(I, J, hijos, parametrizacion=tipo)
            r0, rh = rango(X0), rango(Xh)
            gl = rh - r0
            gl_tesis = {"fil": len(J) - 1, "col": len(I) - 1, "conj": len(I) + len(J) - 3}[tipo]
            registrar(f"(b) gl del corte ({tipo})", "prop:ganancia (b)", gl, gl_tesis)
            registrar("Parámetros del modelo del corte = sum_C(|I_C|+|J_C|-1)", "dem. prop:ganancia",
                      rh, sum(len(Ic) + len(Jc) - 1 for Ic, Jc in hijos))
            registrar_bool("Independencia en B anidada en el modelo del corte", "dem. prop:ganancia",
                           rango(np.hstack([X0, Xh])) == rh)
            registrar_bool(f"Parametrización escrita ({tipo}) = efectos por hijo", "dem. prop:ganancia",
                           rango(Xp) == rh and rango(np.hstack([Xp, Xh])) == rh)

    # Corolario: descomposición total
    registrar("Corolario: D_I(O||C2) = (i)+(ii)+ 1/2 sum G2_{B,s} + 1/2 sum G2_L", "eq:prop-desc-total",
              d_I(O, c2),
              d_I(c5, c2) + d_I(c6, c5) + 0.5 * suma_cortes
              + 0.5 * sum(G2(O[np.ix_(I, J)]) for I, J in hojas))

    # (c) Cortar una hoja reduce la divergencia exactamente en G2_{B,s}/2
    candidatas = [k for k, (I, J) in enumerate(hojas) if tipos_posibles(I, J)]
    if candidatas:
        k = candidatas[int(rng.integers(len(candidatas)))]
        I, J = hojas[k]
        hijos = cortar(I, J, str(rng.choice(tipos_posibles(I, J))))
        nuevas = hojas[:k] + hojas[k + 1:] + hijos
        registrar("(c) Reducción al cortar una hoja = G2_{B,s}/2", "prop:ganancia (c)",
                  d_I(O, A) - d_I(O, aprox_arbol(O, nuevas)), 0.5 * G2_corte(O, I, J, hijos))

    # Parámetros del modelo del árbol (penalización BIC, eq:prop-bic) y EMV
    if con_rangos and m * n <= 200:
        ii, jj = np.divmod(np.arange(m * n), n)
        hoja_celda = np.zeros((m, n), int)
        for L, (I, J) in enumerate(hojas):
            hoja_celda[np.ix_(I, J)] = L
        hc = hoja_celda[ii, jj]
        X = np.hstack([onehot(hc * m + ii), onehot(hc * n + jj)])
        registrar("Parámetros del árbol = sum_L(|I_L|+|J_L|-1)", "def:aprox-arbol / eq:prop-bic",
                  rango(X), sum(len(I) + len(J) - 1 for I, J in hojas))
        if np.all(A > 0):
            registrar("O^T cumple las condiciones de Birch (EMV)", "def:aprox-arbol (texto)",
                      es_emv_birch(X, O.ravel(), A.ravel()), 0.0)


# ============================================================================
# Cap. 3: cortes de filas y 2-medias con KL; patrón en tablero
# ============================================================================

def verificar_kl_2medias():
    k, d = int(rng.integers(3, 9)), int(rng.integers(3, 10))
    B = matriz(k, d, densa=bool(rng.integers(0, 2)))
    B = B[B.sum(1) > 0][:, B.sum(0) > 0] if B.sum() > 0 else B
    k = B.shape[0]
    if k < 2 or B.shape[1] < 2:
        return
    G2B = G2(B)
    perfiles = div(B, B.sum(1, keepdims=True))
    ganancias, costos = [], []
    for mascara in range(1, 2 ** (k - 1)):         # todas las biparticiones de filas
        r = np.array([(mascara >> i) & 1 for i in range(k)])
        T = np.vstack([B[r == 0].sum(0), B[r == 1].sum(0)])
        perfil_grupo = div(T, T.sum(1, keepdims=True))
        costo = sum(B[i].sum() * d_I(perfiles[i], perfil_grupo[r[i]]) for i in range(k))
        g = G2(T)
        registrar("1/2 G2_B = 1/2 G2_{B,s} + sum_i O_i. KL(p(.|i)||p(.|r(i)))",
                  "sec:prop-candidatos (Filas)", 0.5 * G2B, 0.5 * g + costo)
        ganancias.append(g)
        costos.append(costo)
    registrar_bool("argmax G2_{B,s} = argmin costo KL ponderado", "sec:prop-candidatos (Filas)",
                   np.isclose(ganancias[int(np.argmin(costos))], max(ganancias), rtol=1e-12))


def verificar_tablero():
    """Patrón en tablero (XOR) con marginales idénticas dentro del bloque."""
    h, w = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    alto, bajo = rng.uniform(5, 50), rng.uniform(0.5, 5)
    # Filas 0..h-1 activas en columnas 0..w-1; filas h..2h-1 en columnas w..2w-1
    B = np.block([[np.full((h, w), alto), np.full((h, w), bajo)],
                  [np.full((h, w), bajo), np.full((h, w), alto)]])
    perm_f, perm_c = rng.permutation(2 * h), rng.permutation(2 * w)
    B = B[perm_f][:, perm_c]
    grupo = (perm_f >= h).astype(int)     # grupo verdadero de cada fila

    registrar_bool("Tablero: totales de fila y columna constantes", "sec:prop-estadisticos (tablero)",
                   np.allclose(B.sum(1), B.sum(1)[0]) and np.allclose(B.sum(0), B.sum(0)[0]))
    registrar_bool("Tablero: G2_B > 0", "sec:prop-estadisticos (tablero)", G2(B) > 1e-6)
    T = np.vstack([B[grupo == 0].sum(0), B[grupo == 1].sum(0)])
    registrar("Tablero: el corte de filas correcto captura todo G2_B", "sec:prop-estadisticos (tablero)",
              G2(T), G2(B))
    mejores = max(G2(np.vstack([B[r == 0].sum(0), B[r == 1].sum(0)]))
                  for r in (np.array([(mk >> i) & 1 for i in range(2 * h)]) for mk in range(1, 2 ** (2 * h - 1))))
    registrar("Tablero: el corte correcto es el máximo sobre todas las biparticiones",
              "sec:prop-estadisticos (tablero)", G2(T), mejores)


def verificar_nulo_comun():
    """obs:nulo-comun a nivel poblacional: bajo independencia ningún corte tiene
    efecto; fuera de ella, algún corte de una sola fila sí lo tiene."""
    k, d = int(rng.integers(2, 8)), int(rng.integers(2, 8))
    mu = np.outer(rng.gamma(2, 5, k), rng.gamma(2, 5, d))
    maximo = 0.0
    for mascara in range(1, 2 ** (k - 1)):
        r = np.array([(mascara >> i) & 1 for i in range(k)])
        maximo = max(maximo, G2(np.vstack([mu[r == 0].sum(0), mu[r == 1].sum(0)])))
    registrar("Independencia => G2_{B,s}(mu) = 0 para todo corte de filas", "obs:nulo-comun", maximo, 0.0)

    mu2 = mu.copy()
    mu2[rng.integers(k), rng.integers(d)] *= rng.uniform(1.5, 3)
    singulares = [G2(np.vstack([mu2[np.arange(k) != i].sum(0), mu2[i]])) for i in range(k)]
    registrar_bool("No independencia => algún corte de una fila tiene G2 > 0", "obs:nulo-comun",
                   max(singulares) > 1e-8)


# ============================================================================
# Ejecución
# ============================================================================

def main():
    for e in range(N_ENSAYOS):
        m, n, KR, KT = tamanos()
        densa = e % 2 == 0          # la mitad con muchos ceros, como en bloques poco activos
        O = matriz(m, n, densa=densa)
        rho, tau = particion(m, KR), particion(n, KT)
        verificar_cap2(O, rho, tau, KR, KT, con_glm=densa and e < 2 * N_ENSAYOS_GLM)
        verificar_cap3(O, rho, tau, KR, KT, con_rangos=e < 120)
        verificar_bregman()
        verificar_kl_2medias()
        verificar_tablero()
        verificar_nulo_comun()

    lineas = [f"Verificación de identidades  (semilla {SEMILLA}, {N_ENSAYOS} matrices aleatorias)", ""]
    ancho = max(len(k) for k in resultados)
    lineas.append(f"{'Afirmación':<{ancho}}  {'Referencia':<34} {'n':>6} {'err. máx.':>10}  estado")
    lineas.append("-" * (ancho + 66))
    fallos = 0
    for afirmacion, r in resultados.items():
        ok = r["err"] <= r["tol"]
        fallos += not ok
        lineas.append(f"{afirmacion:<{ancho}}  {r['ref']:<34} {r['n']:>6} {r['err']:>10.1e}  "
                      f"{'OK' if ok else 'FALLO'}")
    lineas += ["", f"{len(resultados) - fallos} de {len(resultados)} afirmaciones verificadas; {fallos} fallos."]
    texto = "\n".join(lineas)
    print(texto)
    salida = Path(__file__).resolve().parents[1] / "Resultados/Verificacion_teorica"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resultados_identidades.txt").write_text(texto, encoding="utf-8")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
