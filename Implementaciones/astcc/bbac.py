"""
Co-agrupamiento de Bregman con I-divergencia (BBAC_I), bases C2 y C5.

Cap_2.tex, sec:bbac y sec:idiv-poisson. Para una partición (rho, tau), con
I_g = rho^-1(g), J_h = tau^-1(h):

  C2:  O^_ij = N_gh / (|I_g| |J_h|)                          (eq:c2, bloques constantes)
  C5:  O^_ij = O_i. O_.j N_gh / (O_g. O_.h)                  (eq:c5, ITCC de Dhillon et al.)

El algoritmo es la minimización alternante de Banerjee et al. (2007, sec. 4):
con los estadísticos de la base fijos, cada fila pasa al grupo que minimiza su
contribución sum_j d_I(O_ij, O^_ij); se recalculan los estadísticos y se hace lo
mismo con las columnas. Ningún paso aumenta D_I(O || O^), así que converge a un
mínimo local; por eso se ejecuta desde varias inicializaciones aleatorias.

Contribución de la fila i al grupo g, salvo términos que no dependen de g
(O_ih = sum_{j in J_h} O_ij):
  C2:  sum_h [ |J_h| mu_gh - O_ih log mu_gh ],          mu_gh = N_gh / (|I_g| |J_h|)
  C5:  - sum_h O_ih log w_gh,                           w_gh = N_gh / (O_g. O_.h)
En C5 el término lineal es O_i. sum_h O_.h w_gh = O_i. para todo g y se cancela.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import xlogy

from .estadisticos import div

BASES = ("C2", "C5")
_LOG_MIN = np.log(1e-300)     # log(0) acotado: penaliza sin producir nan


@dataclass
class ResultadoBBAC:
    rho: np.ndarray                 # grupo de cada fila
    tau: np.ndarray                 # grupo de cada columna
    divergencia: float              # D_I(O || O^) de la mejor ejecución
    base: str
    iteraciones: int
    historia: list = field(default_factory=list)   # D_I tras cada medio paso (mejor ejecución)
    divergencias: np.ndarray = None                # D_I final de cada ejecución


def d_I(x, y):
    """I-divergencia total sum x log(x/y) - x + y, con 0 log 0 = 0."""
    return float(np.sum(xlogy(x, x) - xlogy(x, y) - x + y))


def indicadora(etiquetas, K):
    return np.eye(K)[etiquetas]


def aproximacion(O, rho, tau, base, KR=None, KT=None):
    """O^ de la base C2, C5 o C6 para la partición (rho, tau). C6 (eq:c6) es la
    independencia dentro de cada bloque; no se optimiza aquí, pero se usa para el
    término (iii) de la descomposición (prop:descomposicion)."""
    KR = KR or int(rho.max()) + 1
    KT = KT or int(tau.max()) + 1
    R, C = indicadora(rho, KR), indicadora(tau, KT)
    Ngh = R.T @ O @ C
    if base == "C2":
        return div(Ngh, np.outer(R.sum(0), C.sum(0)))[rho][:, tau]
    if base == "C5":
        Oi, Oj = O.sum(1), O.sum(0)
        return np.outer(Oi, Oj) * div(Ngh, np.outer(R.T @ Oi, C.T @ Oj))[rho][:, tau]
    if base == "C6":
        return div((O @ C)[:, tau] * (R.T @ O)[rho, :], Ngh[rho][:, tau])
    raise ValueError(f"base desconocida: {base}")


def descomposicion(O, rho, tau, KR=None, KT=None):
    """Términos de prop:descomposicion: D_I(O||C2) = (i) + (ii) + (iii)."""
    c2, c5, c6 = (aproximacion(O, rho, tau, b, KR, KT) for b in ("C2", "C5", "C6"))
    return {"total": d_I(O, c2), "i": d_I(c5, c2), "ii": d_I(c6, c5), "iii": d_I(O, c6)}


def divergencia(O, rho, tau, base, KR=None, KT=None):
    return d_I(O, aproximacion(O, rho, tau, base, KR, KT))


def _costos_filas(O, rho, tau, KR, KT, base):
    """Matriz (m, KR) con la contribución de cada fila a cada grupo de filas."""
    R, C = indicadora(rho, KR), indicadora(tau, KT)
    Oih = O @ C                                      # (m, KT)
    Ngh = R.T @ Oih                                  # (KR, KT)
    if base == "C2":
        mu = div(Ngh, np.outer(R.sum(0), C.sum(0)))
        log_mu = np.where(mu > 0, np.log(np.where(mu > 0, mu, 1.0)), _LOG_MIN)
        return (C.sum(0) * mu).sum(1)[None, :] - Oih @ log_mu.T
    w = div(Ngh, np.outer(R.T @ O.sum(1), C.T @ O.sum(0)))
    log_w = np.where(w > 0, np.log(np.where(w > 0, w, 1.0)), _LOG_MIN)
    return -Oih @ log_w.T


def _asignar(costos, K):
    """Asigna cada elemento al grupo de menor costo; si un grupo queda vacío, le
    pasa el elemento peor ajustado de un grupo con al menos dos elementos."""
    nuevas = np.argmin(costos, axis=1)
    for g in range(K):
        if not np.any(nuevas == g):
            propio = costos[np.arange(len(nuevas)), nuevas]
            tamanos = np.bincount(nuevas, minlength=K)
            candidatos = np.where(tamanos[nuevas] >= 2)[0]
            nuevas[candidatos[np.argmax(propio[candidatos])]] = g
    return nuevas


def _inicial(m, K, rng):
    e = np.arange(m) % K
    rng.shuffle(e)
    return e


def _ajuste_bloques(Ngh, tam_f, tam_c, base):
    """G = sum_ij O_ij log O^_ij salvo una constante de los datos, calculado solo
    con los totales de bloque. Como ambas bases conservan N,
    D_I(O || O^) = sum O log O - sum O log O^, y para una partición fija:
      C2:  sum O log O^ = sum_gh N_gh log N_gh - sum_g O_g. log|I_g| - sum_h O_.h log|J_h|
      C5:  sum O log O^ = sum_gh N_gh log N_gh - sum_g O_g. log O_g. - sum_h O_.h log O_.h
                          + sum_i O_i. log O_i. + sum_j O_.j log O_.j   (constante)
    tam_f, tam_c: |I_g| y |J_h| en C2; no se usan en C5."""
    Og, Oh = Ngh.sum(-1), Ngh.sum(-2)
    G = np.sum(xlogy(Ngh, Ngh), axis=(-2, -1))
    if base == "C2":
        return G - np.sum(xlogy(Og, tam_f), -1) - np.sum(xlogy(Oh, tam_c), -1)
    return G - np.sum(xlogy(Og, Og), -1) - np.sum(xlogy(Oh, Oh), -1)


def _busqueda_local(O, rho, tau, KR, KT, base, tol):
    """Mueve filas y columnas de una en una mientras alguna mejora D_I (Cho et al.,
    2004). Los pasos por lotes no ven estas mejoras porque asignan con los
    estadísticos fijos. Cada movimiento se evalúa de forma exacta pero solo con los
    totales de bloque (_ajuste_bloques), en O(K_R K_T) por grupo candidato."""
    mejoro = True
    while mejoro:
        mejoro = False
        for eje, (etiquetas, K, otras, Ko) in enumerate(((rho, KR, tau, KT), (tau, KT, rho, KR))):
            M = O if eje == 0 else O.T
            V = M @ indicadora(otras, Ko)                 # perfil de cada elemento por grupo de la otra dim.
            Ngh = indicadora(etiquetas, K).T @ V          # (K, Ko)
            tam = np.bincount(etiquetas, minlength=K).astype(float)
            tam_o = np.bincount(otras, minlength=Ko).astype(float)
            for i in range(len(etiquetas)):
                g0 = etiquetas[i]
                if tam[g0] < 2:
                    continue                              # no vaciar un grupo
                # Totales de bloque y tamaños si el elemento i pasa a cada grupo g
                cand = np.repeat(Ngh[None], K, axis=0)
                cand[:, g0] -= V[i]
                cand[np.arange(K), np.arange(K)] += V[i]
                tams = np.repeat(tam[None], K, axis=0)
                tams[:, g0] -= 1
                tams[np.arange(K), np.arange(K)] += 1
                if eje == 0:
                    G = _ajuste_bloques(cand, tams, tam_o[None], base)
                else:
                    G = _ajuste_bloques(np.swapaxes(cand, 1, 2), tam_o[None], tams, base)
                g = int(np.argmax(G))
                if G[g] > G[g0] + tol * max(1.0, abs(G[g0])):
                    etiquetas[i] = g
                    Ngh, tam = cand[g], tams[g]
                    mejoro = True
    return rho, tau, divergencia(O, rho, tau, base, KR, KT)


def optimizar_c6(O, rho, tau, KR, KT, max_barridos=50):
    """Rejilla global (rho, tau) que minimiza D_I(O || C6) directamente.

    BBAC_I solo optimiza las bases C2 y C5, pero una rejilla se evalúa y se compara con
    AST-CC bajo C6 (independencia dentro de cada bloque, def:aprox-arbol), que es la
    aproximación de cada hoja. Esta búsqueda local parte de (rho, tau) y mueve una fila
    o una columna a la vez al grupo que más reduce D_I(O || C6), hasta que ningún
    movimiento mejora; no vacía grupos. Devuelve (rho, tau, D_I(O || C6))."""
    O = np.asarray(O, float)
    rho, tau = np.array(rho), np.array(tau)
    actual = divergencia(O, rho, tau, "C6", KR, KT)
    for _ in range(max_barridos):
        mejoro = False
        for etiquetas, K in ((rho, KR), (tau, KT)):
            for i in range(len(etiquetas)):
                g0 = etiquetas[i]
                if np.sum(etiquetas == g0) < 2:
                    continue
                mejor, gm = actual, g0
                for g in range(K):
                    if g == g0:
                        continue
                    etiquetas[i] = g
                    v = divergencia(O, rho, tau, "C6", KR, KT)
                    if v < mejor - 1e-9 * max(1.0, abs(mejor)):
                        mejor, gm = v, g
                etiquetas[i] = gm
                if gm != g0:
                    actual, mejoro = mejor, True
        if not mejoro:
            break
    return rho, tau, actual


def bbac(O, KR, KT, base="C5", nruns=20, max_iter=100, tol=1e-9, rng=None,
         rho0=None, tau0=None, local=True):
    """BBAC_I con la base dada. Devuelve la mejor de nruns ejecuciones.

    Cada ejecución alterna pasos por lotes hasta converger y, con local=True,
    sigue con una búsqueda local de movimientos individuales; si esta mejora,
    se vuelve a los pasos por lotes.
    rho0, tau0: partición inicial opcional (se usa en la primera ejecución)."""
    if base not in BASES:
        raise ValueError(f"base debe ser una de {BASES}")
    O = np.asarray(O, float)
    m, n = O.shape
    if not (1 <= KR <= m and 1 <= KT <= n):
        raise ValueError("se necesita 1 <= K_R <= m y 1 <= K_T <= n")
    rng = rng if rng is not None else np.random.default_rng()

    mejor, finales = None, []
    for r in range(nruns):
        rho = np.array(rho0) if (r == 0 and rho0 is not None) else _inicial(m, KR, rng)
        tau = np.array(tau0) if (r == 0 and tau0 is not None) else _inicial(n, KT, rng)
        actual = divergencia(O, rho, tau, base, KR, KT)
        historia = [actual]
        it = 0
        while True:
            for it in range(it + 1, max_iter + 1):
                rho = _asignar(_costos_filas(O, rho, tau, KR, KT, base), KR)
                historia.append(divergencia(O, rho, tau, base, KR, KT))
                tau = _asignar(_costos_filas(O.T, tau, rho, KT, KR, base), KT)
                nueva = divergencia(O, rho, tau, base, KR, KT)
                historia.append(nueva)
                if actual - nueva <= tol * max(1.0, abs(actual)):
                    actual = nueva
                    break
                actual = nueva
            if not local or it >= max_iter:
                break
            rho, tau, nueva = _busqueda_local(O, rho, tau, KR, KT, base, tol)
            historia.append(nueva)
            if actual - nueva <= tol * max(1.0, abs(actual)):
                break
            actual = nueva
        finales.append(actual)
        if mejor is None or actual < mejor.divergencia:
            mejor = ResultadoBBAC(rho, tau, actual, base, it, historia)
    mejor.divergencias = np.array(finales)
    return mejor
