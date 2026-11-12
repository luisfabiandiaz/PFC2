"""
AST-CC: refinamiento jerárquico de la partición de ST-CC (Cap_3.tex, Algoritmos 1 y 2).

Cada bloque base B_gh recibe el nivel alfa_B = alfa N_B / N (eq:prop-alpha) y se
refina de forma recursiva:

  1. Reglas triviales: |I| < 2, |J| < 2, N_B < N_min o profundidad >= d_max -> hoja.
  2. Prueba ómnibus: G^2_B frente a réplicas nulas del bloque (bootstrap de días,
     eq:prop-bootstrap, o Patefield). Si p_omn > alfa_B o I~_B < delta -> hoja
     homogénea (I~_B: información mutua corregida por sesgo, eq:prop-efecto-corregido).
  3. Pruebas dirigidas con ajuste min-P (la búsqueda del mejor corte se repite en
     cada réplica): si p_fil o p_col <= alfa_B se aplica el de menor p-valor
     (desempate por z = (G^2 - nu) / sqrt(2 nu)); si ninguno, se prueba el corte
     conjunto 2 x 2; si tampoco es significativo -> hoja heterogénea no localizada.
  4. Ganancia mínima: si g~_{B,s} < delta -> hoja heterogénea no localizada.
  5. Cada hijo C recibe alfa_C = alfa_B N_C / N_B y se refina.

Las réplicas nulas se generan por lotes (Parametros.lote) y la regla de Besag y
Clifford (1991) se evalúa al cierre de cada lote: si h de ellas superan al
estadístico observado se detiene y el p-valor es la proporción observada. Con
B <= lote no se activa y se usan las B réplicas (Cap_3, sec:prop-multiplicidad). Si
el bloque supera la prueba ómnibus, las mismas réplicas se reutilizan para las
pruebas dirigidas. El número de réplicas de un bloque es
B = max(B_min, ceil(c / alfa_B)), acotado por B_max; si con B_max el p-valor
mínimo 1 / (B_max + 1) ya supera alfa_B, el bloque no puede rechazarse y se
marca como hoja "inalcanzable".
"""

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .bbac import aproximacion, d_I
from .cortes import mejor_corte
from .estadisticos import G2, esperados, pvalor_simulado
from .nulas import replicas_poisson, replicas_semirreales

TIPOS = ("fil", "col", "conj")


@dataclass
class Parametros:
    alfa: float = 0.05
    delta: float = 0.005          # nats por observación
    N_min: float = 500.0
    d_max: int = 6
    nula: str = "bootstrap"       # 'bootstrap' (referencia) o 'poisson'
    B_min: int = 199
    c: float = 2.0                # B >= c / alfa_B
    B_max: int = 20_000
    h: int = 10                   # Besag y Clifford
    lote: int = 500
    R_ini: int = 10
    exacto_hasta: int = 14


@dataclass
class Nodo:
    filas: np.ndarray
    cols: np.ndarray
    profundidad: int
    alfa: float
    N: float
    estado: str = ""              # trivial | inalcanzable | homogenea | no_localizada | interno
    G2: float = float("nan")
    I_corr: float = float("nan")
    p_omn: float = float("nan")
    B_usadas: int = 0
    p_cortes: dict = field(default_factory=dict)
    G2_cortes: dict = field(default_factory=dict)
    corte: str = ""
    G2_corte: float = float("nan")
    g_corr: float = float("nan")
    hijos: list = field(default_factory=list)
    segundos: float = 0.0

    @property
    def es_hoja(self):
        return not self.hijos

    def recorrer(self):
        yield self
        for h in self.hijos:
            yield from h.recorrer()


@dataclass
class ResultadoASTCC:
    raices: list
    rho: np.ndarray
    tau: np.ndarray
    params: Parametros
    segundos: float

    def nodos(self):
        for r in self.raices:
            yield from r.recorrer()

    def hojas(self):
        return [n for n in self.nodos() if n.es_hoja]

    def internos(self):
        return [n for n in self.nodos() if not n.es_hoja]

    def aproximacion(self, O):
        """O^T: independencia dentro de cada hoja (def:aprox-arbol)."""
        A = np.zeros_like(O, dtype=float)
        for h in self.hojas():
            A[np.ix_(h.filas, h.cols)] = esperados(O[np.ix_(h.filas, h.cols)])
        return A

    def descomposicion(self, O):
        """Corolario cor:stcc: D_I(O||C2) = (i) + (ii) + 1/2 sum G2_{B,s} + 1/2 sum G2_L."""
        rho, tau = np.unique(self.rho, return_inverse=True)[1], np.unique(self.tau, return_inverse=True)[1]
        KR, KT = int(rho.max()) + 1, int(tau.max()) + 1
        c2, c5, c6 = (aproximacion(O, rho, tau, b, KR, KT) for b in ("C2", "C5", "C6"))
        hojas = 0.5 * sum(float(G2(O[np.ix_(h.filas, h.cols)])) for h in self.hojas())
        cortes = 0.5 * sum(n.G2_corte for n in self.internos())
        return {"total": d_I(O, c2), "i": d_I(c5, c2), "ii": d_I(c6, c5),
                "explicada": cortes, "residual": hojas,
                "error": d_I(O, c2) - (d_I(c5, c2) + d_I(c6, c5) + cortes + hojas)}


class _Generador:
    """Réplicas nulas de un bloque, por lotes."""

    def __init__(self, X, nula, rng):
        self.X, self.nula, self.rng = X, nula, rng

    def __call__(self, filas, cols, B):
        if self.nula == "poisson":
            T = self.X.sum(0)[np.ix_(filas, cols)]
            return replicas_poisson(T, B, self.rng)
        Xb = self.X[:, filas][:, :, cols]
        return replicas_semirreales(Xb, B, self.rng, inflar=True)


def _hijos(nodo, tipo, etiquetas):
    a, b = etiquetas
    I, J = nodo.filas, nodo.cols
    if tipo == "fil":
        return [(I[a == 0], J), (I[a == 1], J)]
    if tipo == "col":
        return [(I, J[b == 0]), (I, J[b == 1])]
    return [(I[a == r], J[b == c]) for r in (0, 1) for c in (0, 1)]


def _rechaza(p, alfa):
    """p <= alfa con tolerancia: los empates exactos (1+k)/(1+B) no dependen del redondeo."""
    return p <= alfa * (1 + 1e-12)


def _gl(tipo, m, n):
    return {"fil": n - 1, "col": m - 1, "conj": m + n - 3}[tipo]


class ASTCC:
    def __init__(self, params=None, rng=None):
        self.p = params or Parametros()
        self.rng = rng if rng is not None else np.random.default_rng()

    def ajustar(self, X, rho, tau):
        """X: conteos diarios (D, m, n); (rho, tau): partición base de BBAC_I."""
        inicio = time.time()
        X = np.asarray(X, float)
        self.X, self.O = X, X.sum(0)
        self.N = self.O.sum()
        self.generar = _Generador(X, self.p.nula, self.rng)
        raices = []
        for g in np.unique(rho):
            for h in np.unique(tau):
                I, J = np.where(rho == g)[0], np.where(tau == h)[0]
                N_B = self.O[np.ix_(I, J)].sum()
                nodo = Nodo(I, J, 0, self.p.alfa * N_B / self.N, N_B)
                self._refinar(nodo)
                raices.append(nodo)
        return ResultadoASTCC(raices, np.asarray(rho), np.asarray(tau), self.p, time.time() - inicio)

    # ------------------------------------------------------------------
    def _refinar(self, nodo):
        t0 = time.time()
        p = self.p
        T = self.O[np.ix_(nodo.filas, nodo.cols)]
        m, n = T.shape
        nodo.G2 = float(G2(T))
        if m < 2 or n < 2 or nodo.N < p.N_min or nodo.profundidad >= p.d_max:
            nodo.estado = "trivial"
            return
        if nodo.alfa <= 0:
            nodo.estado = "inalcanzable"
            return
        B = min(max(p.B_min, math.ceil(p.c / nodo.alfa)), p.B_max)
        if 1.0 / (B + 1) > nodo.alfa:
            nodo.estado = "inalcanzable"
            return

        # Prueba ómnibus con parada secuencial
        replicas, t_nulos, superan = [], [], 0
        while sum(len(r) for r in replicas) < B:
            r = self.generar(nodo.filas, nodo.cols, min(p.lote, B - sum(len(x) for x in replicas)))
            t = G2(r)
            replicas.append(r)
            t_nulos.append(t)
            superan += int(np.sum(t >= nodo.G2 - 1e-9 * max(1.0, nodo.G2)))
            if superan >= p.h:
                break
        t_nulos = np.concatenate(t_nulos)
        nodo.B_usadas = len(t_nulos)
        if superan >= p.h and len(t_nulos) < B:
            nodo.p_omn = superan / len(t_nulos)          # Besag y Clifford
        else:
            nodo.p_omn = pvalor_simulado(nodo.G2, t_nulos)
        nodo.I_corr = max(0.0, (nodo.G2 - t_nulos.mean()) / (2 * nodo.N))
        if not _rechaza(nodo.p_omn, nodo.alfa) or nodo.I_corr < p.delta:
            nodo.estado = "homogenea"
            nodo.segundos = time.time() - t0
            return

        # Pruebas dirigidas con ajuste min-P sobre las mismas réplicas
        replicas = np.concatenate(replicas)
        resultados = {}
        for tipo in ("fil", "col"):
            resultados[tipo] = self._prueba_dirigida(T, replicas, tipo)
        significativos = [t for t in ("fil", "col") if _rechaza(resultados[t][1], nodo.alfa)]
        if significativos:
            def z(t):
                nu = _gl(t, m, n)
                return (resultados[t][0] - nu) / math.sqrt(2 * nu)
            tipo = significativos[0]
            if len(significativos) == 2:
                p_fil, p_col = resultados["fil"][1], resultados["col"][1]
                # En empate (p. ej. bloque 2 x 2, donde ambos cortes coinciden) se
                # prefiere el de filas; la tolerancia evita que decida el redondeo
                if p_col < p_fil or (p_col == p_fil and z("col") > z("fil") + 1e-9 * max(1.0, abs(z("fil")))):
                    tipo = "col"
        else:
            resultados["conj"] = self._prueba_dirigida(T, replicas, "conj")
            tipo = "conj" if _rechaza(resultados["conj"][1], nodo.alfa) else ""
        nodo.p_cortes = {t: r[1] for t, r in resultados.items()}
        nodo.G2_cortes = {t: r[0] for t, r in resultados.items()}
        if not tipo:
            nodo.estado = "no_localizada"
            nodo.segundos = time.time() - t0
            return

        g_obs, _, t_medio, etiquetas = resultados[tipo]
        nodo.g_corr = max(0.0, (g_obs - t_medio) / (2 * nodo.N))
        if nodo.g_corr < p.delta:
            nodo.estado = "no_localizada"
            nodo.segundos = time.time() - t0
            return

        nodo.estado, nodo.corte, nodo.G2_corte = "interno", tipo, g_obs
        nodo.segundos = time.time() - t0
        for I, J in _hijos(nodo, tipo, etiquetas):
            N_C = self.O[np.ix_(I, J)].sum()
            hijo = Nodo(I, J, nodo.profundidad + 1, nodo.alfa * N_C / nodo.N, N_C)
            self._refinar(hijo)
            nodo.hijos.append(hijo)

    def _prueba_dirigida(self, T, replicas, tipo):
        """(G^2_{B,s} observado, p-valor min-P, media nula, etiquetas del corte)."""
        p = self.p
        g, (a, b) = mejor_corte(T[None], tipo, self.rng, R_ini=p.R_ini, exacto_hasta=p.exacto_hasta)
        t_b, _ = mejor_corte(replicas, tipo, self.rng, R_ini=p.R_ini, exacto_hasta=p.exacto_hasta)
        return float(g[0]), float(pvalor_simulado(g[0], t_b)), float(t_b.mean()), (a[0], b[0])
