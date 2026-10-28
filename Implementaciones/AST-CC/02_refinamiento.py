"""
Refinamiento AST-CC sobre los datos reales y sensibilidad a delta
=================================================================
Cap_3.tex, Algoritmos 1 y 2, y sec:prop-parada ("Elección de delta").

Para cada periodo (laborable, feriado) y cada partición base (C5, la adoptada, y
C2, la variante) se ajusta AST-CC con la nula de bootstrap de días y se recorre la
rejilla de delta de la Propuesta. Para cada configuración se reporta:
  - hojas por tipo y número de cortes de cada tipo
  - profundidad máxima
  - fracción del término de interacción local (iii) de ST-CC que explica el árbol,
    sum G2_{B,s} / sum G2_gh (Corolario cor:stcc), y comprobación del corolario
  - réplicas usadas y tiempo

Para delta = 0.005 se guardan los descriptores de hoja (sec:prop-representacion):
regiones, intervalos, masa, I~_L, tipo y dos niveles de actividad relativa:
  actividad_global = N_L / sum_L O_i. O_.j / N   (independencia de toda la matriz)
  actividad_padre  = N_L / sum_L E^P_ij          (eq:prop-lift, independencia del padre)
La segunda vale exactamente 1 en toda hoja que sale de un corte de filas o de
columnas, porque E^P reproduce los totales de fila y de columna del padre; solo
es informativa tras un corte conjunto.

Nota: ambos periodos usan el bootstrap sin estratificar por tipo de día. En el
feriado, estratificar no mejora la calibración (E5: tipo I medio 0.109 frente a 0.045
sin estratos) y queda como limitación conocida en los bloques nocturnos, 5 de 24
sobre el nivel nominal (Cap_3, sec:prop-bootstrap y sec:prop-limitaciones).

Salidas:
  Resultados/AST-CC/refinamiento.txt
  Resultados/AST-CC/hojas_{periodo}_{base}.csv
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.estadisticos import esperados  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/AST-CC"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261004
DELTAS = [0.001, 0.0025, 0.005, 0.01, 0.02]
DELTA_DETALLE = 0.005
FRANJAS = [f"{h:02d}-{h + 2:02d}" for h in range(0, 24, 2)]


def descriptores(res, O, zonas):
    """Una fila por hoja con su nivel de actividad relativo respecto del padre."""
    E_global = esperados(O)
    filas = []

    def visitar(nodo, E_padre):
        if nodo.es_hoja:
            L = np.ix_(nodo.filas, nodo.cols)
            filas.append({
                "profundidad": nodo.profundidad, "tipo": nodo.estado,
                "n_regiones": len(nodo.filas), "n_intervalos": len(nodo.cols),
                "N": int(nodo.N), "I_corr": nodo.I_corr,
                "actividad_global": nodo.N / E_global[L].sum(),
                "actividad_padre": nodo.N / E_padre[L].sum(),
                "regiones": " ".join(str(zonas[i]) for i in nodo.filas),
                "intervalos": " ".join(FRANJAS[j] for j in nodo.cols),
            })
            return
        E = np.zeros_like(O)
        E[np.ix_(nodo.filas, nodo.cols)] = esperados(O[np.ix_(nodo.filas, nodo.cols)])
        for h in nodo.hijos:
            visitar(h, E)

    for raiz in res.raices:
        visitar(raiz, E_global)
    return pd.DataFrame(filas)


def main():
    zonas = pd.read_csv(DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    L = ["AST-CC sobre los datos reales (nula: bootstrap de días; alfa = 0.05)", ""]
    for periodo in ("weekday", "holiday"):
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        O = X.sum(0)
        for base in ("C5", "C2"):
            rho = np.load(DATOS / f"AST-CC/{periodo}_{base}_rho.npy")
            tau = np.load(DATOS / f"AST-CC/{periodo}_{base}_tau.npy")
            L.append(f"{periodo}, partición base {base}")
            L.append(f"  {'delta':>7} {'hojas':>5} {'homog.':>6} {'no loc.':>7} {'triv.':>5} {'inalc.':>6}  "
                     f"{'fil':>3} {'col':>3} {'conj':>4} {'prof.':>5}  {'(iii) expl.':>11} {'err. cor.':>9} "
                     f"{'B máx':>6} {'seg':>5}")
            for delta in DELTAS:
                rng = np.random.default_rng(SEMILLA)
                res = ASTCC(Parametros(delta=delta), rng).ajustar(X, rho, tau)
                hojas = Counter(h.estado for h in res.hojas())
                cortes = Counter(n.corte for n in res.internos())
                d = res.descomposicion(O)
                iii = d["explicada"] + d["residual"]
                L.append(f"  {delta:>7g} {len(res.hojas()):>5} {hojas['homogenea']:>6} {hojas['no_localizada']:>7} "
                         f"{hojas['trivial']:>5} {hojas['inalcanzable']:>6}  {cortes['fil']:>3} {cortes['col']:>3} "
                         f"{cortes['conj']:>4} {max(n.profundidad for n in res.hojas()):>5}  "
                         f"{d['explicada'] / iii:>11.1%} {d['error'] / d['total']:>9.0e} "
                         f"{max(n.B_usadas for n in res.nodos()):>6} {res.segundos:>5.1f}")
                print(L[-1], flush=True)
                if delta == DELTA_DETALLE:
                    descriptores(res, O, zonas).to_csv(SALIDA / f"hojas_{periodo}_{base}.csv", index=False)
            L.append("")
    texto = "\n".join(L)
    print("\n" + texto)
    (SALIDA / "refinamiento.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
