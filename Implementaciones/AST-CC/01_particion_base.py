"""
Partición base de AST-CC: BBAC_I con las bases C2 y C5 sobre los datos reales
=============================================================================
Cap_3.tex, Algoritmo 1, línea 1: (rho, tau) <- BBAC_I(O, K_R, K_T) con base C5
(Cap_2.tex, sec:stcc-base; la base C2 se conserva como variante).

Para cada periodo (laborable y feriado) y cada base se ejecuta astcc.bbac con
los K_R y K_T de Liu et al. (2021) sobre la matriz de CONTEOS totales, y se
compara con la partición que produjo cgc en ST-CC/01 (sobre promedios diarios;
la partición no depende de la escala, sec:escala). Se reporta:
  - D_I(O || O^) de cada partición con cada base
  - acuerdo entre particiones (índice de Rand ajustado, ARI)
  - descomposición (i) + (ii) + (iii) de la Proposición de descomposición
  - cuántas de las ejecuciones alcanzan el óptimo encontrado (estabilidad)

Salidas:
  DATA/procesados/AST-CC/{periodo}_{base}_{rho,tau}.npy
  Resultados/AST-CC/particion_base.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.bbac import bbac, descomposicion, divergencia  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
SALIDA_DATOS = DATOS / "AST-CC"
SALIDA_RES = RAIZ / "Resultados/AST-CC"
SALIDA_DATOS.mkdir(parents=True, exist_ok=True)
SALIDA_RES.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261004
NRUNS = 200                    # como Liu et al. (2021)
K = {"weekday": (8, 4), "holiday": (6, 4)}


def main():
    rng = np.random.default_rng(SEMILLA)
    L = [f"Partición base de AST-CC con BBAC_I ({NRUNS} inicializaciones, matriz de conteos)", ""]
    for periodo, (kr, kt) in K.items():
        O = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy").sum(0)
        particiones = {"cgc (ST-CC/01)": (np.load(DATOS / f"ST-CC/{periodo}_row_clusters.npy"),
                                          np.load(DATOS / f"ST-CC/{periodo}_col_clusters.npy"))}
        L.append(f"{periodo}: {O.shape[0]} x {O.shape[1]}, N = {int(O.sum()):,}, K_R = {kr}, K_T = {kt}")
        for base in ("C2", "C5"):
            t = time.time()
            r = bbac(O, kr, kt, base=base, nruns=NRUNS, rng=rng)
            optimo = np.isclose(r.divergencias, r.divergencia, rtol=1e-9).mean()
            L.append(f"  BBAC {base}: D_I = {r.divergencia:,.1f}; {optimo:.0%} de las ejecuciones "
                     f"alcanzan el óptimo; {len(np.unique(np.round(r.divergencias, 3)))} óptimos locales "
                     f"distintos ({time.time() - t:.1f} s)")
            np.save(SALIDA_DATOS / f"{periodo}_{base}_rho.npy", r.rho)
            np.save(SALIDA_DATOS / f"{periodo}_{base}_tau.npy", r.tau)
            particiones[f"BBAC {base}"] = (r.rho, r.tau)

        L.append("")
        L.append(f"  {'partición':<16} {'D_I base C2':>14} {'D_I base C5':>14}   "
                 f"{'(i) volumen':>12} {'(ii) marg.':>11} {'(iii) local':>12}")
        for nombre, (rho, tau) in particiones.items():
            d = descomposicion(O, rho, tau, kr, kt)
            L.append(f"  {nombre:<16} {d['total']:>14,.1f} {divergencia(O, rho, tau, 'C5', kr, kt):>14,.1f}   "
                     f"{d['i']:>12,.1f} {d['ii']:>11,.1f} {d['iii']:>12,.1f}")
        L.append("")
        nombres = list(particiones)
        L.append(f"  ARI filas / columnas")
        for a in range(len(nombres)):
            for b in range(a + 1, len(nombres)):
                (r1, t1), (r2, t2) = particiones[nombres[a]], particiones[nombres[b]]
                L.append(f"    {nombres[a]:<16} vs {nombres[b]:<16} {ari(r1, r2):>6.3f} / {ari(t1, t2):.3f}")
        L.append("")

    texto = "\n".join(L)
    print(texto)
    (SALIDA_RES / "particion_base.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
