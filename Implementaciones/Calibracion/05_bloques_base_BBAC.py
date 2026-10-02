"""
E4. ¿Es válido probar los bloques base que eligió BBAC_I con los mismos datos?
==============================================================================
Pregunta (Cap_3.tex, sec:prop-calibracion, "Alcance de la corrección"): la
partición base sale de BBAC_I aplicado a los mismos datos que después se prueban.
La búsqueda del corte se corrige con min-P, pero la selección del nodo no. ¿Infla
eso el error de tipo I de la prueba ómnibus en los bloques base?

Datos nulos de toda la matriz 66 x 12 (independencia global, así que todos los
bloques de cualquier partición cumplen H0), con dos generadores:
  Poisson     O ~ Poisson(E), E = independencia de la matriz real; prueba con Patefield
  semirreal   bootstrap multiplicativo de los días reales alrededor de E; prueba con
              el bootstrap de días corregido (la nula de referencia según E3)

Particiones comparadas:
  BBAC        BBAC_I (cgc) con los K de Liu et al. sobre los datos nulos
  aleatoria   etiquetas al azar con los mismos tamaños de grupo (sin selección)

Se compara la tasa de rechazo por bloque a los niveles 0.05 y 0.01 entre BBAC y la
partición aleatoria: si la selección no influye, ambas deben coincidir con el
nivel. FWER = proporción de réplicas con al menos un bloque rechazado a 0.05. Las
reglas de reparto por masa y de Bonferroni (niveles ~0.002) necesitarían miles de
réplicas por bloque; se evaluarán con el algoritmo completo.

Salidas: Resultados/Calibracion/E4_bloques_base.txt
"""

import logging
import sys
from pathlib import Path

import numpy as np
from cgc.coclustering import Coclustering

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.estadisticos import G2, pvalor_simulado  # noqa: E402
from astcc.nulas import replicas_poisson, replicas_semirreales  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
logging.disable(logging.WARNING)

DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/Calibracion"

SEMILLA = 20261001
R_EXT = 200
B_PERM = 199
ALFA = 0.05
K = {"weekday": (8, 4), "holiday": (6, 4)}


def bbac(O, kr, kt, semilla):
    np.random.seed(semilla)          # cgc usa el generador global de numpy
    r = Coclustering(O, kr, kt, max_iterations=100, conv_threshold=1e-5, nruns=10).run_with_threads(nthreads=4)
    return np.asarray(r.row_clusters), np.asarray(r.col_clusters)


def aleatoria(rho, tau, rng):
    return rng.permutation(rho), rng.permutation(tau)


def probar_bloques(dias, rho, tau, generador, rng):
    """p-valores y masas de los bloques con al menos 2 filas y 2 columnas."""
    S = dias.sum(0)
    res = []
    for g in np.unique(rho):
        for h in np.unique(tau):
            I, J = np.where(rho == g)[0], np.where(tau == h)[0]
            if len(I) < 2 or len(J) < 2:
                continue
            B = S[np.ix_(I, J)]
            t = float(G2(B))
            if generador == "Poisson":
                t_b = G2(replicas_poisson(B, B_PERM, rng))
            else:
                t_b = G2(replicas_semirreales(dias[:, I][:, :, J], B_PERM, rng, inflar=True))
            res.append((pvalor_simulado(t, t_b), B.sum()))
    return res


def main():
    rng = np.random.default_rng(SEMILLA)
    lineas = [f"E4. Prueba ómnibus en los bloques base bajo independencia global (nivel {ALFA}, {R_EXT} réplicas)",
              ""]
    for periodo, (kr, kt) in K.items():
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        S = X.sum(0)
        N = S.sum()
        E = np.outer(S.sum(1), S.sum(0)) / N
        # Días nulos semirreales de toda la matriz: misma media de independencia global
        dias_semi = replicas_semirreales(X, R_EXT, rng, diarias=True)
        lineas.append(f"{periodo} (K_R = {kr}, K_T = {kt})")
        lineas.append(f"  {'datos / partición':<22} {'bloques':>7}  {'tasa 0.05':>9} {'tasa 0.01':>9}  "
                      f"{'FWER 0.05':>9}")
        for generador in ("Poisson", "semirreal"):
            for particion in ("BBAC", "aleatoria"):
                n_bloques = rech = rech01 = fw_libre = 0
                for r in range(R_EXT):
                    if generador == "Poisson":
                        O = rng.poisson(E).astype(float)
                        dias = O[None]
                    else:
                        dias = dias_semi[r]
                        O = dias.sum(0)
                    rho, tau = bbac(O, kr, kt, SEMILLA + r)
                    if particion == "aleatoria":
                        rho, tau = aleatoria(rho, tau, rng)
                    res = probar_bloques(dias, rho, tau, generador, rng)
                    n_bloques += len(res)
                    rech += sum(p <= ALFA for p, _ in res)
                    rech01 += sum(p <= 0.01 for p, _ in res)
                    fw_libre += any(p <= ALFA for p, _ in res)
                lineas.append(f"  {generador + ' / ' + particion:<22} {n_bloques / R_EXT:>7.1f}  "
                              f"{rech / n_bloques:>9.3f} {rech01 / n_bloques:>9.3f}  {fw_libre / R_EXT:>9.3f}")
                print(lineas[-1], flush=True)
        lineas.append("")
    texto = "\n".join(lineas)
    (SALIDA / "E4_bloques_base.txt").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    main()
