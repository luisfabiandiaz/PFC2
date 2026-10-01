"""
E1. Exactitud de la prueba ómnibus bajo la nula de Poisson
============================================================
Pregunta (Cap_3.tex, sec:prop-calibracion): con conteos de Poisson independientes
dentro del bloque, ¿el p-valor por simulación de Patefield con la fórmula de
Phipson y Smyth mantiene el nivel nominal? ¿Y la aproximación chi^2, sobre todo
en bloques con pocos viajes?

Se generan tablas O_ij ~ Poisson(c a_i b_j) con efectos de fila y columna
desiguales (lognormales, como en los datos de movilidad) y distintos tamaños de
bloque y conteos esperados por celda.

Salidas: Resultados/Calibracion/E1_exactitud_omnibus.txt
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.estadisticos import G2, grados_libertad, pvalor_simulado  # noqa: E402
from astcc.nulas import replicas_poisson  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RAIZ / "Resultados/Calibracion"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261001
R_EXT = 4000      # tablas nulas por escenario
B_PERM = 199
NIVELES = (0.05, 0.01)
TAMANOS = [(2, 2), (4, 3), (7, 5), (13, 5), (15, 12)]
MEDIAS = [0.5, 2, 10, 100]      # conteo esperado medio por celda


def main():
    rng = np.random.default_rng(SEMILLA)
    se = np.sqrt(0.05 * 0.95 / R_EXT)
    lineas = [f"E1. Nivel real de la prueba ómnibus con datos de Poisson independientes",
              f"    ({R_EXT} tablas por escenario, B = {B_PERM}; error Monte Carlo a 0.05 ~ {se:.3f})", "",
              f"{'tamaño':>7} {'media/celda':>11}   {'Patefield 0.05':>14} {'0.01':>6}   {'chi2 0.05':>9} {'0.01':>6}"]
    for m, n in TAMANOS:
        for c in MEDIAS:
            rech = {("pat", a): 0 for a in NIVELES} | {("chi", a): 0 for a in NIVELES}
            validas = 0
            for _ in range(R_EXT):
                mu = np.outer(rng.lognormal(0, 1, m), rng.lognormal(0, 0.7, n))
                O = rng.poisson(c * mu / mu.mean()).astype(float)
                if (O.sum(1) > 0).sum() < 2 or (O.sum(0) > 0).sum() < 2:
                    continue        # tabla degenerada: no hay nada que probar
                validas += 1
                t = float(G2(O))
                p_pat = pvalor_simulado(t, G2(replicas_poisson(O, B_PERM, rng)))
                p_chi = chi2.sf(t, grados_libertad(m, n))
                for a in NIVELES:
                    rech[("pat", a)] += p_pat <= a
                    rech[("chi", a)] += p_chi <= a
            f = {k: v / validas for k, v in rech.items()}
            lineas.append(f"{m:>3}x{n:<3} {c:>11g}   {f[('pat', 0.05)]:>14.3f} {f[('pat', 0.01)]:>6.3f}   "
                          f"{f[('chi', 0.05)]:>9.3f} {f[('chi', 0.01)]:>6.3f}")
            print(lineas[-1], flush=True)
    texto = "\n".join(lineas)
    (SALIDA / "E1_exactitud_omnibus.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
