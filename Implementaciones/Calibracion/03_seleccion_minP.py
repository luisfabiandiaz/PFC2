"""
E2. Corrección por la búsqueda del mejor corte (ajuste min-P)
==============================================================
Pregunta (Cap_3.tex, sec:prop-minP): si se elige el corte que maximiza G^2_{B,s}
y se compara con la chi^2 de un corte fijado de antemano, ¿cuánto se infla el
error de tipo I? ¿Lo corrige repetir la búsqueda completa en cada réplica nula?

Bajo la nula de Poisson se generan bloques independientes con los tamaños y las
marginales de los bloques base reales de ST-CC (laborable), y para cada uno:
  ingenuo   max_s G^2_{B,s} frente a chi^2 con los gl de un corte
            (|J|-1 para filas, |I|-1 para columnas)
  min-P     max_s G^2_{B,s} frente al máximo en B réplicas de Patefield,
            repitiendo la búsqueda en cada una

Salidas: Resultados/Calibracion/E2_seleccion_minP.txt
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.cortes import mejor_corte_columnas, mejor_corte_filas  # noqa: E402
from astcc.estadisticos import esperados, pvalor_simulado  # noqa: E402
from astcc.nulas import replicas_poisson  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/Calibracion"

SEMILLA = 20261001
R_EXT = 500
B_PERM = 199
ALFA = 0.05


def estadistico(T, tipo, rng):
    """Máximo de G^2_{B,s} sobre los cortes del tipo dado, para un lote (B, m, n)."""
    if tipo == "filas":
        return mejor_corte_filas(T, rng)[0]
    return mejor_corte_columnas(T)[0]


def main():
    rng = np.random.default_rng(SEMILLA)
    X = np.load(DATOS / "Calibracion/conteos_diarios_weekday.npy")
    S = X.sum(0)
    rho = np.load(DATOS / "ST-CC/weekday_row_clusters.npy")
    tau = np.load(DATOS / "ST-CC/weekday_col_clusters.npy")

    se = np.sqrt(ALFA * (1 - ALFA) / R_EXT)
    lineas = [f"E2. Error de tipo I del mejor corte, ingenuo frente a min-P (nula de Poisson, nivel {ALFA})",
              f"    ({R_EXT} bloques nulos por fila, B = {B_PERM}; error Monte Carlo ~ {se:.3f})", "",
              f"{'bloque':<7} {'tamaño':>6} {'tipo':<9} {'candidatos':>10}  {'ingenuo':>8} {'min-P':>7}"]
    vistos = set()
    for g in np.unique(rho):
        for h in np.unique(tau):
            I, J = np.where(rho == g)[0], np.where(tau == h)[0]
            if len(I) < 2 or len(J) < 2 or (len(I), len(J)) in vistos:
                continue
            vistos.add((len(I), len(J)))          # un bloque por cada tamaño distinto
            E = esperados(S[np.ix_(I, J)])
            for tipo, gl, cand in (("filas", len(J) - 1, 2 ** (len(I) - 1) - 1),
                                   ("columnas", len(I) - 1, 2 ** (len(J) - 1) - 1)):
                if cand < 1:
                    continue
                ingenuo = minp = 0
                for _ in range(R_EXT):
                    O = rng.poisson(E).astype(float)
                    t = float(estadistico(O[None], tipo, rng)[0])
                    ingenuo += chi2.sf(t, gl) <= ALFA
                    t_b = estadistico(replicas_poisson(O, B_PERM, rng), tipo, rng)
                    minp += pvalor_simulado(t, t_b) <= ALFA
                lineas.append(f"({g},{h}){'':<2} {len(I):>3}x{len(J):<2} {tipo:<9} {cand:>10}  "
                              f"{ingenuo / R_EXT:>8.3f} {minp / R_EXT:>7.3f}")
                print(lineas[-1], flush=True)
    texto = "\n".join(lineas)
    (SALIDA / "E2_seleccion_minP.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
