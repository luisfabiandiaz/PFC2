"""
E5. Periodo feriado: bootstrap de días estratificado por tipo de día
====================================================================
Cap_3.tex, sec:prop-sobredispersion, conclusión 4: en el periodo feriado el
bootstrap de días resultó algo liberal (E3), y se propone remuestrear los sábados
entre los sábados y los domingos entre los domingos.

La matriz real de feriados tiene siempre la misma composición: 4 sábados,
4 domingos y el Día de Acción de Gracias. La nula semirreal de E3 remuestreaba los
9 días sin estratos, así que cada réplica tenía una composición distinta y más
variabilidad que los datos reales. Aquí el generador de datos nulos también se
estratifica, para reproducir el diseño real, y se comparan:
  Boot-c       bootstrap corregido sin estratos (nula de referencia de E3)
  Boot-estr    bootstrap corregido estratificado por tipo de día
  Taylor-C     nula paramétrica con la varianza cruda
  Poisson      Patefield

Salidas: Resultados/Calibracion/E5_feriado_estratificado.txt
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.estadisticos import G2, pvalor_simulado  # noqa: E402
from astcc.nulas import (ajustar_taylor, replicas_poisson, replicas_semirreales,  # noqa: E402
                         replicas_taylor, varianzas_diarias)

sys.stdout.reconfigure(encoding="utf-8")
DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/Calibracion"

SEMILLA = 20261005
R_EXT = 300
B_PERM = 199
ALFA = 0.05
DIAS_FERIADO = [4, 5, 11, 12, 18, 19, 23, 25, 26]       # como en ST-CC/01
METODOS = ["Poisson", "Taylor-C", "Boot-c", "Boot-estr"]


def tipo_de_dia(d):
    """0 = sábado, 1 = domingo, 2 = otro (Día de Acción de Gracias)."""
    wd = dt.date(2017, 11, d).weekday()
    return {5: 0, 6: 1}.get(wd, 2)


def main():
    rng = np.random.default_rng(SEMILLA)
    estratos = np.array([tipo_de_dia(d) for d in DIAS_FERIADO])
    X = np.load(DATOS / "Calibracion/conteos_diarios_holiday.npy")
    D = X.shape[0]
    media, var, _ = varianzas_diarias(X)
    cruda = ajustar_taylor(media, var)
    rho = np.load(DATOS / "ST-CC/holiday_row_clusters.npy")
    tau = np.load(DATOS / "ST-CC/holiday_col_clusters.npy")

    se = np.sqrt(ALFA * (1 - ALFA) / R_EXT)
    L = [f"E5. Periodo feriado con datos nulos estratificados (estratos por día: {estratos.tolist()};",
         f"    {R_EXT} réplicas por bloque, B = {B_PERM}; error Monte Carlo por bloque ~ {se:.3f})", "",
         f"  {'bloque':<7} {'tamaño':>6} {'N':>7}  " + "  ".join(f"{k:>9}" for k in METODOS)]
    tasas = {k: [] for k in METODOS}
    for g in np.unique(rho):
        for h in np.unique(tau):
            I, J = np.where(rho == g)[0], np.where(tau == h)[0]
            if len(I) < 2 or len(J) < 2:
                continue
            Xb = X[:, I][:, :, J]
            nulos = replicas_semirreales(Xb, R_EXT, rng, diarias=True, inflar=True, estratos=estratos)
            rech = {k: 0 for k in METODOS}
            for Xr in nulos:
                S = Xr.sum(0)
                t = float(G2(S))
                p = {
                    "Poisson": pvalor_simulado(t, G2(replicas_poisson(S, B_PERM, rng))),
                    "Taylor-C": pvalor_simulado(t, G2(replicas_taylor(S, B_PERM, *cruda, D, rng))),
                    "Boot-c": pvalor_simulado(t, G2(replicas_semirreales(Xr, B_PERM, rng, inflar=True))),
                    "Boot-estr": pvalor_simulado(t, G2(replicas_semirreales(
                        Xr, B_PERM, rng, inflar=True, estratos=estratos))),
                }
                for k in METODOS:
                    rech[k] += p[k] <= ALFA
            for k in METODOS:
                tasas[k].append(rech[k] / R_EXT)
            L.append(f"  ({g},{h}){'':<2} {len(I):>3}x{len(J):<2} {int(Xb.sum()):>7}  "
                     + "  ".join(f"{rech[k] / R_EXT:>9.3f}" for k in METODOS))
            print(L[-1], flush=True)
    L += ["", f"  {'media':<23}" + "  ".join(f"{np.mean(tasas[k]):>9.3f}" for k in METODOS),
          f"  {'mediana':<23}" + "  ".join(f"{np.median(tasas[k]):>9.3f}" for k in METODOS),
          f"  {'máximo':<23}" + "  ".join(f"{np.max(tasas[k]):>9.3f}" for k in METODOS),
          f"  {'bloques > 0.05 + 2se':<23}" + "  ".join(
              f"{int(np.sum(np.array(tasas[k]) > ALFA + 2 * se)):>6}/{len(tasas[k])}" for k in METODOS)]
    texto = "\n".join(L)
    (SALIDA / "E5_feriado_estratificado.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)


if __name__ == "__main__":
    main()
