"""
E3. Error de tipo I de la prueba ómnibus con la variabilidad real de los días
==============================================================================
Pregunta (Cap_3.tex, sec:prop-sobredispersion): ¿qué distribución nula mantiene
el error de tipo I en su nivel nominal cuando los conteos tienen la
sobredispersión real de los datos de taxis?

Datos nulos semirreales: para cada bloque base de ST-CC se toma la media de
independencia de su tabla agregada y se le aplican las fluctuaciones
multiplicativas de D días reales remuestreados (astcc.nulas.replicas_semirreales).
Así H0 se cumple por construcción, pero el ruido es el de los datos.

Pruebas comparadas sobre cada réplica (todas por simulación con B réplicas):
  chi2       G^2 frente a chi^2 asintótica con (|I|-1)(|J|-1) gl
  Poisson    Patefield (nula de Poisson)
  phi cte    Patefield con t_b escalado por phi^ (versión original de la Propuesta)
  Taylor-C   nula gamma-Poisson con var = a mu^b estimada con la varianza cruda
  Taylor-R   ídem con la varianza residual del modelo [DZ][DT][ZT]
  Boot-días  bootstrap multiplicativo de los días de la propia réplica alrededor de
             su E^B (doble bootstrap: la prueba solo ve los D días de la réplica)
  Boot-c     ídem con la corrección de muestra pequeña sqrt(D / (D - 1))
  máx(B,T)   rechaza solo si rechazan Boot-c y Taylor-C (p = máximo de ambos)

Salidas: Resultados/Calibracion/E3_tipoI_semirreal.{txt,csv,png}
"""

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from astcc.estadisticos import G2, grados_libertad, pvalor_simulado  # noqa: E402
from astcc.nulas import (ajustar_taylor, phi_constante, replicas_poisson,  # noqa: E402
                         replicas_semirreales, replicas_taylor, varianzas_diarias)

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/Calibracion"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261001
R_EXT = 300        # réplicas semirreales por bloque (datos nulos)
B_PERM = 199       # réplicas de cada prueba
ALFA = 0.05
METODOS = ["chi2", "Poisson", "phi cte", "Taylor-C", "Taylor-R", "Boot-días", "Boot-c", "máx(B,T)"]


def p_valores(Xr, params, rng):
    """p-valores de las pruebas para una réplica con días Xr (D, m, n)."""
    D = Xr.shape[0]
    S = Xr.sum(0)
    t_obs = float(G2(S))
    nu = grados_libertad(*S.shape)
    t_pois = G2(replicas_poisson(S, B_PERM, rng))
    p = {
        "chi2": chi2.sf(t_obs, nu),
        "Poisson": pvalor_simulado(t_obs, t_pois),
        "phi cte": pvalor_simulado(t_obs, params["phi"] * t_pois),
        "Taylor-C": pvalor_simulado(t_obs, G2(replicas_taylor(S, B_PERM, *params["cruda"], D, rng))),
        "Taylor-R": pvalor_simulado(t_obs, G2(replicas_taylor(S, B_PERM, *params["residual"], D, rng))),
        "Boot-días": pvalor_simulado(t_obs, G2(replicas_semirreales(Xr, B_PERM, rng))),
        "Boot-c": pvalor_simulado(t_obs, G2(replicas_semirreales(Xr, B_PERM, rng, inflar=True))),
    }
    p["máx(B,T)"] = max(p["Boot-c"], p["Taylor-C"])
    return p, t_obs


def main():
    rng = np.random.default_rng(SEMILLA)
    filas_csv = []
    inicio = time.time()
    for periodo in ("weekday", "holiday"):
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        D = X.shape[0]
        media, var, var_res = varianzas_diarias(X)
        params = {"phi": phi_constante(media, var),
                  "cruda": ajustar_taylor(media, var),
                  "residual": ajustar_taylor(media, var_res)}
        print(f"{periodo}: D={D}, phi^={params['phi']:.2f}, "
              f"Taylor cruda a={params['cruda'][0]:.2f} b={params['cruda'][1]:.2f}, "
              f"residual a={params['residual'][0]:.2f} b={params['residual'][1]:.2f}")

        rho = np.load(DATOS / f"ST-CC/{periodo}_row_clusters.npy")
        tau = np.load(DATOS / f"ST-CC/{periodo}_col_clusters.npy")
        for g in np.unique(rho):
            for h in np.unique(tau):
                I, J = np.where(rho == g)[0], np.where(tau == h)[0]
                if len(I) < 2 or len(J) < 2:
                    continue
                Xb = X[:, I][:, :, J]
                nulos = replicas_semirreales(Xb, R_EXT, rng, diarias=True)
                rechazos = {k: 0 for k in METODOS}
                t_obs_media = 0.0
                for Xr in nulos:
                    p, t = p_valores(Xr, params, rng)
                    t_obs_media += t / R_EXT
                    for k in METODOS:
                        rechazos[k] += p[k] <= ALFA
                fila = {"periodo": periodo, "bloque": f"({g},{h})", "filas": len(I), "cols": len(J),
                        "N": int(Xb.sum()), "gl": grados_libertad(len(I), len(J)),
                        "G2_nulo_medio": t_obs_media}
                fila.update({k: rechazos[k] / R_EXT for k in METODOS})
                filas_csv.append(fila)
                print(f"  bloque {fila['bloque']:<6} {len(I)}x{len(J)} N={fila['N']:>7}  "
                      + "  ".join(f"{k}={fila[k]:.2f}" for k in METODOS), flush=True)

    df = pd.DataFrame(filas_csv)
    df.to_csv(SALIDA / "E3_tipoI_semirreal.csv", index=False)
    resumen(df, time.time() - inicio)
    figura(df)


def resumen(df, segundos):
    se = np.sqrt(ALFA * (1 - ALFA) / R_EXT)
    lineas = [f"E3. Error de tipo I con datos nulos semirreales (nivel {ALFA}; {R_EXT} réplicas por bloque,",
              f"    B = {B_PERM} réplicas por prueba; error Monte Carlo por bloque ~ {se:.3f})", ""]
    for periodo, d in df.groupby("periodo", sort=False):
        lineas.append(f"{periodo} ({len(d)} bloques)")
        lineas.append(f"  {'método':<10} {'media':>7} {'mediana':>8} {'mín':>6} {'máx':>6}  bloques > {ALFA}+2se")
        for k in METODOS:
            v = d[k]
            lineas.append(f"  {k:<10} {v.mean():>7.3f} {v.median():>8.3f} {v.min():>6.2f} {v.max():>6.2f}  "
                          f"{int((v > ALFA + 2 * se).sum()):>3}/{len(d)}")
        lineas.append("")
    lineas.append(f"Tiempo total: {segundos / 60:.1f} min")
    texto = "\n".join(lineas)
    print("\n" + texto)
    (SALIDA / "E3_tipoI_semirreal.txt").write_text(texto, encoding="utf-8")


def figura(df):
    tinta2, azul = "#52514e", "#2a78d6"
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True, facecolor="#fcfcfb")
    for ax, (periodo, d) in zip(ejes, df.groupby("periodo", sort=False)):
        ax.set_facecolor("#fcfcfb")
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(d))
        for x, k in enumerate(METODOS):
            ax.scatter(x + jit, d[k], s=14, color=azul, alpha=0.6, linewidths=0)
            ax.plot([x - 0.25, x + 0.25], [d[k].mean()] * 2, color="#0b0b0b", lw=2)
        ax.axhline(ALFA, color=tinta2, ls="--", lw=1.2)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(0, 1.05)
        marcas = [0, 0.01, 0.05, 0.2, 0.5, 1]
        ax.set_yticks(marcas, [f"{v:g}" for v in marcas])
        ax.minorticks_off()
        ax.set_xticks(range(len(METODOS)), METODOS)
        ax.set_title(periodo, loc="left", fontsize=11)
        ax.grid(True, axis="y", color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=tinta2)
    ejes[0].set_ylabel("error de tipo I por bloque", color=tinta2)
    ejes[0].text(len(METODOS) - 0.5, ALFA * 1.15, f"nominal {ALFA}", color=tinta2, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(SALIDA / "E3_tipoI_semirreal.png", dpi=150)


if __name__ == "__main__":
    main()
