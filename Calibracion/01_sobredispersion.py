"""
Estimación de la sobredispersión φ̂ con las réplicas diarias
=============================================================
Propuesta (Cap_3.tex, sec:prop-sobredispersion): para cada celda con
actividad suficiente se calcula varianza / media de sus conteos diarios y
φ̂ es la mediana de esos cocientes.

La prueba de AST-CC se hace sobre la matriz SUMADA de D días,
S_ij = sum_d X_dij. Un factor que multiplica todo un día (lluvia, feriado,
huelga) inflaría var/media en cada celda, pero no altera la independencia
fila-columna de S dentro de un bloque. Por eso se comparan cuatro
estimadores, de más a menos ingenuo:

  A. Propuesta: mediana por celda de var/media (celdas con media >= 5).
  B. Pearson X^2/gl del modelo [celda]: E_dij = media diaria de la celda.
     Es la versión global de A.
  C. Pearson X^2/gl del modelo [D][ZT]: E_dij = X_d.. X_.ij / X...
     Quita el volumen total de cada día.
  D. Pearson X^2/gl del modelo [DZ][DT][ZT] (sin interacción triple, IPF).
     Quita además los efectos día-zona y día-horario; lo que queda es la
     variación extra-Poisson propia de cada celda.

Después se evalúa el efecto práctico: G^2 de cada bloque base de ST-CC
sobre la matriz de conteos totales, con y sin la corrección G^2 / φ̂.

Salidas:
  DATA/procesados/Calibracion/conteos_diarios_{periodo}.npy   (D x 66 x 12)
  Resultados/Calibracion/sobredispersion.txt
  Resultados/Calibracion/sobredispersion_media_varianza.png
"""

import sys

import duckdb
import matplotlib
import numpy as np
from pathlib import Path
from scipy.stats import chi2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parents[1]      # Implementaciones/
sys.path.insert(0, str(RAIZ))
from astcc.estadisticos import G2, div  # noqa: E402
from astcc.nulas import ajuste_sin_interaccion_triple  # noqa: E402

CRUDOS_DIR = RAIZ / "DATA/crudos"
STCC_DIR = RAIZ / "DATA/procesados/ST-CC"
SALIDA_DATOS = RAIZ / "DATA/procesados/Calibracion"
SALIDA_RES = RAIZ / "Resultados/Calibracion"
SALIDA_DATOS.mkdir(parents=True, exist_ok=True)
SALIDA_RES.mkdir(parents=True, exist_ok=True)

# Mismos días que ST-CC/01_matriz_y_coclustering.py
PERIODOS = {
    "weekday": [1, 2, 7, 8, 9, 14, 15, 16, 21, 28, 29, 30],
    "holiday": [4, 5, 11, 12, 18, 19, 23, 25, 26],
}
N_BINS = 12
MEDIA_MIN = 5.0     # "actividad suficiente": media diaria mínima de la celda
TRAMOS = [(5, 50), (50, 500), (500, np.inf)]    # tramos de media diaria
ALFA = 0.05


# ----------------------------------------------------------------
# 1. Conteos diarios por zona y bloque de 2 h
# ----------------------------------------------------------------
def conteos_diarios():
    con = duckdb.connect()
    zonas = con.execute(f"SELECT * FROM '{CRUDOS_DIR / 'taxi_zone_lookup.csv'}'").fetchdf()
    manhattan = zonas[zonas["Borough"] == "Manhattan"]
    islas = manhattan["Zone"].str.contains("Governor|Ellis|Liberty", case=False, na=False)
    validas = sorted(manhattan.loc[~islas, "LocationID"].tolist())
    assert len(validas) == 66

    orden_stcc = np.loadtxt(STCC_DIR / "zone_order.csv", delimiter=",", skiprows=1, dtype=int)[:, 1]
    assert np.array_equal(orden_stcc, validas), "El orden de zonas no coincide con ST-CC"

    lista = ",".join(map(str, validas))
    df = con.execute(f"""
        WITH viajes AS (
            SELECT DOLocationID AS zona, tpep_dropoff_datetime AS ts
            FROM '{CRUDOS_DIR / 'yellow_tripdata_2017-11.parquet'}'
            UNION ALL
            SELECT DOLocationID AS zona, lpep_dropoff_datetime AS ts
            FROM '{CRUDOS_DIR / 'green_tripdata_2017-11.parquet'}'
        )
        SELECT zona,
               CAST(EXTRACT(day FROM ts) AS INTEGER)              AS dia,
               CAST(FLOOR(EXTRACT(hour FROM ts) / 2) AS INTEGER)  AS bin,
               COUNT(*)                                           AS n
        FROM viajes
        WHERE ts >= TIMESTAMP '2017-11-01' AND ts < TIMESTAMP '2017-12-01'
          AND zona IN ({lista})
        GROUP BY ALL
    """).fetchdf()

    fila = {z: i for i, z in enumerate(validas)}
    tensores = {}
    for periodo, dias in PERIODOS.items():
        X = np.zeros((len(dias), len(validas), N_BINS))
        pos = {d: k for k, d in enumerate(dias)}
        sub = df[df["dia"].isin(dias)]
        X[sub["dia"].map(pos).to_numpy(), sub["zona"].map(fila).to_numpy(), sub["bin"].to_numpy()] = sub["n"].to_numpy()
        tensores[periodo] = X
    return tensores


# ----------------------------------------------------------------
# 2. Estimadores de φ
# ----------------------------------------------------------------


def pearson(X, E, gl):
    m = E > 0
    return float(np.sum((X[m] - E[m]) ** 2 / E[m]) / gl)




def estimadores(X):
    D, Z, T = X.shape
    media = X.mean(0)
    var = X.var(0, ddof=1)
    activas = media >= MEDIA_MIN
    celdas = media > 0

    res = {}
    res["A"] = float(np.median(var[activas] / media[activas]))

    E0 = np.broadcast_to(media, X.shape)
    res["B"] = pearson(X, E0, celdas.sum() * (D - 1))

    E1 = np.einsum("d,zt->dzt", X.sum((1, 2)), X.sum(0)) / X.sum()
    res["C"] = pearson(X, E1, (D - 1) * (celdas.sum() - 1))

    E2 = ajuste_sin_interaccion_triple(X)
    res["D"] = pearson(X, E2, (D - 1) * (Z - 1) * (T - 1))

    # Versión por celda de D (misma idea que A, sobre residuos del modelo D)
    contrib = np.sum(div((X - E2) ** 2, E2), axis=0) / ((D - 1) * (Z - 1) * (T - 1) / (Z * T))
    res["D_mediana"] = float(np.median(contrib[activas]))

    # Función de varianza. Varianza residual por celda tras el modelo D, escalada
    # por los gl que le corresponden a cada celda para que sea comparable con var.
    var_res = np.sum((X - E2) ** 2, axis=0) / ((D - 1) * (Z - 1) * (T - 1) / (Z * T))
    lm = np.log(media[activas])
    for sufijo, v in (("", var), ("_res", var_res)):
        # Ley de potencia de Taylor: var = a μ^b (b = 1: φ constante; b = 2: BN dominada por μ²)
        b, log_a = np.polyfit(lm, np.log(v[activas]), 1)
        res["b" + sufijo], res["a" + sufijo] = float(b), float(np.exp(log_a))
        res["ecm_pot" + sufijo] = float(np.mean((np.log(v[activas]) - (log_a + b * lm)) ** 2))
        # φ constante con la misma medida de ajuste, para comparar
        log_phi = np.median(np.log(v[activas]) - lm)
        res["ecm_phi" + sufijo] = float(np.mean((np.log(v[activas]) - (log_phi + lm)) ** 2))
        # φ efectivo por tramo de volumen
        for lo, hi in TRAMOS:
            tramo = activas & (media >= lo) & (media < hi)
            res[f"φ[{lo},{hi})" + sufijo] = float(np.median(v[tramo] / media[tramo]))

    return res, media, var, activas


# ----------------------------------------------------------------
# 3. Efecto sobre las pruebas de los bloques base de ST-CC
# ----------------------------------------------------------------


def pruebas_bloques(S, filas, cols, phis):
    K = len(np.unique(filas)) * len(np.unique(cols))
    tabla = []
    for g in np.unique(filas):
        for h in np.unique(cols):
            B = S[np.ix_(filas == g, cols == h)]
            gl = (B.shape[0] - 1) * (B.shape[1] - 1)
            if gl == 0:
                continue
            g2 = G2(B)
            tabla.append({
                "bloque": f"({g},{h})", "tam": f"{B.shape[0]}x{B.shape[1]}", "N": B.sum(),
                "gl": gl, "G2": g2, "I": g2 / (2 * B.sum()),
                "p": {k: chi2.sf(g2 / phi, gl) for k, phi in phis.items()},
            })
    return tabla, K


# ----------------------------------------------------------------
# 4. Ejecución
# ----------------------------------------------------------------
def main():
    print("Leyendo los parquet y contando viajes por día (uno o dos minutos)...")
    tensores = conteos_diarios()
    nombres = {
        "A": "A. Mediana var/media por celda (Propuesta)",
        "B": "B. Pearson X²/gl, modelo [celda]",
        "C": "C. Pearson X²/gl, modelo [D][ZT] (quita el día)",
        "D": "D. Pearson X²/gl, modelo [DZ][DT][ZT]",
        "D_mediana": "D'. Mediana por celda del modelo D",
    }
    for sufijo, desc in (("", "var cruda"), ("_res", "var residual modelo D")):
        nombres.update({
            f"φ[{lo},{hi})" + sufijo: f"φ por tramo, media en [{lo:g},{hi:g}) ({desc})" for lo, hi in TRAMOS})
        nombres.update({
            "b" + sufijo: f"Taylor var = a μ^b: exponente b ({desc})",
            "a" + sufijo: f"Taylor var = a μ^b: coeficiente a ({desc})",
            "ecm_pot" + sufijo: f"ECM log-var, ley de potencia ({desc})",
            "ecm_phi" + sufijo: f"ECM log-var, φ constante ({desc})",
        })
    lineas = [f"Sobredispersión con réplicas diarias (celdas activas: media diaria >= {MEDIA_MIN:g})", ""]
    resultados, datos_fig = {}, {}

    for periodo, X in tensores.items():
        np.save(SALIDA_DATOS / f"conteos_diarios_{periodo}.npy", X)
        S = X.sum(0)
        # Control: la matriz de ST-CC es el promedio diario de estos mismos conteos
        M = np.load(STCC_DIR / f"{periodo}_matrix.npy")
        assert np.allclose(S / X.shape[0], M), f"Los conteos no reproducen {periodo}_matrix.npy"
        res, media, var, activas = estimadores(X)
        resultados[periodo] = res
        datos_fig[periodo] = (media, var, activas, res)
        lineas.append(f"{periodo}: D = {X.shape[0]} días, N = {int(S.sum()):,} viajes, "
                      f"{activas.sum()} de {activas.size} celdas activas")
    lineas.append("")
    lineas.append(f"{'Estimador':<58} {'weekday':>9} {'holiday':>9}")
    lineas.append("-" * 78)
    for k, nombre in nombres.items():
        lineas.append(f"{nombre:<58} {resultados['weekday'][k]:>9.2f} {resultados['holiday'][k]:>9.2f}")

    # Pruebas en los bloques base de ST-CC
    for periodo, X in tensores.items():
        S = X.sum(0)
        filas = np.load(STCC_DIR / f"{periodo}_row_clusters.npy")
        cols = np.load(STCC_DIR / f"{periodo}_col_clusters.npy")
        phis = {"Poisson (φ=1)": 1.0, "φ̂ A": resultados[periodo]["A"], "φ̂ D": resultados[periodo]["D"]}
        tabla, K = pruebas_bloques(S, filas, cols, phis)
        bonf = ALFA / K
        lineas += ["", f"Bloques base de ST-CC, {periodo} (K_R x K_T = {K}; χ² asintótica; "
                       f"Bonferroni α = {ALFA}/{K})", ""]
        lineas.append(f"{'bloque':<7} {'tam':>6} {'N':>8} {'gl':>4} {'G²':>10} {'G²/gl':>7} {'Î (nats)':>9}  "
                      + "  ".join(f"{k:>14}" for k in phis))
        for r in sorted(tabla, key=lambda r: -r["I"]):
            lineas.append(f"{r['bloque']:<7} {r['tam']:>6} {int(r['N']):>8} {r['gl']:>4} {r['G2']:>10.1f} "
                          f"{r['G2'] / r['gl']:>7.1f} {r['I']:>9.4f}  "
                          + "  ".join(f"{r['p'][k]:>14.2g}" for k in phis))
        lineas.append("rechazos (Bonferroni): " + ", ".join(
            f"{k}: {sum(r['p'][k] <= bonf for r in tabla)}/{len(tabla)}" for k in phis))

    texto = "\n".join(lineas)
    print(texto)
    (SALIDA_RES / "sobredispersion.txt").write_text(texto, encoding="utf-8")
    figura(datos_fig)


def figura(datos_fig):
    tinta, tinta2, azul, naranja, aqua = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True, facecolor="#fcfcfb")
    for ax, (periodo, (media, var, activas, res)) in zip(ejes, datos_fig.items()):
        ax.set_facecolor("#fcfcfb")
        m = media > 0
        ax.scatter(media[m & activas], var[m & activas], s=10, color=azul, alpha=0.55,
                   linewidths=0, label=f"celdas activas (media ≥ {MEDIA_MIN:g})")
        ax.scatter(media[m & ~activas], var[m & ~activas], s=10, color=tinta2, alpha=0.35,
                   linewidths=0, label="celdas poco activas")
        x = np.logspace(np.log10(media[m].min()), np.log10(media.max()), 100)
        ax.plot(x, x, color=tinta2, lw=1.5, ls="--", label="Poisson: var = media")
        ax.plot(x, res["A"] * x, color=naranja, lw=2, label="φ constante: var = φ̂_A · media")
        ax.plot(x, res["a"] * x ** res["b"], color=aqua, lw=2, label="ley de Taylor: var = a · media^b")
        ax.text(0.97, 0.04, f"φ̂_A = {res['A']:.1f}    b = {res['b']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", color=tinta2, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{periodo}", color=tinta, loc="left", fontsize=11)
        ax.set_xlabel("media diaria de la celda", color=tinta2)
        ax.grid(True, which="major", color="#e6e5e1", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=tinta2)
    ejes[0].set_ylabel("varianza entre días", color=tinta2)
    ejes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=tinta2)
    fig.tight_layout()
    fig.savefig(SALIDA_RES / "sobredispersion_media_varianza.png", dpi=150)


if __name__ == "__main__":
    main()
