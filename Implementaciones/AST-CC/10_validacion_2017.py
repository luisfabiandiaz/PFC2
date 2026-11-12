"""
¿Qué separan los cortes de AST-CC y se repiten fuera de noviembre?
==================================================================
El árbol de AST-CC/07 (partición base C5, delta = 0.005) se ajusta con los días de
noviembre de 2017. Dentro de un bloque el modelo supone independencia: todas las regiones
reparten sus llegadas entre los intervalos del bloque en la misma proporción. Un corte de
filas separa las regiones en grupos con repartos distintos.

Para cada corte se calcula, con los datos de noviembre y con los otros once meses de 2017
(que el árbol no vio), la cuota de llegadas de cada grupo en cada intervalo del bloque y la
información mutua entre grupo e intervalo en la tabla colapsada (nats por observación; con
los datos de noviembre es la ganancia del corte sin corregir). Si las cuotas y la información
se mantienen, el corte describe estructura estable y no ruido del mes.

Los grupos de regiones son los que eligió el árbol con noviembre: no se vuelve a ajustar el
árbol con los otros meses.

Días del resto de 2017 (sin noviembre):
  weekday  martes, miércoles y jueves
  holiday  sábados y domingos
menos los días atípicos de ATIPICOS (festivos, nevadas y las fiestas de fin de año).

Entradas: DATA/crudos/{yellow,green}_tripdata_2017-MM.parquet (los doce meses; ver README).
          Solo hacen falta la primera vez: el tensor diario se guarda en
          DATA/procesados/AST-CC/conteos_diarios_2017.npy (365 x 66 x 12).
Salidas:  Resultados/AST-CC/validacion_2017.txt, validacion_2017.csv y validacion_2017.tex
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("vis07", RAIZ / "AST-CC/07_visualizacion.py")
vis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vis)

DATOS, SALIDA, CRUDOS = vis.DATOS, vis.SALIDA, RAIZ / "DATA/crudos"
TENSOR = DATOS / "AST-CC/conteos_diarios_2017.npy"
DIAS_SEMANA = {"weekday": [1, 2, 3], "holiday": [5, 6]}          # lunes = 0
ATIPICOS = ["2017-01-01",                                         # Año Nuevo
            "2017-02-09", "2017-03-14", "2017-03-15",             # nevadas
            "2017-07-04",                                         # 4 de julio
            *[f"2017-12-{d}" for d in range(21, 32)]]             # fiestas de fin de año


def tensor_2017(zonas):
    """Llegadas por día del año, zona e intervalo de 2 h. Se calcula una vez con duckdb."""
    if TENSOR.exists():
        return np.load(TENSOR)
    import duckdb
    faltan = [f"{c}_tripdata_2017-{m:02d}.parquet" for m in range(1, 13) for c in ("yellow", "green")
              if not (CRUDOS / f"{c}_tripdata_2017-{m:02d}.parquet").exists()]
    if faltan:
        sys.exit("Faltan datos crudos de 2017 (ver README): " + ", ".join(faltan))
    df = duckdb.sql(f"""
        WITH viajes AS (
            SELECT DOLocationID AS zona, tpep_dropoff_datetime AS ts
            FROM read_parquet('{(CRUDOS / "yellow_tripdata_2017-*.parquet").as_posix()}', union_by_name = true)
            UNION ALL
            SELECT DOLocationID AS zona, lpep_dropoff_datetime AS ts
            FROM read_parquet('{(CRUDOS / "green_tripdata_2017-*.parquet").as_posix()}', union_by_name = true)
        )
        SELECT zona, dayofyear(ts) AS dia, hour(ts) // 2 AS bin, COUNT(*) AS n
        FROM viajes
        WHERE ts >= TIMESTAMP '2017-01-01' AND ts < TIMESTAMP '2018-01-01'
          AND zona IN ({",".join(map(str, zonas))})
        GROUP BY ALL
    """).df()
    fila = {z: i for i, z in enumerate(zonas)}
    X = np.zeros((365, len(zonas), 12), dtype=np.int32)
    X[df["dia"].to_numpy() - 1, df["zona"].map(fila).to_numpy(), df["bin"].to_numpy()] = df["n"].to_numpy()
    np.save(TENSOR, X)
    return X


def informacion(T):
    """Información mutua de una tabla de conteos, en nats por observación."""
    P = T / T.sum()
    E = P.sum(1, keepdims=True) * P.sum(0, keepdims=True)
    return float(np.sum(P[P > 0] * np.log(P[P > 0] / E[P > 0])))


def main():
    zonas = pd.read_csv(DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    Y = tensor_2017(zonas.tolist())
    fechas = pd.date_range("2017-01-01", "2017-12-31")
    usable = ~fechas.isin(pd.to_datetime(ATIPICOS)) & (fechas.month != 11)

    L = ["Cortes de AST-CC (noviembre de 2017, base C5, delta = 0.005) y su repetición en el resto de 2017",
         "cuota = fracción de las llegadas del grupo que cae en cada intervalo del bloque",
         "I = información mutua entre grupo e intervalo en la tabla colapsada (nats por observación)", ""]
    filas_csv, filas_tex = [], []
    for periodo in ("weekday", "holiday"):
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        O = X.sum(0)
        rho = np.load(DATOS / f"AST-CC/{periodo}_C5_rho.npy")
        tau = np.load(DATOS / f"AST-CC/{periodo}_C5_tau.npy")
        res = ASTCC(Parametros(delta=vis.DELTA), np.random.default_rng(vis.SEMILLA)).ajustar(X, rho, tau)
        dias = usable & np.isin(fechas.dayofweek, DIAS_SEMANA[periodo])
        O2 = Y[dias].sum(0).astype(float)
        orden_g, orden_h = vis.disposicion_global(O, rho, tau)[:2]
        nombre_g = {g: f"R{i + 1}" for i, g in enumerate(orden_g)}
        nombre_h = {h: f"T{i + 1}" for i, h in enumerate(orden_h)}

        L += [f"{periodo}: {len(res.internos())} cortes; noviembre D = {len(X)} días, N = {int(O.sum()):,}; "
              f"resto de 2017 D = {int(dias.sum())} días, N = {int(O2.sum()):,}", ""]
        for raiz in res.raices:
            bloque = f"{nombre_g[rho[raiz.filas[0]]]}-{nombre_h[tau[raiz.cols[0]]]}"
            for n in raiz.recorrer():
                if n.es_hoja:
                    continue
                franjas = [vis.FRANJAS[j] for j in n.cols]
                T1 = np.array([O[np.ix_(h.filas, n.cols)].sum(0) if n.corte == "fil" else
                               O[np.ix_(n.filas, h.cols)].sum(1) for h in n.hijos])
                T2 = np.array([O2[np.ix_(h.filas, n.cols)].sum(0) if n.corte == "fil" else
                               O2[np.ix_(n.filas, h.cols)].sum(1) for h in n.hijos])
                if n.corte != "fil" or len(n.hijos) != 2:
                    L += [f"  {bloque}: corte de {n.corte} (profundidad {n.profundidad}); I nov = {informacion(T1):.4f}, "
                          f"I resto = {informacion(T2):.4f}", ""]
                    continue
                c1, c2 = T1 / T1.sum(1, keepdims=True), T2 / T2.sum(1, keepdims=True)
                j = int(np.argmax(np.round(np.abs(c1[0] - c1[1]), 9)))   # intervalo de mayor contraste (el primero si empatan)
                a, b = (0, 1) if c1[0, j] >= c1[1, j] else (1, 0)    # a = grupo con más cuota en j
                L.append(f"  {bloque} (profundidad {n.profundidad}): {len(n.filas)} regiones x [{' '.join(franjas)}]; "
                         f"I nov = {informacion(T1):.4f}, I resto = {informacion(T2):.4f}")
                L.append(f"    {'':<28} " + " ".join(f"{f:>7}" for f in franjas) + "   | resto de 2017: "
                         + " ".join(f"{f:>7}" for f in franjas))
                for k in (a, b):
                    h = n.hijos[k]
                    L.append(f"    {len(h.filas):>2} regiones, {T1[k].sum() / len(X) / len(h.filas):>5.0f}/día/reg  "
                             + " ".join(f"{v:>7.1%}" for v in c1[k]) + "   |                "
                             + " ".join(f"{v:>7.1%}" for v in c2[k]))
                    L.append("       " + ", ".join(nombres[zonas[i]] for i in h.filas))
                L.append("")
                filas_csv.append({"periodo": periodo, "bloque": bloque, "profundidad": n.profundidad,
                                  "intervalos": " ".join(franjas), "intervalo_contraste": franjas[j],
                                  "n_regiones_a": len(n.hijos[a].filas), "n_regiones_b": len(n.hijos[b].filas),
                                  "cuota_a_nov": c1[a, j], "cuota_b_nov": c1[b, j],
                                  "cuota_a_resto": c2[a, j], "cuota_b_resto": c2[b, j],
                                  "I_nov": informacion(T1), "I_resto": informacion(T2),
                                  "regiones_a": "; ".join(nombres[zonas[i]] for i in n.hijos[a].filas),
                                  "regiones_b": "; ".join(nombres[zonas[i]] for i in n.hijos[b].filas)})
                filas_tex.append(f"{'Laborable' if periodo == 'weekday' else 'Feriado'} & {bloque} & {franjas[j]} & "
                                 f"{len(n.hijos[a].filas)} / {len(n.hijos[b].filas)} & "
                                 f"{100 * c1[a, j]:.1f} / {100 * c1[b, j]:.1f} & {100 * c2[a, j]:.1f} / {100 * c2[b, j]:.1f} & "
                                 f"{1000 * informacion(T1):.1f} & {1000 * informacion(T2):.1f} \\\\")
        print(f"{periodo}: listo", flush=True)

    t = pd.DataFrame(filas_csv)
    L += [f"Resumen de los {len(t)} cortes de filas: diferencia de cuota entre grupos en el intervalo de mayor contraste",
          f"  noviembre: de {100 * (t.cuota_a_nov - t.cuota_b_nov).min():.1f} a {100 * (t.cuota_a_nov - t.cuota_b_nov).max():.1f} puntos; "
          f"resto de 2017: de {100 * (t.cuota_a_resto - t.cuota_b_resto).min():.1f} a "
          f"{100 * (t.cuota_a_resto - t.cuota_b_resto).max():.1f} puntos",
          f"  mayor cambio de una cuota entre noviembre y el resto: "
          f"{100 * max((t.cuota_a_nov - t.cuota_a_resto).abs().max(), (t.cuota_b_nov - t.cuota_b_resto).abs().max()):.1f} puntos",
          f"  cociente I resto / I nov: de {(t.I_resto / t.I_nov).min():.2f} a {(t.I_resto / t.I_nov).max():.2f}"]
    texto = "\n".join(L)
    print("\n" + texto)
    (SALIDA / "validacion_2017.txt").write_text(texto, encoding="utf-8")
    t.to_csv(SALIDA / "validacion_2017.csv", index=False)
    (SALIDA / "validacion_2017.tex").write_text("\n".join([
        r"\begin{tabular}{@{}llccccrr@{}}", r"\toprule",
        r" & & & & \multicolumn{2}{c}{Cuota en el intervalo (\%)} & \multicolumn{2}{c}{$I$ ($10^{-3}$ nats)} \\",
        r"\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"Periodo & Bloque & Intervalo & Regiones & Noviembre & Resto de 2017 & Nov. & Resto \\", r"\midrule",
        *filas_tex, r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
