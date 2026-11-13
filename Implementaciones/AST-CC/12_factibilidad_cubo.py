"""
¿Es factible hacer crecer el árbol sobre el cubo regiones x intervalos x días?
==============================================================================
En el cubo, los días son un tercer modo. Dentro de un bloque B (regiones x intervalos) la
hipótesis de homogeneidad pasa a ser: todos los días reparten sus llegadas entre las
celdas del bloque en la misma proporción. Un «corte de días» separa los días en dos grupos
con repartos distintos; un evento de un solo día sería el corte {d} frente al resto.

El script mide, con todo 2017 y sin ajustar ningún árbol nuevo, cuatro cosas que deciden
si esa extensión puede funcionar con las reglas actuales de AST-CC:

  1. Nula. Sin réplicas de días solo queda una nula paramétrica (Poisson). Se calcula la
     sobredispersión G^2 / gl de la tabla días x celdas de cada bloque: si es mucho mayor
     que 1, la prueba rechaza siempre y solo decide delta.
  2. Escala de delta. Ganancia en nats por observación del corte {d} frente al resto, para
     cada día y bloque, comparada con los delta usados (0.005 a 0.0001). La ganancia de
     aislar un día está acotada por su peso N_d / N, que es ~1/D.
  3. Qué corte elegiría el árbol. Mejor bipartición de días de cada bloque (2-medias KL,
     la misma búsqueda de AST-CC): tamaños, y si separa días de la semana o estaciones.
  4. Profundidad. Cortes sucesivos de días hasta aislar el día de un evento conocido.

Bloques: la matriz completa («todo») y las hojas del árbol de AST-CC (base C5, delta = 0.005).

Entradas: DATA/procesados/AST-CC/conteos_diarios_2017.npy, Resultados/AST-CC/hojas_{periodo}_C5.csv
Salidas:  Resultados/AST-CC/factibilidad_cubo.txt
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.cortes import corte_filas_2medias  # noqa: E402
from astcc.estadisticos import G2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("cubo11", RAIZ / "AST-CC/11_eventos_cubo.py")
cubo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cubo)

DELTAS = [0.005, 0.001, 0.0005, 0.0001]
SEMILLA = 20261004
PROF_MAX = 10


def ganancia_aislar(T):
    """Información (nats por observación del bloque) del corte {d} frente al resto, para cada día."""
    resto = T.sum(0) - T
    return G2(np.stack([T, resto], 1)) / (2 * T.sum())


def mejor_biparticion(T, rng):
    g, e = corte_filas_2medias(T[None].astype(float), rng, R_ini=20)
    return float(g[0]) / (2 * T.sum()), e[0].astype(bool)


def asociacion(grupo, categorias):
    """V de Cramér entre la bipartición y una variable categórica de los días."""
    t = pd.crosstab(grupo, categorias).to_numpy().astype(float)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return float(np.sqrt(((t - e) ** 2 / e).sum() / t.sum() / (min(t.shape) - 1)))


def main():
    zonas = pd.read_csv(cubo.DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(cubo.CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    Y_todo = np.load(cubo.DATOS / "AST-CC/conteos_diarios_2017.npy")
    todas = pd.date_range("2017-01-01", "2017-12-31")
    rng = np.random.default_rng(SEMILLA)

    L = ["Factibilidad de aplicar el árbol sobre el cubo regiones x intervalos x días (todo 2017)",
         "ganancias en nats por observación del bloque; delta de referencia: " + ", ".join(map(str, DELTAS)), ""]
    for periodo in ("weekday", "holiday"):
        sel = np.isin(todas.dayofweek, cubo.DIAS_SEMANA[periodo])
        fechas, Y = todas[sel], Y_todo[sel].astype(float)
        D = len(fechas)
        hojas = cubo.unidades(periodo, zonas)["hojas"]
        bloques = [("todo", (np.arange(len(zonas)), np.arange(12)))] + [(f"hoja {k + 1}", u) for k, u in enumerate(hojas)]
        tablas = [Y[:, f][:, :, c].reshape(D, -1) for _, (f, c) in bloques]
        L += [f"== {periodo}: D = {D} días, {len(hojas)} hojas", ""]

        # 1. Sobredispersión frente a Poisson
        disp = np.array([G2(T) / ((D - 1) * (T.shape[1] - 1)) if T.shape[1] > 1 else np.nan for T in tablas])
        info = np.array([G2(T) / (2 * T.sum()) for T in tablas])
        L += ["  1. Nula de Poisson: G^2 / gl de la tabla días x celdas (1 = sin sobredispersión)",
              f"     todo: {disp[0]:.1f}; hojas: mínimo {np.nanmin(disp[1:]):.1f}, mediana {np.nanmedian(disp[1:]):.1f}, "
              f"máximo {np.nanmax(disp[1:]):.1f}",
              f"     información total entre días (nats/obs): todo {info[0]:.4f}; hojas mediana {np.median(info[1:]):.4f}, "
              f"máximo {info[1:].max():.4f}", ""]

        # 2. Ganancia de aislar un día
        A = np.stack([ganancia_aislar(T) for T in tablas])           # bloques x días
        L += ["  2. Corte {d} frente al resto: pares (hoja, día) con ganancia >= delta, de "
              f"{A[1:].size} pares; cota por peso del día ~ 1/D = {1 / D:.4f}",
              "     " + "; ".join(f"delta {dl}: {(A[1:] >= dl).sum()} pares en {(A[1:] >= dl).any(0).sum()} días" for dl in DELTAS),
              f"     ganancia típica de un día cualquiera en una hoja: mediana {np.median(A[1:]):.6f}, "
              f"percentil 99 {np.percentile(A[1:], 99):.6f}",
              f"     {'fecha':<11}{'todo':>10}{'mejor hoja':>12}{'puesto':>8}   evento (hoja donde más gana; puesto del día en esa hoja)"]
        for f, (_, desc) in cubo.EVENTOS.items():
            if pd.Timestamp(f) not in fechas:
                continue
            d = fechas.get_loc(pd.Timestamp(f))
            b = 1 + int(A[1:, d].argmax())
            puesto = int((A[b] > A[b, d]).sum()) + 1
            L.append(f"     {f:<11}{A[0, d]:>10.6f}{A[b, d]:>12.6f}{puesto:>8}   {desc} "
                     f"({cubo.describir(bloques[b][1], zonas, nombres)})")
        L.append("")

        # 3. Mejor bipartición de días de cada bloque
        dsem = fechas.dayofweek.to_numpy()
        estacion = np.where(np.isin(fechas.month, [6, 7, 8]), "verano", "resto")
        mitad = np.where(fechas.month <= 6, "ene-jun", "jul-dic")
        res = []
        for (nombre, _), T in zip(bloques, tablas):
            g, e = mejor_biparticion(T, rng)
            menor = e if e.sum() <= D / 2 else ~e
            res.append((nombre, g, int(menor.sum()), asociacion(menor, dsem), asociacion(menor, estacion),
                        asociacion(menor, mitad), menor))
        gh = np.array([r[1] for r in res[1:]])
        tam = np.array([r[2] for r in res[1:]])
        L += ["  3. Mejor bipartición de días (2-medias KL): qué separaría el primer corte de días",
              f"     todo: ganancia {res[0][1]:.5f}, grupo menor de {res[0][2]} días; V de Cramér con día de la semana "
              f"{res[0][3]:.2f}, verano {res[0][4]:.2f}, semestre {res[0][5]:.2f}",
              f"     hojas: ganancia mediana {np.median(gh):.5f}, máxima {gh.max():.5f}; "
              + "; ".join(f">= {dl}: {(gh >= dl).sum()}" for dl in DELTAS),
              f"     tamaño del grupo menor: mínimo {tam.min()}, mediana {np.median(tam):.0f}; hojas con grupo menor <= 3 días: {(tam <= 3).sum()}",
              f"     V de Cramér mediana en las hojas: día de la semana {np.median([r[3] for r in res[1:]]):.2f}, "
              f"verano {np.median([r[4] for r in res[1:]]):.2f}, semestre {np.median([r[5] for r in res[1:]]):.2f}"]
        for r in res[1:]:
            if r[2] <= 3:
                L.append(f"     {r[0]}: aísla {', '.join(fechas[r[6]].strftime('%Y-%m-%d'))} con ganancia {r[1]:.5f} "
                         f"({cubo.describir(bloques[int(r[0].split()[1])][1], zonas, nombres)})")
        L.append("")

        # 4. Profundidad hasta aislar un evento, siguiendo el grupo que contiene el día
        L.append("  4. Cortes de días sucesivos hasta aislar el día del evento (tamaño del grupo que lo contiene -> ganancia del corte en ese nodo)")
        for f in ("2017-11-05", "2017-03-19", "2017-07-04", "2017-09-19"):
            if pd.Timestamp(f) not in fechas:
                continue
            d = fechas.get_loc(pd.Timestamp(f))
            for b in (0, 1 + int(A[1:, d].argmax())):
                dias, pasos = np.arange(D), []
                for _ in range(PROF_MAX):
                    if len(dias) <= 1:
                        break
                    if len(dias) == 2:
                        g, e = float(G2(tablas[b][dias])) / (2 * tablas[b][dias].sum()), np.array([False, True])
                    else:
                        g, e = mejor_biparticion(tablas[b][dias], rng)
                    dias = dias[e == e[np.flatnonzero(dias == d)[0]]]
                    pasos.append(f"{len(dias)} ({g:.5f})")
                L.append(f"     {f} {cubo.EVENTOS[f][1]}, {bloques[b][0]}: {D} -> " + " -> ".join(pasos)
                         + ("" if len(dias) == 1 else "  (sin aislar)"))
        L.append("")
        print(f"{periodo}: listo", flush=True)

    texto = "\n".join(L)
    print("\n" + texto)
    (cubo.SALIDA / "factibilidad_cubo.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
