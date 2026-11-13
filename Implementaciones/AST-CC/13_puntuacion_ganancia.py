"""
¿Sirve la ganancia del corte «día d frente al resto» como puntuación de la segunda etapa?
==========================================================================================
La ganancia de un corte en AST-CC es el G^2 de la tabla colapsada. Aquí el corte separa el
día d de los días de su línea base (mismo día de la semana, a menos de VENTANA días) dentro
de una unidad (bloque u hoja), con las celdas de la unidad como la otra dimensión:

    g(d, u) = G^2([T_d ; suma de T en la línea base]) / (2 N_{d,u})      nats por llegada del día

Dividir entre las llegadas del día (y no entre las del bloque) quita la cota 1/D que impide
comparar con delta. Como el nivel de g depende del tamaño de la unidad, se estandariza
log g de forma robusta entre los días del mismo día de la semana. La puntuación del día es
el máximo sobre las unidades.

Se compara con el z por celda de AST-CC/11 en las mismas fechas reales, y con cuatro
particiones: toda la matriz, rejilla de ST-CC, partición base C5 y hojas de AST-CC.

Aviso: los eventos de HALLADOS se encontraron mirando el z por celda, así que favorecen a
esa puntuación; la comparación limpia es la de EVENTOS.

Salidas: Resultados/AST-CC/puntuacion_ganancia.txt
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.estadisticos import G2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("cubo11", RAIZ / "AST-CC/11_eventos_cubo.py")
cubo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cubo)

TOP = 25


def ganancia(Y, fechas, us):
    """z robusto de log g(d, u). Devuelve (días x unidades de más de una celda, índices de esas unidades)."""
    dsem, dia = fechas.dayofweek.to_numpy(), fechas.dayofyear.to_numpy()
    D = len(fechas)
    validas = [k for k, (f, c) in enumerate(us) if len(f) * len(c) > 1]
    g = np.zeros((D, len(validas)))
    for a, k in enumerate(validas):
        f, c = us[k]
        T = Y[:, f][:, :, c].reshape(D, -1).astype(float)
        for d in range(D):
            base = (dsem == dsem[d]) & (np.abs(dia - dia[d]) <= cubo.VENTANA) & (np.arange(D) != d)
            g[d, a] = G2(np.stack([T[d], T[base].sum(0)])) / (2 * max(T[d].sum(), 1))
    lg = np.log(np.maximum(g, 1e-12))
    Z = np.zeros_like(lg)
    for w in np.unique(dsem):
        k = dsem == w
        med = np.median(lg[k], 0)
        mad = 1.4826 * np.median(np.abs(lg[k] - med), 0)
        Z[k] = (lg[k] - med) / np.maximum(mad, 1e-9)
    return Z, validas


def main():
    zonas = pd.read_csv(cubo.DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(cubo.CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    Y_todo = np.load(cubo.DATOS / "AST-CC/conteos_diarios_2017.npy")
    todas = pd.date_range("2017-01-01", "2017-12-31")

    L = ["Ganancia del corte «día d frente a su línea base» como puntuación de la segunda etapa",
         "puesto = posición del día entre los de su periodo (1 = el más atípico)",
         "z celda = max |z| de celda (AST-CC/11); G = max sobre unidades del z de log g", ""]
    for periodo in ("weekday", "holiday"):
        sel = np.isin(todas.dayofweek, cubo.DIAS_SEMANA[periodo])
        fechas, Y = todas[sel], Y_todo[sel]
        U = cubo.unidades(periodo, zonas)
        U["todo"] = [(np.arange(len(zonas)), np.arange(12))]
        punt = {"z celda": np.abs(cubo.puntuar(Y, fechas, U["celda"])).max(1)}
        donde = {}
        for b in ("todo", "stcc", "base", "hojas"):
            Z, validas = ganancia(Y, fechas, U[b])
            punt[f"G {b}"] = Z.max(1)
            donde[b] = [U[b][validas[k]] for k in Z.argmax(1)]
        puesto = {b: pd.Series(-p).rank(method="min").astype(int).to_numpy() for b, p in punt.items()}
        amp = (np.abs(cubo.puntuar(Y, fechas, U["hojas"])) >= cubo.Z_AMP).mean(1)
        clave = fechas.strftime("%Y-%m-%d")

        L += [f"== {periodo}: {len(fechas)} días", "",
              f"  {'fecha':<11}{'tipo':<11}" + "".join(f"{b:>9}" for b in punt) + "   evento  ->  hoja con mayor ganancia"]
        for f, (tipo, desc) in cubo.EVENTOS.items():
            if f in clave:
                d = list(clave).index(f)
                L.append(f"  {f:<11}{tipo:<11}" + "".join(f"{puesto[b][d]:>9}" for b in punt)
                         + f"   {desc}  ->  {cubo.describir(donde['hojas'][d], zonas, nombres)}")
        for f, desc in cubo.HALLADOS.items():
            if f in clave:
                d = list(clave).index(f)
                L.append(f"  {f:<11}{'hallado':<11}" + "".join(f"{puesto[b][d]:>9}" for b in punt) + f"   {desc}")
        for tipo in ("local", "calendario"):
            ds = [list(clave).index(f) for f, (t, _) in cubo.EVENTOS.items() if t == tipo and f in clave]
            L.append(f"  {'mediana ' + tipo:<22}" + "".join(f"{np.median(puesto[b][ds]):>9.0f}" for b in punt))
        ds = [list(clave).index(f) for f in cubo.HALLADOS if f in clave]
        L += [f"  {'mediana hallado':<22}" + "".join(f"{np.median(puesto[b][ds]):>9.0f}" for b in punt), "",
              f"  Los {TOP} días más atípicos de cada puntuación: cuántos son de EVENTOS local, de calendario o difusos "
              f"(amplitud >= {cubo.AMP_DIFUSO}), hallados, o sin explicar",
              f"  {'':<22}" + "".join(f"{b:>9}" for b in punt)]
        cuenta = {b: {"local": 0, "difuso": 0, "hallado": 0, "sin explicar": 0} for b in punt}
        for b in punt:
            for d in np.argsort(-punt[b])[:TOP]:
                f = clave[d]
                if cubo.EVENTOS.get(f, ("",))[0] == "calendario" or amp[d] >= cubo.AMP_DIFUSO:
                    cuenta[b]["difuso"] += 1
                elif f in cubo.EVENTOS:
                    cuenta[b]["local"] += 1
                elif f in cubo.HALLADOS:
                    cuenta[b]["hallado"] += 1
                else:
                    cuenta[b]["sin explicar"] += 1
        for fila in ("local", "hallado", "difuso", "sin explicar"):
            L.append(f"  {fila:<22}" + "".join(f"{cuenta[b][fila]:>9}" for b in punt))
        L.append("")
        print(f"{periodo}: listo", flush=True)

    texto = "\n".join(L)
    print("\n" + texto)
    (cubo.SALIDA / "puntuacion_ganancia.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
