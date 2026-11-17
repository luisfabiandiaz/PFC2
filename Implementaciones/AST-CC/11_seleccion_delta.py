"""
Elección de delta con datos que el árbol no vio
===============================================
Cap_3, sec:prop-parada ("Elección de delta"). El árbol se ajusta con noviembre de 2017 para
cada delta de la rejilla, y se mide qué tan bien se sostiene fuera de noviembre con tres
criterios independientes:

A. Ajuste fuera de muestra. La aproximación de noviembre (independencia dentro de cada hoja)
   predice la matriz de cada uno de los otros once meses de 2017. La pérdida es
   D_I(O_mes || predicción) / N_mes, en nats por viaje, y se resume con la media sobre los
   meses y el error estándar de la diferencia pareada frente al mejor modelo. Se incluyen
   rejillas globales optimizadas para C6 y el modelo saturado (las cuotas de noviembre de
   cada celda) como referencia.

B. Estabilidad de los cortes. Soporte (Jaccard >= kappa sobre celdas, astcc.estabilidad) de las
   hojas producidas por un corte del árbol de referencia en R ajustes:
     submuestra  la misma partición base con el 75 % de los días de noviembre, sin reemplazo
                 (variabilidad muestral de los días)
     BBAC 1 ini  todos los días con la partición base de una sola inicialización de BBAC C5,
                 mejorada con optimizar_c6 en la base C6 (la variabilidad de la base, como
                 AST-CC/06); no aplica a la base de ST-CC, que es una partición fija

Tres particiones base con los K_R x K_T de Liu et al. (2021):
  C5  BBAC con la base C5 (AST-CC/01)
  C6  la misma rejilla optimizada para D_I(O || C6), el modelo de las hojas (AST-CC/05)
  ST  la partición de ST-CC tal cual (ST-CC/01), la que el título promete refinar

C. Replicación en el resto de 2017. Para cada corte, la ganancia sin corregir del corte en la
   matriz del resto de 2017 (información mutua de la tabla colapsada: grupos del corte por
   intervalos, o regiones por grupos de intervalos, o 2 x 2), con los grupos que eligió
   noviembre. Con decenas de millones de viajes el sesgo es despreciable, así que el corte se
   replica si esa ganancia es >= delta, el mismo umbral con que se aceptó. También se cuenta
   en cuántos de los once meses, por separado, la ganancia alcanza delta.

Los días del resto de 2017 son los de AST-CC/10 (martes a jueves y fines de semana, sin
festivos, nevadas ni fiestas de fin de año).

Uso: python AST-CC/11_seleccion_delta.py [weekday] [holiday]
Salidas: Resultados/AST-CC/seleccion_delta_{periodo}.txt y seleccion_delta_{periodo}.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.bbac import aproximacion, bbac, optimizar_c6  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

from astcc.estabilidad import KAPPA, divergencia_fuera_de_muestra, frecuencia_de_soporte  # noqa: E402
from astcc.referencia import BASES, cargar_base  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/AST-CC"

SEMILLA = 20261117
DELTAS = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]
DELTAS_ESTAB = [0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01]
K_BASE = {"weekday": (8, 4), "holiday": (6, 4)}
REJILLAS = {"weekday": [(8, 4), (12, 4), (16, 4), (20, 4), (12, 6), (16, 6), (24, 6), (24, 8), (32, 8)],
            "holiday": [(6, 4), (8, 4), (12, 4), (8, 6), (12, 6), (16, 6), (24, 6), (24, 8), (32, 8)]}
NRUNS_REJILLA = 20
R_ESTAB = 20
FRACCION_SUB = 0.75
DIAS_SEMANA = {"weekday": [1, 2, 3], "holiday": [5, 6]}          # lunes = 0
ATIPICOS = ["2017-01-01", "2017-02-09", "2017-03-14", "2017-03-15", "2017-07-04",
            *[f"2017-12-{d}" for d in range(21, 32)]]


def k_hojas(res):
    return sum(len(h.filas) + len(h.cols) - 1 for h in res.hojas())


def k_rejilla(m, n, KR, KT):
    return KT * m + KR * n - KR * KT


def meses_2017(periodo):
    """Matriz zona x intervalo de cada mes de 2017 salvo noviembre, con los días del periodo."""
    Y = np.load(DATOS / "AST-CC/conteos_diarios_2017.npy").astype(float)
    fechas = pd.date_range("2017-01-01", "2017-12-31")
    usable = ~fechas.isin(pd.to_datetime(ATIPICOS)) & np.isin(fechas.dayofweek, DIAS_SEMANA[periodo])
    return {mes: Y[usable & (fechas.month == mes)].sum(0) for mes in range(1, 13) if mes != 11}


def informacion(T):
    """Información mutua de una tabla de conteos, en nats por observación."""
    P = T / T.sum()
    E = P.sum(1, keepdims=True) * P.sum(0, keepdims=True)
    return float(np.sum(P[P > 0] * np.log(P[P > 0] / E[P > 0])))


def tabla_colapsada(nodo, O):
    """Tabla del corte del nodo sobre la matriz O: grupos de regiones x intervalos (fil),
    regiones x grupos de intervalos (col) o 2 x 2 (conj)."""
    if nodo.corte == "fil":
        return np.array([O[np.ix_(h.filas, nodo.cols)].sum(0) for h in nodo.hijos])
    if nodo.corte == "col":
        return np.array([O[np.ix_(nodo.filas, h.cols)].sum(1) for h in nodo.hijos]).T
    return np.array([O[np.ix_(h.filas, h.cols)].sum() for h in nodo.hijos]).reshape(2, 2)


def soporte_cortes(ref, ejecuciones):
    """Soporte medio de las hojas producidas por un corte y fracción con soporte >= 0.5."""
    hojas = [h for h in ref.hojas() if h.profundidad > 0]
    if not hojas:
        return np.nan, np.nan
    s = frecuencia_de_soporte(hojas, ejecuciones, KAPPA)
    return float(s.mean()), float(np.mean(s >= 0.5))


def main(periodo):
    t0 = time.time()
    X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
    O = X.sum(0)
    D, m, n = X.shape
    kr, kt = K_BASE[periodo]
    meses = meses_2017(periodo)
    O_resto = sum(meses.values())
    rng = np.random.default_rng(SEMILLA)

    def perdidas(A):
        return np.array([divergencia_fuera_de_muestra(Om, A) / Om.sum() for Om in meses.values()])

    filas = []
    # --- Referencias: rejillas optimizadas para C6 y saturado
    for KR, KT in REJILLAS[periodo]:
        b = bbac(O, KR, KT, base="C5", nruns=NRUNS_REJILLA, rng=rng)
        rho6, tau6, _ = optimizar_c6(O, b.rho, b.tau, KR, KT)
        filas.append({"modelo": f"rejilla {KR}x{KT}", "base": "", "delta": np.nan, "k": k_rejilla(m, n, KR, KT),
                      "hojas": KR * KT, "cortes": 0, "perdida": perdidas(aproximacion(O, rho6, tau6, "C6", KR, KT))})
    filas.append({"modelo": "saturado", "base": "", "delta": np.nan, "k": m * n, "hojas": m * n, "cortes": 0,
                  "perdida": perdidas(O)})
    print(f"  {periodo}: referencias listas ({time.time() - t0:.0f} s)", flush=True)

    # --- AST-CC por delta y base
    bases = {b: cargar_base(periodo, b) for b in BASES}
    arboles = {}
    for nb, (rho, tau) in bases.items():
        for delta in DELTAS + [np.inf]:
            res = ASTCC(Parametros(delta=delta), np.random.default_rng(SEMILLA)).ajustar(X, rho, tau)
            arboles[nb, delta] = res
            cortes = res.internos()
            g_resto = [informacion(tabla_colapsada(c, O_resto)) for c in cortes]
            g_meses = np.array([[informacion(tabla_colapsada(c, Om)) for Om in meses.values()] for c in cortes])
            filas.append({
                "modelo": "AST-CC" if np.isfinite(delta) else "base sin cortes", "base": nb, "delta": delta,
                "k": k_hojas(res), "hojas": len(res.hojas()), "cortes": len(cortes),
                "cortes_fil": sum(c.corte == "fil" for c in cortes), "cortes_col": sum(c.corte == "col" for c in cortes),
                "perdida": perdidas(res.aproximacion(O)),
                "replica_resto": float(np.mean([g >= delta for g in g_resto])) if cortes else np.nan,
                "replica_meses": float(np.mean(g_meses >= delta)) if cortes else np.nan,
                "g_nov_medio": float(np.mean([c.g_corr for c in cortes])) if cortes else np.nan,
                "g_resto_medio": float(np.mean(g_resto)) if cortes else np.nan})
        print(f"  {periodo}: árboles con base {nb} listos ({time.time() - t0:.0f} s)", flush=True)

    # --- Estabilidad
    sub = {(b, d): [] for b in BASES for d in DELTAS_ESTAB}
    ini = {(b, d): [] for b in ("C5", "C6") for d in DELTAS_ESTAB}
    n_sub = int(round(FRACCION_SUB * D))
    for r in range(R_ESTAB):
        dias = np.sort(rng.choice(D, n_sub, replace=False))
        b5 = bbac(O, kr, kt, base="C5", nruns=1, rng=rng)
        rho6, tau6, _ = optimizar_c6(O, b5.rho, b5.tau, kr, kt)
        una = {"C5": (b5.rho, b5.tau), "C6": (rho6, tau6)}
        for delta in DELTAS_ESTAB:
            for nb, (rho, tau) in bases.items():
                semilla = np.random.default_rng(SEMILLA + 1000 + r)
                sub[nb, delta].append(ASTCC(Parametros(delta=delta), semilla).ajustar(X[dias], rho, tau).hojas())
                if nb in una:
                    ini[nb, delta].append(ASTCC(Parametros(delta=delta), semilla).ajustar(X, *una[nb]).hojas())
        print(f"  {periodo}: estabilidad {r + 1}/{R_ESTAB} ({time.time() - t0:.0f} s)", flush=True)
    for f in filas:
        clave = (f["base"], f["delta"])
        if f["modelo"] == "AST-CC" and clave in sub:
            ref = arboles[clave]
            f["sop_sub"], f["sop05_sub"] = soporte_cortes(ref, sub[clave])
            if clave in ini:
                f["sop_ini"], f["sop05_ini"] = soporte_cortes(ref, ini[clave])

    # --- Resumen
    perd = np.array([f["perdida"] for f in filas])
    mejor = int(np.argmin(perd.mean(1)))
    for i, f in enumerate(filas):
        dif = perd[i] - perd[mejor]
        f["perdida_media"] = float(perd[i].mean())
        f["dif_mejor"] = float(dif.mean())
        f["ee_dif"] = float(dif.std(ddof=1) / np.sqrt(len(dif))) if i != mejor else 0.0
        f["ee_perdida"] = float(perd[i].std(ddof=1) / np.sqrt(perd.shape[1]))
    # Regla de un error estándar: el modelo de menos parámetros cuya diferencia pareada con el
    # mejor no supera su propio error estándar
    dentro = [f for f in filas if f["dif_mejor"] <= f["ee_dif"]]
    elegido = min(dentro, key=lambda f: f["k"])

    tabla = pd.DataFrame([{k: v for k, v in f.items() if k != "perdida"} for f in filas])
    for j, mes in enumerate(meses):
        tabla[f"perdida_mes{mes:02d}"] = perd[:, j]
    tabla.to_csv(SALIDA / f"seleccion_delta_{periodo}.csv", index=False)

    L = [f"Elección de delta: {periodo}, noviembre D = {D} días, N = {int(O.sum()):,}; resto de 2017: "
         f"{len(meses)} meses, N = {int(O_resto.sum()):,}",
         "pérdida = D_I(O_mes || predicción de noviembre) / N_mes, media de los 11 meses, en milésimas de nat por viaje;",
         "dif. = diferencia pareada con el mejor modelo (ee: error estándar entre meses)", "",
         "ARI de las particiones base con la de ST-CC (filas / columnas): "
         + ", ".join(f"{b} {ari(bases['ST'][0], bases[b][0]):.3f} / {ari(bases['ST'][1], bases[b][1]):.3f}"
                     for b in BASES if b != "ST"), "",
         "A. Ajuste fuera de muestra", "",
         f"  {'modelo':<16} {'base':>4} {'delta':>8} {'k':>4} {'hojas':>5} {'cortes':>6} {'f/c':>5} "
         f"{'pérdida':>8} {'dif.':>7} {'ee':>6}"]
    for f in sorted(filas, key=lambda f: f["k"]):
        d = "" if not np.isfinite(f["delta"]) else f"{f['delta']:g}"
        fc = f"{f.get('cortes_fil', 0)}/{f.get('cortes_col', 0)}" if f["cortes"] else ""
        marca = "  <- mejor" if f is filas[mejor] else ("  <- regla de 1 ee" if f is elegido else "")
        L.append(f"  {f['modelo']:<16} {f['base']:>4} {d:>8} {f['k']:>4} {f['hojas']:>5} {f['cortes']:>6} {fc:>5} "
                 f"{1000 * f['perdida_media']:>8.3f} {1000 * f['dif_mejor']:>7.3f} {1000 * f['ee_dif']:>6.3f}{marca}")
    L += ["", "B y C. Estabilidad y replicación de los cortes en el resto de 2017"]
    for nb in BASES:
        L += ["", f"  base {nb}",
              f"  {'delta':>8} {'cortes':>6} {'sop. sub':>8} {'>=0.5':>6} {'sop. ini':>8} {'>=0.5':>6} "
              f"{'g~ nov':>7} {'g resto':>7} {'replica':>7} {'meses':>6}"]
        for f in filas:
            if f["modelo"] != "AST-CC" or f["base"] != nb:
                continue
            if not f["cortes"]:
                L.append(f"  {f['delta']:>8g} {0:>6}")
                continue
            est = lambda c, fmt: format(f[c], fmt) if c in f and np.isfinite(f[c]) else "—"
            L.append(f"  {f['delta']:>8g} {f['cortes']:>6} {est('sop_sub', '.2f'):>8} {est('sop05_sub', '.0%'):>6} "
                     f"{est('sop_ini', '.2f'):>8} {est('sop05_ini', '.0%'):>6} {f['g_nov_medio']:>7.4f} "
                     f"{f['g_resto_medio']:>7.4f} {f['replica_resto']:>7.0%} {f['replica_meses']:>6.0%}")
    L += ["",
          "  sop. sub / sop. ini: soporte medio de las hojas producidas por un corte con el 75 % de los días /",
          "  con la partición base de una sola inicialización; >=0.5: fracción de esas hojas con soporte >= 0.5.",
          "  replica: fracción de cortes con ganancia >= delta en el resto de 2017; meses: lo mismo, mes a mes.",
          "", f"Tiempo: {(time.time() - t0) / 60:.1f} min"]
    (SALIDA / f"seleccion_delta_{periodo}.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:] or ["weekday", "holiday"]:
        main(p)
