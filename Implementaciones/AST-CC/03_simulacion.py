"""
Paso 3. ¿Funciona AST-CC? Calibración y potencia sobre datos sintéticos
========================================================================
Cap_1, objetivo 3: evaluar la calibración y la potencia del método sobre datos
sintéticos con patrones locales conocidos y sobre datos nulos.

Datos (astcc.sinteticos): matriz 66 x 12 con un tablero base 8 x 4, efectos de
fila y columna desiguales y D = 12 días, con N ~ 3 millones de viajes como en el
periodo laborable real. Tres escenarios de variabilidad entre días:
  poisson    días de Poisson, sin efectos de día
  realista   efecto de día 0.03, efectos día-región y día-intervalo 0.07, ruido de
             celda gamma con k = 200: phi^ ~ 3.1, como el periodo laborable real
  fuerte     0.10, 0.15 y k = 100: phi^ ~ 10, más parecido al periodo feriado

A. Calibración (sin patrones: ningún bloque base debe refinarse).
   FWER = proporción de réplicas con al menos un corte. Con delta = 0 se mide la
   validez de las pruebas; con delta = 0.005, el comportamiento práctico.
   Se usa la partición base verdadera y, en el escenario realista, también la
   estimada con BBAC C5 (incluye el efecto de la selección).

B. Potencia y recuperación (escenario realista, nula de bootstrap).
   Dos patrones plantados en bloques distintos: la mitad de las filas del bloque
   se activa en la mitad de sus columnas con un factor f. Se reporta:
     I_pob    información mutua poblacional del bloque con patrón (tamaño de efecto)
     detecta  proporción de patrones cuyo bloque base se refina
     ARI      acuerdo entre el primer corte y las filas (o columnas) activadas
     falsos   proporción de réplicas con algún corte en un bloque sin patrón

Salidas: Resultados/AST-CC/simulacion.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.bbac import bbac  # noqa: E402
from astcc.estadisticos import info_mutua  # noqa: E402
from astcc.sinteticos import Patron, generar  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RAIZ / "Resultados/AST-CC"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261005
M, N_, KR, KT, D = 66, 12, 8, 4, 12
ESCENARIOS = {
    "poisson": dict(var_dia=0.0, k=np.inf),
    "realista": dict(var_dia=0.03, k=200, var_dia_fila=0.07, var_dia_col=0.07),
    "fuerte": dict(var_dia=0.10, k=100, var_dia_fila=0.15, var_dia_col=0.15),
}
R_CAL = 200
R_POT = 60
FACTORES = [1.1, 1.2, 1.35, 1.5, 2.0]
DELTAS_POT = [0.0, 0.001, 0.005]


def particion(datos, estimada, rng):
    if not estimada:
        return datos.rho, datos.tau
    r = bbac(datos.X.sum(0), KR, KT, base="C5", nruns=20, rng=rng)
    return r.rho, r.tau


def calibracion(lineas, rng):
    se = np.sqrt(0.05 * 0.95 / R_CAL)
    lineas += [f"A. Calibración: FWER sin patrones ({R_CAL} réplicas; alfa = 0.05; "
               f"error Monte Carlo ~ {se:.3f})", "",
               f"  {'escenario':<10} {'partición':<10} {'nula':<10} {'delta':>6}  {'FWER':>6} {'cortes/rép.':>11} {'seg/rép.':>8}"]
    casos = [(e, False, nula, delta) for e in ESCENARIOS for nula in ("poisson", "bootstrap")
             for delta in (0.0, 0.005)]
    casos += [("realista", True, "bootstrap", delta) for delta in (0.0, 0.005)]
    for esc, estimada, nula, delta in casos:
        con_corte, cortes, t0 = 0, 0, time.time()
        for _ in range(R_CAL):
            datos = generar(rng, m=M, n=N_, KR=KR, KT=KT, D=D, **ESCENARIOS[esc])
            rho, tau = particion(datos, estimada, rng)
            res = ASTCC(Parametros(nula=nula, delta=delta), rng).ajustar(datos.X, rho, tau)
            k = len(res.internos())
            cortes += k
            con_corte += k > 0
        lineas.append(f"  {esc:<10} {'BBAC C5' if estimada else 'verdadera':<10} {nula:<10} {delta:>6g}  "
                      f"{con_corte / R_CAL:>6.3f} {cortes / R_CAL:>11.2f} {(time.time() - t0) / R_CAL:>8.2f}")
        print(lineas[-1], flush=True)
    lineas.append("")


def primer_corte_ari(raiz, filas_act, cols_act):
    """ARI entre el primer corte de la raíz y las filas o columnas activadas."""
    if not raiz.hijos:
        return 0.0
    def etiquetas(indices, eje):
        return np.array([next(k for k, h in enumerate(raiz.hijos)
                              if i in (h.filas if eje == 0 else h.cols)) for i in indices])
    verdad_f = np.isin(raiz.filas, filas_act)
    verdad_c = np.isin(raiz.cols, cols_act)
    if raiz.corte == "fil":
        return ari(verdad_f, etiquetas(raiz.filas, 0))
    if raiz.corte == "col":
        return ari(verdad_c, etiquetas(raiz.cols, 1))
    return 0.5 * (ari(verdad_f, etiquetas(raiz.filas, 0) // 2) + ari(verdad_c, etiquetas(raiz.cols, 1) % 2))


def potencia(lineas, rng):
    lineas += [f"B. Potencia y recuperación (escenario realista, nula de bootstrap, partición verdadera; "
               f"{R_POT} réplicas, 2 patrones por réplica)", "",
               f"  {'factor':>6} {'I_pob':>8} {'delta':>6}  {'detecta':>7} {'ARI':>6} {'falsos':>6}"]
    for factor in FACTORES:
        patrones = [Patron(0, 0, 0.5, 0.5, factor), Patron(3, 2, 0.5, 0.5, factor)]
        res_por_delta = {d: {"detecta": 0, "ari": [], "falsos": 0} for d in DELTAS_POT}
        I_pob = []
        for _ in range(R_POT):
            datos = generar(rng, m=M, n=N_, KR=KR, KT=KT, D=D, patrones=patrones, **ESCENARIOS["realista"])
            bloques_pat = []
            for pt in patrones:
                I, J = np.where(datos.rho == pt.g)[0], np.where(datos.tau == pt.h)[0]
                I_pob.append(float(info_mutua(datos.mu[np.ix_(I, J)])))
                bloques_pat.append((set(I), set(J)))
            for delta in DELTAS_POT:
                r = ASTCC(Parametros(nula="bootstrap", delta=delta), rng).ajustar(datos.X, datos.rho, datos.tau)
                acum = res_por_delta[delta]
                falso = False
                for raiz in r.raices:
                    clave = (set(raiz.filas), set(raiz.cols))
                    if clave in bloques_pat:
                        k = bloques_pat.index(clave)
                        acum["detecta"] += bool(raiz.hijos)
                        if raiz.hijos:
                            acum["ari"].append(primer_corte_ari(raiz, *datos.activas[k]))
                    elif raiz.hijos:
                        falso = True
                acum["falsos"] += falso
        for delta in DELTAS_POT:
            a = res_por_delta[delta]
            lineas.append(f"  {factor:>6} {np.mean(I_pob):>8.4f} {delta:>6g}  {a['detecta'] / (2 * R_POT):>7.2f} "
                          f"{np.mean(a['ari']) if a['ari'] else float('nan'):>6.2f} {a['falsos'] / R_POT:>6.2f}")
            print(lineas[-1], flush=True)
    lineas.append("")
    lineas.append("ARI: media sobre los patrones detectados; 1 = el primer corte separa exactamente")
    lineas.append("las filas (o columnas) activadas.")


def main():
    rng = np.random.default_rng(SEMILLA)
    t0 = time.time()
    lineas = ["Paso 3. AST-CC sobre datos sintéticos (66 x 12, tablero 8 x 4, D = 12, N ~ 3 M)", ""]
    calibracion(lineas, rng)
    potencia(lineas, rng)
    lineas.append(f"\nTiempo total: {(time.time() - t0) / 60:.1f} min")
    texto = "\n".join(lineas)
    (SALIDA / "simulacion.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)


if __name__ == "__main__":
    main()
