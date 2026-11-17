"""
¿Qué hace AST-CC cuando la partición base agrupa mal?
=====================================================
Complementa la calibración de AST-CC/03. Allí la base es la verdadera o la de BBAC C5, que
recupera los bloques del generador (astcc.sinteticos: media = efecto de fila x efecto de
columna x nivel del bloque, el modelo C5), así que dentro de cada bloque base se cumple la
independencia y todo corte es un falso positivo.

Con los datos reales, en cambio, el árbol refina la partición de ST-CC, que usa el esquema
C2 (una constante por bloque; coincide con BBAC C2 con ARI 0.867, AST-CC/01). C2 agrupa las
regiones por volumen y mezcla regiones con perfiles horarios distintos, así que dentro de
sus bloques la independencia no se cumple y los cortes son legítimos: reparan la base. Este
experimento comprueba que esos cortes recuperan la estructura verdadera y no la inventan.

Datos sin patrones locales (la verdad es una rejilla 8 x 4), escenarios realista y fuerte de
AST-CC/03. Para cada réplica y cada partición base (verdadera, BBAC C5 y BBAC C2, el esquema
de ST-CC) se ajusta el modelo de la base sin cortes (independencia en cada bloque base) y
AST-CC con los dos niveles de delta de astcc.referencia. Se reporta:
  ARI      de la partición base con la verdadera (filas / columnas)
  cortes   cortes del árbol
  KL       KL(p || p^) entre las proporciones verdaderas de las celdas y las del ajuste, en
           milésimas de nat por viaje (0 = el ajuste reproduce la media verdadera)
  pureza   fracción de la masa verdadera de cada hoja que cae en su bloque verdadero
           mayoritario, ponderada por masa (1 = ninguna hoja mezcla bloques verdaderos)

Salidas: Resultados/AST-CC/reparacion_base.txt
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.bbac import aproximacion, bbac  # noqa: E402
from astcc.referencia import DELTA_COMPLETO, DELTA_RESUMEN  # noqa: E402
from astcc.sinteticos import generar  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RAIZ / "Resultados/AST-CC"

SEMILLA = 20261118
M, N_, KR, KT, D = 66, 12, 8, 4, 12                     # como AST-CC/03
ESCENARIOS = {
    "realista": dict(var_dia=0.03, k=200, var_dia_fila=0.07, var_dia_col=0.07),
    "fuerte": dict(var_dia=0.10, k=100, var_dia_fila=0.15, var_dia_col=0.15),
}
R = 40
DELTAS = (DELTA_COMPLETO, DELTA_RESUMEN["weekday"])
BASES = {"verdadera": None, "BBAC C5": "C5", "BBAC C2 (ST-CC)": "C2"}


def kl(mu, A):
    p, q = mu / mu.sum(), A / A.sum()
    return float(np.sum(p * np.log(p / q)))


def pureza(hojas, mu, rho, tau):
    """Masa verdadera de cada hoja en su bloque verdadero mayoritario, ponderada por masa."""
    total, mayoritaria = 0.0, 0.0
    for h in hojas:
        masa = {}
        for i in h.filas:
            for j in h.cols:
                masa[rho[i], tau[j]] = masa.get((rho[i], tau[j]), 0.0) + mu[i, j]
        total += sum(masa.values())
        mayoritaria += max(masa.values())
    return mayoritaria / total


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEMILLA)
    L = [f"Reparación de la partición base por AST-CC: datos sin patrones locales ({R} réplicas por escenario,",
         f"{M} x {N_}, rejilla verdadera {KR} x {KT}, D = {D} días; nula de bootstrap; "
         f"delta completo = {DELTAS[0]:g}, resumen = {DELTAS[1]:g})",
         "KL en milésimas de nat por viaje; media (y desviación típica) sobre las réplicas", ""]
    for esc, kw in ESCENARIOS.items():
        acum = {}
        for r in range(R):
            datos = generar(rng, m=M, n=N_, KR=KR, KT=KT, D=D, **kw)
            O = datos.X.sum(0)
            for nombre, esquema in BASES.items():
                if esquema is None:
                    rho, tau = datos.rho, datos.tau
                else:
                    b = bbac(O, KR, KT, base=esquema, nruns=20, rng=rng)
                    rho, tau = b.rho, b.tau
                a = acum.setdefault(nombre, {"ari_f": [], "ari_c": [], "kl_base": [], "pur_base": [],
                                             **{f"{c}_{d}": [] for d in DELTAS for c in ("cortes", "kl", "pur")}})
                a["ari_f"].append(ari(datos.rho, rho))
                a["ari_c"].append(ari(datos.tau, tau))
                a["kl_base"].append(kl(datos.mu, aproximacion(O, rho, tau, "C6")))
                for delta in DELTAS:
                    res = ASTCC(Parametros(delta=delta), rng).ajustar(datos.X, rho, tau)
                    a[f"cortes_{delta}"].append(len(res.internos()))
                    a[f"kl_{delta}"].append(kl(datos.mu, res.aproximacion(O)))
                    a[f"pur_{delta}"].append(pureza(res.hojas(), datos.mu, datos.rho, datos.tau))
                    if delta == DELTAS[0]:
                        a["pur_base"].append(pureza(res.raices, datos.mu, datos.rho, datos.tau))
            print(f"  {esc}: réplica {r + 1}/{R} ({time.time() - t0:.0f} s)", flush=True)

        ms = lambda v, f=1.0, d=2: f"{f * np.mean(v):.{d}f} ({f * np.std(v):.{d}f})"  # noqa: E731
        L += [f"{esc}", "",
              f"  {'base':<17} {'ARI filas':>9} {'ARI cols':>8} | {'sin cortes':^23} | "
              f"{f'delta = {DELTAS[0]:g}':^36} | {f'delta = {DELTAS[1]:g}':^36}",
              f"  {'':<17} {'':>9} {'':>8} | {'KL':>11} {'pureza':>11} | {'cortes':>12} {'KL':>11} {'pureza':>11} | "
              f"{'cortes':>12} {'KL':>11} {'pureza':>11}"]
        for nombre, a in acum.items():
            fila = (f"  {nombre:<17} {np.mean(a['ari_f']):>9.2f} {np.mean(a['ari_c']):>8.2f} | "
                    f"{ms(a['kl_base'], 1000):>11} {np.mean(a['pur_base']):>11.3f} |")
            for d in DELTAS:
                fila += (f" {ms(a[f'cortes_{d}'], 1, 1):>12} {ms(a[f'kl_{d}'], 1000):>11} "
                         f"{np.mean(a[f'pur_{d}']):>11.3f} |")
            L.append(fila.rstrip(" |"))
        L.append("")
    L.append(f"Tiempo: {(time.time() - t0) / 60:.1f} min")
    (SALIDA / "reparacion_base.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
