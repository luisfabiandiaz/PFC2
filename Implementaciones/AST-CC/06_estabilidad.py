"""
Estabilidad de AST-CC y ajuste fuera de muestra
===============================================
Cap_3, sec:prop-representacion ("Frecuencia de soporte") y sec:prop-limitaciones
("Parámetros", "Validez de las pruebas"); cierra la cautela de 05_comparacion.py, cuyo
D_I y cuyo BIC no descuentan que la partición y los cortes se eligen con los mismos datos.

A. Frecuencia de soporte. La referencia es el árbol de AST-CC/02 (partición base de
   BBAC C5 con 200 inicializaciones, delta = 0.005). Se repite el procedimiento
   R_SOPORTE veces con una partición base distinta, la de UNA sola inicialización de
   BBAC (la más sensible a los mínimos locales), y se mide en qué proporción de las
   ejecuciones reaparece cada hoja de referencia con Jaccard >= kappa = 0.5 sobre celdas.
   Una hoja que aparece en menos de la mitad no debe interpretarse como hallazgo.

B. División de días. Se parten los D días en dos mitades al azar, R_DIVISION veces. Con una
   mitad se ajusta cada modelo (rejillas globales de BBAC C5 evaluadas con la base C6,
   y AST-CC con dos valores de delta, sobre una partición base estimada solo con esa
   mitad) y se mide D_I en la otra mitad, en ambos sentidos. Un modelo que solo se
   ajusta al ruido de su mitad empeora fuera de ella. Para cada AST-CC se compara con
   la rejilla de menos parámetros que tenga al menos tantos como él, que le da ventaja
   a la rejilla, y se cuenta en cuántos ajustes gana el árbol.

Con la mitad de los días, el bootstrap de días trabaja con muy pocos (6 y 4 o 5) y las
mitades no conservan la composición del periodo, sobre todo en el feriado; el ruido
resultante afecta por igual a todos los modelos, pero conviene leer las diferencias y
no los niveles.

Uso: python AST-CC/06_estabilidad.py [weekday] [holiday]
Salidas: Resultados/AST-CC/estabilidad_{periodo}.txt, soporte_{periodo}.csv,
         division_{periodo}.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.bbac import aproximacion, bbac, d_I  # noqa: E402
from astcc.estabilidad import KAPPA, divergencia_fuera_de_muestra, frecuencia_de_soporte  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/AST-CC"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261030
K_BASE = {"weekday": (8, 4), "holiday": (6, 4)}
REJILLAS = {"weekday": [(8, 4), (12, 4), (16, 4), (20, 4), (12, 6), (16, 6), (24, 8)],
            "holiday": [(6, 4), (8, 4), (10, 4), (12, 4), (8, 6), (12, 6), (24, 8)]}
DELTA_REF = 0.005
DELTAS = [0.001, 0.005]
R_SOPORTE = 20
R_DIVISION = 8
NRUNS_REJILLA = 20


def k_hojas(res):
    return sum(len(h.filas) + len(h.cols) - 1 for h in res.hojas())


def k_rejilla(m, n, KR, KT):
    return KT * m + KR * n - KR * KT


def soporte(periodo, X, O, L):
    kr, kt = K_BASE[periodo]
    rho = np.load(DATOS / f"AST-CC/{periodo}_C5_rho.npy")
    tau = np.load(DATOS / f"AST-CC/{periodo}_C5_tau.npy")
    ref = ASTCC(Parametros(delta=DELTA_REF), np.random.default_rng(SEMILLA)).ajustar(X, rho, tau)
    hojas_ref = ref.hojas()
    rng = np.random.default_rng(SEMILLA + 1)
    ejecuciones, aris, optimo = [], [], []
    d_ref = d_I(O, aproximacion(O, rho, tau, "C5", kr, kt))
    for r in range(R_SOPORTE):
        b = bbac(O, kr, kt, base="C5", nruns=1, rng=rng)
        res = ASTCC(Parametros(delta=DELTA_REF), np.random.default_rng(SEMILLA + 100 + r)).ajustar(X, b.rho, b.tau)
        ejecuciones.append(res.hojas())
        aris.append((ari(rho, b.rho), ari(tau, b.tau)))
        optimo.append(bool(b.divergencia <= d_ref * (1 + 1e-9)))
        print(f"  {periodo}: soporte, ejecución {r + 1}/{R_SOPORTE}: {len(res.hojas())} hojas", flush=True)
    sop = frecuencia_de_soporte(hojas_ref, ejecuciones, KAPPA)
    tabla = pd.DataFrame({
        "profundidad": [h.profundidad for h in hojas_ref], "tipo": [h.estado for h in hojas_ref],
        "n_regiones": [len(h.filas) for h in hojas_ref], "n_intervalos": [len(h.cols) for h in hojas_ref],
        "N": [int(h.N) for h in hojas_ref], "I_corr": [h.I_corr for h in hojas_ref], "soporte": sop})
    tabla.to_csv(SALIDA / f"soporte_{periodo}.csv", index=False)

    refinada = tabla["profundidad"] > 0
    aris = np.array(aris)
    L += [f"A. Frecuencia de soporte (referencia: delta = {DELTA_REF}, {len(hojas_ref)} hojas; "
          f"{R_SOPORTE} ejecuciones con una sola inicialización de BBAC; kappa = {KAPPA})", "",
          f"  La partición base de una inicialización iguala el óptimo de 200 en {np.mean(optimo):.0%} de las "
          f"ejecuciones; ARI medio con la de referencia: filas {aris[:, 0].mean():.2f}, columnas {aris[:, 1].mean():.2f}",
          f"  Soporte medio de las hojas: {sop.mean():.2f}; con soporte >= 0.5: {np.mean(sop >= 0.5):.0%} "
          f"({int((sop >= 0.5).sum())} de {len(sop)})",
          f"    hojas que son bloques base (sin cortes encima):   {int((~refinada).sum()):>3}, "
          f"soporte medio {sop[~refinada.to_numpy()].mean():.2f}" if (~refinada).any() else "    (no hay bloques base)"]
    if refinada.any():
        L.append(f"    hojas producidas por un corte (profundidad >= 1): {int(refinada.sum()):>3}, "
                 f"soporte medio {sop[refinada.to_numpy()].mean():.2f}; con soporte >= 0.5: "
                 f"{np.mean(sop[refinada.to_numpy()] >= 0.5):.0%}")
        L += ["", f"  {'prof.':>5} {'tipo':<14} {'regiones':>8} {'intervalos':>10} {'N':>9} {'I_corr':>7} {'soporte':>8}"]
        for _, f in tabla[refinada].sort_values("soporte", ascending=False).iterrows():
            L.append(f"  {int(f.profundidad):>5} {f.tipo:<14} {int(f.n_regiones):>8} {int(f.n_intervalos):>10} "
                     f"{int(f.N):>9,} {f.I_corr:>7.4f} {f.soporte:>8.2f}")
    L.append("")


def division(periodo, X, L):
    D, m, n = X.shape
    kr, kt = K_BASE[periodo]
    rng = np.random.default_rng(SEMILLA + 2)
    filas = []
    for s in range(R_DIVISION):
        perm = rng.permutation(D)
        mitades = (perm[:D // 2], perm[D // 2:])
        for sentido, (ent, pru) in enumerate((mitades, mitades[::-1])):
            Xe, Oe, Op = X[ent], X[ent].sum(0), X[pru].sum(0)
            ajustes, base = {}, None
            for KR, KT in REJILLAS[periodo]:
                b = bbac(Oe, KR, KT, base="C5", nruns=NRUNS_REJILLA, rng=rng)
                A = aproximacion(Oe, b.rho, b.tau, "C6", KR, KT)
                ajustes[f"rejilla {KR}x{KT}"] = ("rejilla", k_rejilla(m, n, KR, KT), A, 0)
                if (KR, KT) == (kr, kt):
                    base = b
            for delta in DELTAS:
                res = ASTCC(Parametros(delta=delta), rng).ajustar(Xe, base.rho, base.tau)
                ajustes[f"AST-CC d={delta:g}"] = ("AST-CC", k_hojas(res), res.aproximacion(Oe), len(res.internos()))
            ajustes["saturado"] = ("saturado", m * n, Oe, 0)
            for nombre, (tipo, k, A, cortes) in ajustes.items():
                filas.append({"division": s, "sentido": sentido, "modelo": nombre, "tipo": tipo, "k": k,
                              "cortes": cortes, "D_entrenamiento": d_I(Oe, A) * (Op.sum() / Oe.sum()),
                              "D_prueba": divergencia_fuera_de_muestra(Op, A)})
        print(f"  {periodo}: división {s + 1}/{R_DIVISION} lista", flush=True)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / f"division_{periodo}.csv", index=False)

    L += [f"B. División de días ({R_DIVISION} divisiones al azar, ambos sentidos = {2 * R_DIVISION} ajustes; "
          f"D_I medio y desviación típica, reescalado a la masa de la mitad de prueba)", "",
          f"  {'modelo':<18} {'k':>6} {'cortes':>6} {'D_I entren.':>12} {'D_I prueba':>11} {'± sd':>7} {'brecha':>8}"]
    resumen = tabla.groupby("modelo", sort=False).agg(
        tipo=("tipo", "first"), k=("k", "mean"), cortes=("cortes", "mean"),
        ent=("D_entrenamiento", "mean"), pru=("D_prueba", "mean"), sd=("D_prueba", "std"))
    orden = resumen.sort_values(["k"])
    for nombre, f in orden.iterrows():
        L.append(f"  {nombre:<18} {f.k:>6.0f} {f.cortes:>6.1f} {f.ent:>12,.0f} {f.pru:>11,.0f} {f.sd:>7,.0f} "
                 f"{f.pru - f.ent:>8,.0f}")
    L += ["", "  brecha = D_I de prueba - D_I de entrenamiento (optimismo del ajuste)", "",
          "  AST-CC frente a la rejilla de menos parámetros que tiene al menos tantos como él "
          "(ventaja para la rejilla); diferencia negativa = mejor el árbol", "",
          f"  {'modelo':<18} {'rejilla':<14} {'k árbol':>7} {'k rej.':>6} {'dif. D_I prueba':>15} {'gana el árbol':>14}"]
    for delta in DELTAS:
        nombre = f"AST-CC d={delta:g}"
        dif, gana, k_a, k_r, usada = [], [], [], [], ""
        for (s, sent), g in tabla.groupby(["division", "sentido"]):
            a = g[g.modelo == nombre].iloc[0]
            cand = g[(g.tipo == "rejilla") & (g.k >= a.k)].sort_values("k")
            if cand.empty:
                continue
            r = cand.iloc[0]
            dif.append(a.D_prueba - r.D_prueba)
            gana.append(a.D_prueba < r.D_prueba)
            k_a.append(a.k)
            k_r.append(r.k)
            usada = r.modelo
        if dif:
            L.append(f"  {nombre:<18} {usada:<14} {np.mean(k_a):>7.0f} {np.mean(k_r):>6.0f} "
                     f"{np.mean(dif):>+15,.0f} {np.mean(gana):>13.0%}")
        else:
            L.append(f"  {nombre:<18} ninguna rejilla con tantos parámetros")
    L.append("")


def main():
    periodos = [p for p in sys.argv[1:] if p in K_BASE] or list(K_BASE)
    for periodo in periodos:
        t0 = time.time()
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        O = X.sum(0)
        L = [f"Estabilidad de AST-CC: {periodo}, {O.shape[0]} x {O.shape[1]}, N = {int(O.sum()):,}, "
             f"D = {X.shape[0]} días", ""]
        soporte(periodo, X, O, L)
        division(periodo, X, L)
        L.append(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")
        texto = "\n".join(L)
        print("\n" + texto)
        (SALIDA / f"estabilidad_{periodo}.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
