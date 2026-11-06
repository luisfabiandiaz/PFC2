"""
AST-CC frente a rejillas globales sobre datos sintéticos con verdad conocida
============================================================================
Cap_1, objetivo 4, y Cap_3 sec:prop-limitaciones ("Comparación con rejillas más
finas"): ¿una rejilla global con el mismo número de parámetros recupera los patrones
locales que encuentra AST-CC?

En los datos reales (05 y 06) no hay verdad, y la rejilla de comparación se optimiza con
BBAC (bases C2 y C5) aunque se evalúa con C6. Aquí la rejilla se optimiza directamente
para D_I(O || C6) (astcc.bbac.optimizar_c6), que es el criterio de las hojas de AST-CC, y se
compara con la verdad:

  err_verdad   D_I(mu || ajuste): distancia del ajuste a la media verdadera mu (menor es
               mejor; se mide con la media y no con otra realización del ruido)
  ARI_bloque   acuerdo, dentro de cada bloque con patrón, entre las celdas del patrón
               plantado y los grupos del método (hojas o bloques de la rejilla); 1 = el
               método aísla exactamente el patrón sin fragmentar el resto

Métodos, todos con la partición base estimada con BBAC C5 (sin usar la verdad):
  base         partición base con la aproximación C6 (k = K_T m + K_R n - K_R K_T)
  AST-CC       delta = 0 y delta = 0.001, sobre la base
  rejilla C6   la rejilla (K_R, K_T) de menos parámetros con al menos tantos como AST-CC,
               optimizada para C6 (ventaja para la rejilla)
  rejilla BBAC la misma (K_R, K_T) con la partición de BBAC C5, como en 05 y 06

Escenarios (datos como 03 y 04: 66 x 12, tablero 8 x 4, D = 12, variabilidad "realista"):
  simple-1.25 / 1.5 / 2.0   dos patrones de la mitad de las filas y la mitad de las columnas
  xor-1.5                   patrón de tablero dentro de un bloque
  anidado-1.5               dos columnas con distintos subconjuntos de filas activos

Uso: python AST-CC/09_rejillas_sinteticos.py [escenario ...]   (sin argumentos: todos)
     python AST-CC/09_rejillas_sinteticos.py resumen             (solo reúne los CSV)
Salidas: Resultados/AST-CC/rejillas_sinteticos_{escenario}.csv, rejillas_sinteticos.txt
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
from astcc.bbac import aproximacion, bbac, optimizar_c6  # noqa: E402
from astcc.estabilidad import divergencia_fuera_de_muestra  # noqa: E402
from astcc.sinteticos import Patron, generar  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RAIZ / "Resultados/AST-CC"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261105
M, N_, KR, KT, D = 66, 12, 8, 4, 12
REALISTA = dict(var_dia=0.03, k=200, var_dia_fila=0.07, var_dia_col=0.07)
R = 100
DELTAS = [0.0, 0.001]
REJILLAS = [(kr, kt) for kt in (4, 5, 6) for kr in (8, 10, 12, 14, 16, 20, 24)]


# ----------------------------------------------------------------------------
# Escenarios: cada uno devuelve los datos y la lista de verdades (I, J, máscara)
# ----------------------------------------------------------------------------

def _bloque(rho, tau, g, h):
    return np.where(rho == g)[0], np.where(tau == h)[0]


def simple(f):
    def gen(rng):
        pats = [Patron(0, 0, 0.5, 0.5, f), Patron(3, 2, 0.5, 0.5, f)]
        d = generar(rng, m=M, n=N_, KR=KR, KT=KT, D=D, patrones=pats, **REALISTA)
        verdades = []
        for pt, (fi, fj) in zip(pats, d.activas):
            I, J = _bloque(d.rho, d.tau, pt.g, pt.h)
            verdades.append((I, J, np.isin(I, fi)[:, None] & np.isin(J, fj)[None, :]))
        return d, verdades
    return gen


def _con_modificar(mod_f, mascara):
    def gen(rng):
        d = generar(rng, m=M, n=N_, KR=KR, KT=KT, D=D, modificar=mod_f, **REALISTA)
        return d, [(I, J, mascara(gf, gc)) for I, J, gf, gc in d.activas]
    return gen


def xor(f):
    def mod(mu, rho, tau, rng):
        I, J = _bloque(rho, tau, 0, 0)
        A = rng.choice(I, len(I) // 2, replace=False)
        C = rng.choice(J, max(1, len(J) // 2), replace=False)
        B, Dd = np.setdiff1d(I, A), np.setdiff1d(J, C)
        mu[np.ix_(A, C)] *= f
        mu[np.ix_(B, Dd)] *= f
        return [(I, J, np.isin(I, A).astype(int), np.isin(J, C).astype(int))]
    return _con_modificar(mod, lambda gf, gc: gf[:, None] == gc[None, :])


def anidado(f):
    def mod(mu, rho, tau, rng):
        I, J = _bloque(rho, tau, 0, 0)
        orden = rng.permutation(I)
        I1, I2 = orden[: len(I) // 3], orden[len(I) // 3: 2 * (len(I) // 3)]
        c1, c2 = rng.choice(J, 2, replace=False)
        mu[np.ix_(I1, [c1])] *= f
        mu[np.ix_(I2, [c2])] *= f
        gf = np.where(np.isin(I, I1), 1, np.where(np.isin(I, I2), 2, 0))
        gc = np.where(J == c1, 1, np.where(J == c2, 2, 0))
        return [(I, J, gf, gc)]
    return _con_modificar(mod, lambda gf, gc: (gf[:, None] == gc[None, :]) & (gf[:, None] > 0))


ESCENARIOS = {"simple-1.25": simple(1.25), "simple-1.5": simple(1.5), "simple-2.0": simple(2.0),
              "xor-1.5": xor(1.5), "anidado-1.5": anidado(1.5)}


# ----------------------------------------------------------------------------

def k_rejilla(m, n, kr, kt):
    return kt * m + kr * n - kr * kt


def k_hojas(res):
    return sum(len(h.filas) + len(h.cols) - 1 for h in res.hojas())


def etiquetas_hojas(res, forma):
    mapa = np.full(forma, -1)
    for k, h in enumerate(res.hojas()):
        mapa[np.ix_(h.filas, h.cols)] = k
    return mapa


def etiquetas_rejilla(rho, tau, kt):
    return rho[:, None] * kt + tau[None, :]


def ari_bloques(etiquetas, verdades):
    return float(np.mean([ari(mask.ravel(), etiquetas[np.ix_(I, J)].ravel()) for I, J, mask in verdades]))


def err_verdad(mu, A):
    return divergencia_fuera_de_muestra(mu, A)


def una_replica(gen, rng):
    d, verdades = gen(rng)
    O = d.X.sum(0)
    forma = O.shape
    base = bbac(O, KR, KT, base="C5", nruns=20, rng=rng)
    filas = [dict(metodo="base C6", k=k_rejilla(M, N_, KR, KT),
                  err=err_verdad(d.mu, aproximacion(O, base.rho, base.tau, "C6", KR, KT)),
                  ari=ari_bloques(etiquetas_rejilla(base.rho, base.tau, KT), verdades))]
    cache = {}

    def rejilla(kr, kt):
        if (kr, kt) not in cache:
            mejores = []
            for b in ("C2", "C5"):
                r = bbac(O, kr, kt, base=b, nruns=10, rng=rng)
                mejores.append(optimizar_c6(O, r.rho, r.tau, kr, kt)[:2])
            r5 = bbac(O, kr, kt, base="C5", nruns=10, rng=rng)
            c6 = min(mejores, key=lambda p: np.sum(_d6(O, p, kr, kt)))
            cache[(kr, kt)] = (c6, (r5.rho, r5.tau))
        return cache[(kr, kt)]

    for delta in DELTAS:
        res = ASTCC(Parametros(delta=delta), rng).ajustar(d.X, base.rho, base.tau)
        k = k_hojas(res)
        filas.append(dict(metodo=f"AST-CC d={delta:g}", k=k, err=err_verdad(d.mu, res.aproximacion(O)),
                          ari=ari_bloques(etiquetas_hojas(res, forma), verdades)))
        cand = sorted((k_rejilla(M, N_, a, b), a, b) for a, b in REJILLAS if k_rejilla(M, N_, a, b) >= k)
        kg, a, b = cand[0] if cand else sorted((k_rejilla(M, N_, a, b), a, b) for a, b in REJILLAS)[-1]
        (rho6, tau6), (rho5, tau5) = rejilla(a, b)
        filas.append(dict(metodo=f"rejilla C6 (d={delta:g})", k=kg,
                          err=err_verdad(d.mu, aproximacion(O, rho6, tau6, "C6", a, b)),
                          ari=ari_bloques(etiquetas_rejilla(rho6, tau6, b), verdades)))
        filas.append(dict(metodo=f"rejilla BBAC (d={delta:g})", k=kg,
                          err=err_verdad(d.mu, aproximacion(O, rho5, tau5, "C6", a, b)),
                          ari=ari_bloques(etiquetas_rejilla(rho5, tau5, b), verdades)))
    filas.append(dict(metodo="saturado", k=M * N_, err=err_verdad(d.mu, O), ari=np.nan))
    return filas


def _d6(O, p, kr, kt):
    from astcc.bbac import divergencia
    return divergencia(O, p[0], p[1], "C6", kr, kt)


def correr(nombre):
    rng = np.random.default_rng(SEMILLA + sum(map(ord, nombre)))
    t0 = time.time()
    filas = []
    for r in range(R):
        for f in una_replica(ESCENARIOS[nombre], rng):
            filas.append({"escenario": nombre, "rep": r, **f})
        print(f"  {nombre}: réplica {r + 1}/{R} ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(filas).to_csv(SALIDA / f"rejillas_sinteticos_{nombre}.csv", index=False)


def resumen():
    L = ["AST-CC frente a rejillas globales sobre datos sintéticos con verdad conocida",
         f"(66 x 12, tablero 8 x 4, D = 12, variabilidad realista, {R} réplicas por escenario; partición base de BBAC C5)",
         "err = D_I(mu || ajuste), menor es mejor; ARI = acuerdo con el patrón plantado dentro de su bloque", ""]
    for nombre in ESCENARIOS:
        f = SALIDA / f"rejillas_sinteticos_{nombre}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        L += [f"{nombre}", f"  {'método':<24} {'k':>6} {'err':>9} {'ARI':>6}"]
        orden = ["base C6", "AST-CC d=0", "rejilla C6 (d=0)", "rejilla BBAC (d=0)",
                 "AST-CC d=0.001", "rejilla C6 (d=0.001)", "rejilla BBAC (d=0.001)", "saturado"]
        g = t.groupby("metodo")
        for m in orden:
            if m in g.groups:
                x = g.get_group(m)
                L.append(f"  {m:<24} {x.k.mean():>6.0f} {x.err.mean():>9,.0f} {x.ari.mean():>6.2f}")
        L.append("  AST-CC frente a la rejilla C6 de igual o mayor k, por réplica:")
        for d in ("0", "0.001"):
            a = t[t.metodo == f"AST-CC d={d}"].set_index("rep")
            r = t[t.metodo == f"rejilla C6 (d={d})"].set_index("rep")
            b = t[t.metodo == f"rejilla BBAC (d={d})"].set_index("rep")
            L.append(f"    d={d:<6} err: gana el árbol {np.mean(a.err < r.err):.0%} frente a rejilla C6 "
                     f"(dif. media {np.mean(a.err - r.err):+,.0f}), {np.mean(a.err < b.err):.0%} frente a rejilla BBAC "
                     f"(dif. media {np.mean(a.err - b.err):+,.0f}); ARI árbol {a.ari.mean():.2f} "
                     f"frente a {r.ari.mean():.2f} (rejilla C6)")
        L.append("")
    texto = "\n".join(L)
    print(texto)
    (SALIDA / "rejillas_sinteticos.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["resumen"]:
        resumen()
    else:
        for nombre in (args or list(ESCENARIOS)):
            correr(nombre)
        resumen()
