"""
Paso 3b. ¿Funciona el núcleo de AST-CC con otros tipos de patrón?
==================================================================
Complementa a 03_simulacion.py, que solo usó un tipo de patrón (la mitad de las
filas de un bloque activadas en la mitad de sus columnas) con la partición base
verdadera. Aquí, en el escenario realista y con la nula de bootstrap:

  simple        el patrón de 03, como referencia
  simple-BBAC   el mismo patrón, con la partición base estimada por BBAC C5
  xor           tablero dentro del bloque: filas A activas en columnas C y filas B
                en columnas D, con marginales del bloque casi iguales
                (Cap_3, "un patrón en tablero ... sí lo captura un corte de filas")
  anidado       dos grupos de filas con perfiles distintos en el mismo bloque
                (I1 activas en la columna c1, I2 en c2): exige dos cortes de filas,
                es decir, llegar a profundidad 2
  cruza         filas de un grupo activas en columnas de dos grupos de columnas: el
                patrón queda repartido en dos bloques base y debe refinarse en ambos

Métricas, para cada patrón:
  refina   proporción de réplicas en que se refina algún bloque que contiene el patrón
  ARI      acuerdo, en las celdas del bloque verdadero, entre las hojas y la
           partición verdadera inducida por filas o por columnas (la mayor de las
           dos, porque un corte de filas y uno de columnas pueden ser igual de
           correctos); 1 = las hojas reproducen exactamente los grupos
  prof     profundidad máxima de las hojas dentro de los bloques con patrón
  falsos   proporción de réplicas con algún bloque sin patrón refinado
  absorbe  (solo simple-BBAC) proporción en que la partición base ya separa el patrón

Salidas: Resultados/AST-CC/patrones.txt
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
from astcc.sinteticos import generar  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
SALIDA = RAIZ / "Resultados/AST-CC"

SEMILLA = 20261006
M, N_, KR, KT, D = 66, 12, 8, 4, 12
REALISTA = dict(var_dia=0.03, k=200, var_dia_fila=0.07, var_dia_col=0.07)
R = 60
FACTORES = [1.25, 1.5, 2.0]
DELTAS = [0.0, 0.001]


# ----------------------------------------------------------------------------
# Patrones. Cada uno devuelve una lista de verdades: (filas, cols, grupo_fila,
# grupo_col), con las filas y columnas del bloque verdadero y el grupo de cada una.
# ----------------------------------------------------------------------------

def _bloque(rho, tau, g, h):
    return np.where(rho == g)[0], np.where(tau == h)[0]


def simple(f):
    def mod(mu, rho, tau, rng):
        I, J = _bloque(rho, tau, 0, 0)
        fi = rng.choice(I, len(I) // 2, replace=False)
        fj = rng.choice(J, max(1, len(J) // 2), replace=False)
        mu[np.ix_(fi, fj)] *= f
        return [(I, J, np.isin(I, fi).astype(int), np.isin(J, fj).astype(int))]
    return mod


def xor(f):
    def mod(mu, rho, tau, rng):
        I, J = _bloque(rho, tau, 0, 0)
        A = rng.choice(I, len(I) // 2, replace=False)
        C = rng.choice(J, max(1, len(J) // 2), replace=False)
        B, Dd = np.setdiff1d(I, A), np.setdiff1d(J, C)
        mu[np.ix_(A, C)] *= f
        mu[np.ix_(B, Dd)] *= f
        return [(I, J, np.isin(I, A).astype(int), np.isin(J, C).astype(int))]
    return mod


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
    return mod


def cruza(f):
    def mod(mu, rho, tau, rng):
        verdades = []
        I = np.where(rho == 0)[0]
        fi = rng.choice(I, len(I) // 2, replace=False)
        for h in (0, 1):
            J = np.where(tau == h)[0]
            fj = rng.choice(J, max(1, len(J) // 2), replace=False)
            mu[np.ix_(fi, fj)] *= f
            verdades.append((I, J, np.isin(I, fi).astype(int), np.isin(J, fj).astype(int)))
        return verdades
    return mod


ESCENARIOS = {"simple": (simple, False), "simple-BBAC": (simple, True), "xor": (xor, False),
              "anidado": (anidado, False), "cruza": (cruza, False)}


# ----------------------------------------------------------------------------

def mapa_hojas(res, forma):
    mapa = np.full(forma, -1)
    for k, h in enumerate(res.hojas()):
        mapa[np.ix_(h.filas, h.cols)] = k
    return mapa


def evaluar(res, verdades, forma):
    mapa = mapa_hojas(res, forma)
    celdas_pat = np.zeros(forma, bool)
    aris, refina, prof = [], False, 0
    for I, J, gf, gc in verdades:
        celdas_pat[np.ix_(I, J)] = True
        hojas = mapa[np.ix_(I, J)].ravel()
        por_fila = np.repeat(gf, len(J))
        por_col = np.tile(gc, len(I))
        aris.append(max(ari(por_fila, hojas), ari(por_col, hojas)))
    for raiz in res.raices:
        toca = celdas_pat[np.ix_(raiz.filas, raiz.cols)].any()
        if toca and raiz.hijos:
            refina = True
            prof = max(prof, max(n.profundidad for n in raiz.recorrer() if n.es_hoja))
    falsos = any(raiz.hijos and not celdas_pat[np.ix_(raiz.filas, raiz.cols)].any() for raiz in res.raices)
    return float(np.mean(aris)), refina, prof, falsos


def absorbe(rho, tau, verdades):
    """La partición base ya separa los grupos del patrón (ARI >= 0.9 en el bloque)."""
    for I, J, gf, gc in verdades:
        base = (rho[I][:, None] * 100 + tau[J][None, :]).ravel()
        if max(ari(np.repeat(gf, len(J)), base), ari(np.tile(gc, len(I)), base)) < 0.9:
            return False
    return True


def main():
    rng = np.random.default_rng(SEMILLA)
    t0 = time.time()
    L = [f"Paso 3b. Tipos de patrón (escenario realista, nula de bootstrap, {R} réplicas)", "",
         f"  {'escenario':<12} {'factor':>6} {'delta':>6}  {'refina':>6} {'ARI':>5} {'prof':>4} "
         f"{'falsos':>6} {'absorbe':>7}"]
    for nombre, (patron, estimada) in ESCENARIOS.items():
        for f in FACTORES:
            acum = {d: {"ari": [], "refina": 0, "prof": [], "falsos": 0} for d in DELTAS}
            absorbidos = 0
            for _ in range(R):
                datos = generar(rng, m=M, n=N_, KR=KR, KT=KT, D=D, modificar=patron(f), **REALISTA)
                verdades = datos.activas
                if estimada:
                    b = bbac(datos.X.sum(0), KR, KT, base="C5", nruns=20, rng=rng)
                    rho, tau = b.rho, b.tau
                    absorbidos += absorbe(rho, tau, verdades)
                else:
                    rho, tau = datos.rho, datos.tau
                for delta in DELTAS:
                    res = ASTCC(Parametros(nula="bootstrap", delta=delta), rng).ajustar(datos.X, rho, tau)
                    a, refina, prof, falsos = evaluar(res, verdades, datos.mu.shape)
                    acum[delta]["ari"].append(a)
                    acum[delta]["refina"] += refina
                    acum[delta]["prof"].append(prof)
                    acum[delta]["falsos"] += falsos
            for delta in DELTAS:
                c = acum[delta]
                L.append(f"  {nombre:<12} {f:>6} {delta:>6g}  {c['refina'] / R:>6.2f} {np.mean(c['ari']):>5.2f} "
                         f"{np.mean(c['prof']):>4.1f} {c['falsos'] / R:>6.2f} "
                         f"{(absorbidos / R if estimada else float('nan')):>7.2f}")
                print(L[-1], flush=True)
    L += ["", "ARI medio sobre todas las réplicas (0 si no se refina y la base no separa el patrón).",
          f"Tiempo total: {(time.time() - t0) / 60:.1f} min"]
    texto = "\n".join(L)
    (SALIDA / "patrones.txt").write_text(texto, encoding="utf-8")
    print("\n" + texto)


if __name__ == "__main__":
    main()
