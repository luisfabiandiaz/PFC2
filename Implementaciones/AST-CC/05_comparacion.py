"""
Comparación de AST-CC con ST-CC y con rejillas globales
=======================================================
Cap_1, objetivo específico 4, y Cap_3, eq:prop-bic y sec:prop-limitaciones
("Comparación con rejillas más finas"): ¿una rejilla global con el mismo número de
parámetros recupera lo que encuentra el refinamiento local?

Todos los modelos son de Poisson sobre la matriz de conteos O, así que se comparan con
la misma escala: la divergencia D_I(O || O^) (= devianza / 2), el número de parámetros
libres k y el criterio de la tesis

    J = D_I + (1/2) log(N) k                                    (eq:prop-bic).

Modelos (tab:parametros y def:aprox-arbol):
  ST-CC C2   bloques constantes de la partición de cgc             k = K_R K_T
  ST-CC C5   efectos de fila y columna, bloques de cgc             k = K_R K_T + (m-K_R) + (n-K_T)
  ST-CC C6   independencia dentro de cada bloque de cgc            k = K_T m + K_R n - K_R K_T
  Rejilla    partición global (K_R, K_T) evaluada con la base C6   k = K_T m + K_R n - K_R K_T
  AST-CC     árbol de bloques anidados (hojas con independencia)   k = sum_L (|I_L| + |J_L| - 1)

Las rejillas usan la base C6 porque es la familia de las hojas del árbol: un árbol cuyas
hojas son los bloques base es exactamente la rejilla con C6. Para que sean un rival
exigente, cada (K_R, K_T) toma la mejor de las particiones que optimizan C2 y C5
(BBAC_I, con búsqueda local), con las que se minimiza D_I(O || C6) de forma indirecta.

Como la matriz es una suma de días con varianza mayor que la de Poisson, J se reporta
también con la devianza dividida entre la dispersión de Pearson del modelo
[DZ][DT][ZT] (J_phi, estimador D de Calibracion/01_sobredispersion.py): con la
verosimilitud de Poisson, el BIC sobrestima la evidencia y favorece particiones
demasiado finas.

Limitación: ni J ni D_I descuentan la búsqueda de la partición (BBAC) ni la del
árbol, que usan los mismos datos; comparan el ajuste a igual número de parámetros,
no el rendimiento fuera de muestra (eso es la estabilidad, paso siguiente).

Salidas:
  Resultados/AST-CC/comparacion.txt
  Resultados/AST-CC/comparacion_{periodo}.csv
  Resultados/AST-CC/comparacion_{periodo}.png
"""

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.bbac import bbac, d_I, divergencia  # noqa: E402
from astcc.nulas import ajuste_sin_interaccion_triple  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
SALIDA = RAIZ / "Resultados/AST-CC"
SALIDA.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261029
K_ST = {"weekday": (8, 4), "holiday": (6, 4)}        # K_R, K_T de Liu et al. (2021)
KR_REJILLA = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32]
KT_REJILLA = [2, 3, 4, 6, 8, 12]
NRUNS = 20
DELTAS = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]


def phi_pearson(X):
    """Dispersión de Pearson del modelo [DZ][DT][ZT] (estimador D de la Propuesta)."""
    D, Z, T = X.shape
    E = ajuste_sin_interaccion_triple(X)
    m = E > 0
    return float(np.sum((X[m] - E[m]) ** 2 / E[m]) / ((D - 1) * (Z - 1) * (T - 1)))


def k_rejilla(m, n, KR, KT):
    return KT * m + KR * n - KR * KT


def k_modelo(base, m, n, KR, KT):
    if base == "C2":
        return KR * KT
    if base == "C5":
        return KR * KT + (m - KR) + (n - KT)
    return k_rejilla(m, n, KR, KT)


def fila(modelo, config, k, D_I, N, phi):
    return {"modelo": modelo, "config": config, "k": int(k), "D_I": D_I,
            "J": D_I + 0.5 * np.log(N) * k, "J_phi": D_I / phi + 0.5 * np.log(N) * k}


def rejillas(O, rng, periodo):
    """Mejor partición global para cada (K_R, K_T): la de menor D_I(O || C6) entre las
    que optimizan C2 y C5."""
    m, n = O.shape
    N = O.sum()
    filas = []
    for KR in KR_REJILLA:
        for KT in KT_REJILLA:
            if KR > m or KT > n:
                continue
            mejor = None
            for base in ("C2", "C5"):
                r = bbac(O, KR, KT, base=base, nruns=NRUNS, rng=rng)
                d6 = divergencia(O, r.rho, r.tau, "C6", KR, KT)
                if mejor is None or d6 < mejor[0]:
                    mejor = (d6, base)
            filas.append({**fila("Rejilla", f"{KR}x{KT}", k_rejilla(m, n, KR, KT), mejor[0], N, 1.0),
                          "base": mejor[1], "KR": KR, "KT": KT})
        print(f"  {periodo}: rejillas con K_R = {KR} listas", flush=True)
    return filas


def frontera(filas):
    """Rejillas no dominadas: ninguna otra tiene menos parámetros y menor D_I."""
    orden = sorted(filas, key=lambda f: (f["k"], f["D_I"]))
    salida, mejor = [], np.inf
    for f in orden:
        if f["D_I"] < mejor - 1e-9:
            salida.append(f)
            mejor = f["D_I"]
    return salida


def mejor_con_a_lo_sumo(filas, k):
    cand = [f for f in filas if f["k"] <= k]
    return min(cand, key=lambda f: f["D_I"]) if cand else None


def menos_parametros_que_igualan(filas, D_I):
    """Rejilla con menos parámetros cuyo D_I no supera al dado (None si ninguna llega)."""
    cand = [f for f in filas if f["D_I"] <= D_I]
    return min(cand, key=lambda f: f["k"]) if cand else None


def main():
    t_total = time.time()
    rng = np.random.default_rng(SEMILLA)
    L = ["Comparación de AST-CC con ST-CC y con rejillas globales (modelos de Poisson sobre conteos)",
         "J = D_I + (1/2) log(N) k;  J_phi usa D_I / phi (dispersión de Pearson, modelo [DZ][DT][ZT])", ""]
    for periodo, (kr, kt) in K_ST.items():
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        O = X.sum(0)
        m, n = O.shape
        N = float(O.sum())
        phi = phi_pearson(X)
        L += [f"{periodo}: {m} x {n}, N = {int(N):,}, D = {X.shape[0]} días, phi = {phi:.2f}, "
              f"saturado: k = {m * n}", ""]
        print(L[-2], flush=True)

        # ST-CC (cgc) en sus tres bases
        rho_c = np.load(DATOS / f"ST-CC/{periodo}_row_clusters.npy")
        tau_c = np.load(DATOS / f"ST-CC/{periodo}_col_clusters.npy")
        st = [fila(f"ST-CC {b}", f"{kr}x{kt}", k_modelo(b, m, n, kr, kt),
                   divergencia(O, rho_c, tau_c, b, kr, kt), N, phi) for b in ("C2", "C5", "C6")]

        # Rejillas globales
        rej = rejillas(O, rng, periodo)
        for f in rej:
            f["J_phi"] = f["D_I"] / phi + 0.5 * np.log(N) * f["k"]
        pareto = frontera(rej)

        # AST-CC con la partición base C5 de AST-CC/01
        rho = np.load(DATOS / f"AST-CC/{periodo}_C5_rho.npy")
        tau = np.load(DATOS / f"AST-CC/{periodo}_C5_tau.npy")
        ast = []
        for delta in DELTAS:
            t0 = time.time()
            res = ASTCC(Parametros(delta=delta), np.random.default_rng(SEMILLA)).ajustar(X, rho, tau)
            k = sum(len(h.filas) + len(h.cols) - 1 for h in res.hojas())
            f = fila("AST-CC", f"d={delta:g}", k, d_I(O, res.aproximacion(O)), N, phi)
            f.update(hojas=len(res.hojas()), cortes=len(res.internos()))
            ast.append(f)
            print(f"  {periodo}: AST-CC delta = {delta:g}: {len(res.hojas())} hojas, k = {k} "
                  f"({time.time() - t0:.1f} s)", flush=True)

        # --- Tabla 1: AST-CC frente a la mejor rejilla con a lo sumo los mismos parámetros
        L += ["  A. A igual número de parámetros: AST-CC frente a la mejor rejilla global con a lo sumo "
              "esos parámetros;",
              "     a la derecha, la rejilla con menos parámetros que iguala el D_I de AST-CC", "",
              f"  {'delta':>7} {'hojas':>5} {'k':>5} {'D_I AST-CC':>11}   {'rejilla':>7} {'base':>4} {'k':>5} "
              f"{'D_I rejilla':>11}  {'reducción':>9}   {'iguala':>7} {'k':>5} {'ahorro':>7}"]
        for f in ast:
            r = mejor_con_a_lo_sumo(rej, f["k"])
            q = menos_parametros_que_igualan(rej, f["D_I"])
            d = f["config"][2:]
            izq = f"  {d:>7} {f['hojas']:>5} {f['k']:>5} {f['D_I']:>11,.0f}   "
            cmp_ = f"{'—':>7}" if r is None else (f"{r['config']:>7} {r['base']:>4} {r['k']:>5} "
                                                  f"{r['D_I']:>11,.0f}  {1 - f['D_I'] / r['D_I']:>9.1%}")
            igual = (f"{'ninguna':>7}" if q is None else f"{q['config']:>7} {q['k']:>5} {q['k'] - f['k']:>+7}")
            L.append(izq + cmp_ + "   " + igual)
        L.append("")

        # --- Tabla 2: criterios de información
        todos = st + rej + ast
        best = {c: min(todos, key=lambda f: f[c]) for c in ("J", "J_phi")}
        mejor_rej = {c: min(rej, key=lambda f: f[c]) for c in ("J", "J_phi")}
        mejor_ast = {c: min(ast, key=lambda f: f[c]) for c in ("J", "J_phi")}
        L += ["  B. Criterio de información (menor es mejor)", "",
              f"  {'modelo':<10} {'config':<10} {'k':>5} {'D_I':>11} {'J':>11} {'J_phi':>11}"]
        for f in st + [mejor_rej["J"], mejor_rej["J_phi"]] + ast:
            marca = ""
            if f is mejor_rej["J"]:
                marca = "  <- mejor rejilla por J"
            if f is mejor_rej["J_phi"]:
                marca += "  <- mejor rejilla por J_phi"
            if f is mejor_ast["J"]:
                marca = "  <- mejor AST-CC por J"
            if f is mejor_ast["J_phi"]:
                marca += "  <- mejor AST-CC por J_phi"
            L.append(f"  {f['modelo']:<10} {f['config']:<10} {f['k']:>5} {f['D_I']:>11,.0f} "
                     f"{f['J']:>11,.0f} {f['J_phi']:>11,.0f}{marca}")
        for c, nombre in (("J", "J"), ("J_phi", "J_phi")):
            L.append(f"  Mínimo de {nombre}: {best[c]['modelo']} {best[c]['config']}; "
                     f"AST-CC frente a la mejor rejilla: "
                     f"{mejor_ast[c][c] - mejor_rej[c][c]:+,.0f}")
        L.append("")

        # --- CSV y figura
        tabla = pd.DataFrame(st + rej + ast)
        tabla.to_csv(SALIDA / f"comparacion_{periodo}.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter([f["k"] for f in rej], [f["D_I"] for f in rej], s=14, c="0.75", label="rejillas globales")
        ax.plot([f["k"] for f in pareto], [f["D_I"] for f in pareto], "-", c="0.35", lw=1.2,
                label="mejor rejilla con ≤ k parámetros")
        ax.plot([f["k"] for f in ast], [f["D_I"] for f in ast], "o-", c="C3", label="AST-CC (δ)")
        for f in ast:
            ax.annotate(f["config"][2:], (f["k"], f["D_I"]), textcoords="offset points", xytext=(5, 4),
                        fontsize=7, color="C3")
        ax.plot([f["k"] for f in st], [f["D_I"] for f in st], "s", c="C0", label="ST-CC (C2, C5, C6)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([30, 50, 100, 200, 300, 500, 700])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("parámetros libres k")
        ax.set_ylabel("divergencia $D_I(O\\,\\|\\,\\hat O)$")
        ax.set_title(f"{periodo}: ajuste frente a número de parámetros")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(SALIDA / f"comparacion_{periodo}.png", dpi=150)
        plt.close(fig)

    L.append(f"Tiempo total: {(time.time() - t_total) / 60:.1f} min")
    texto = "\n".join(L)
    print("\n" + texto)
    (SALIDA / "comparacion.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
