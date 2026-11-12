"""
Refinamiento K-means "non-checkerboard" sobre las hojas de AST-CC
=================================================================
Misma idea que ST-CC/04_refinamiento_kmeans.py (Sección IV-D, Liu et al. 2021), con las
hojas del árbol de AST-CC como muestras en lugar de los K_R x K_T bloques de la rejilla:

    "The mean and standard deviation of each checkerboard co-cluster were employed as
    the attribute dimensions of the samples in K-means."

Para cada hoja se calculan la media y la desviación típica de sus celdas (drop-offs
promedio por día, la misma matriz que ve ST-CC), cada atributo se normaliza a [0, 1] como
hace cgc.kmeans, y se agrupan con K-means (50 inicializaciones aleatorias, semilla 0).
Se muestran la silhouette y el SSE para K de 2 a 8 (K = 2 solo de referencia, el artículo
lo excluye de la selección) y se fija K = 4, el valor que reporta el artículo. Los
grupos se etiquetan Very Low / Low / Medium / High por su promedio ascendente, que es el
de las celdas de todas sus hojas.

Dos hojas lejanas en la teselación, incluso de bloques base distintos, pueden quedar en
el mismo grupo: esa es la señal "non-checkerboard".

Se usa el árbol de AST-CC/07 (partición base C5, delta = 0.005).
Salidas: Resultados/AST-CC/figuras/kmeans_hojas_{periodo}.png, elbow_hojas_{periodo}.png
         Resultados/AST-CC/kmeans_hojas_{periodo}.csv, kmeans_hojas.txt
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # evita el aviso de memory leak de sklearn + MKL en Windows

import importlib.util  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

# Se reutiliza la geometría y las constantes de 07_visualizacion.py
_spec = importlib.util.spec_from_file_location("vis07", RAIZ / "AST-CC/07_visualizacion.py")
vis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vis)

DATOS, SALIDA, FIGURAS = vis.DATOS, vis.SALIDA, vis.FIGURAS
RANGO_K = range(2, 9)
K_FINAL = 4
ESTADOS = ["Very Low", "Low", "Medium", "High"]
COLORES = ["#2ca02c", "#f2e341", "#ff8c00", "#d62728"]     # los de ST-CC/04
SIGLAS = ["VL", "L", "M", "H"]


def atributos(O, D, hojas):
    """(media, desviación típica) de las celdas de cada hoja, en drop-offs promedio por día."""
    V = O / D
    return np.array([[V[np.ix_(h.filas, h.cols)].mean(), V[np.ix_(h.filas, h.cols)].std()] for h in hojas])


def normalizar(a):
    """Cada atributo a [0, 1], como cgc.kmeans."""
    mn, mx = a.min(0), a.max(0)
    return np.divide(a - mn, mx - mn, out=np.zeros_like(a), where=(mx - mn) != 0)


def km(Z, k):
    return KMeans(n_clusters=k, init="random", n_init=50, random_state=0).fit(Z)


def figura_codo(periodo, ks, silhouette, sse):
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(ks, silhouette, marker="s", color="black")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Silhouette promedio")
    ax1.axvline(K_FINAL, color="gray", linestyle="--", linewidth=1)
    ax2 = ax1.twinx()
    ax2.plot(ks, sse, marker="o", linestyle=":", color="red")
    ax2.set_ylabel("SSE", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax1.set_title(f"{vis.TITULO[periodo]}: silhouette + SSE vs K, hojas de AST-CC\n"
                  f"(línea punteada = K={K_FINAL}, el que usa ST-CC)")
    fig.tight_layout()
    fig.savefig(FIGURAS / f"elbow_hojas_{periodo}.png", dpi=150)
    plt.close(fig)


def figura_kmeans(periodo, res, O, rho, tau, estado_de):
    orden_g, orden_h, ancho, alto, x_de, y_de = vis.disposicion_global(O, rho, tau)
    fig, ax = plt.subplots(figsize=(9, 10))
    for raiz in res.raices:
        g, h = rho[raiz.filas[0]], tau[raiz.cols[0]]
        for hoja, x, y, w, alt in vis.disponer(raiz, x_de[h], y_de[g]):
            e = estado_de[id(hoja)]
            refinada = hoja.profundidad > 0
            ax.add_patch(Rectangle((x, y), w, alt, facecolor=COLORES[e], edgecolor="0.25",
                                   linewidth=1.4 if refinada else 0.5))
            ax.text(x + w / 2, y + alt / 2, SIGLAS[e], ha="center", va="center", fontsize=10,
                    fontweight="bold")
        ax.add_patch(Rectangle((x_de[h], y_de[g]), ancho[h], alto[g], fill=False,
                               edgecolor="black", linewidth=1.6))
    ax.set_xlim(0, len(tau))
    ax.set_ylim(0, len(rho))
    ax.set_xticks([x_de[h] + ancho[h] / 2 for h in orden_h])
    ax.set_xticklabels(["\n".join(vis.FRANJAS[j] for j in np.where(tau == h)[0]) for h in orden_h], fontsize=6)
    ax.set_yticks([y_de[g] + alto[g] / 2 for g in orden_g])
    ax.set_yticklabels([f"{alto[g]} regiones" for g in orden_g], fontsize=7)
    ax.set_xlabel("grupos de intervalos de la partición base (orden de volumen creciente)")
    ax.set_ylabel("grupos de regiones (volumen creciente hacia arriba)")
    ax.set_title(f"{vis.TITULO[periodo].capitalize()}: {K_FINAL} clusters refinados (non-checkerboard) "
                 f"sobre las {len(res.hojas())} hojas de AST-CC\n(δ = {vis.DELTA}; borde grueso = hoja "
                 f"producida por un corte)", fontsize=9)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="0.25") for c in COLORES],
              labels=ESTADOS, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURAS / f"kmeans_hojas_{periodo}.png", dpi=150)
    plt.close(fig)


def figura_antes_despues(periodo, res, O, rho, tau, estado_de, estado_base):
    """Dos paneles con la misma geometría y la misma escala de colores: a la izquierda la
    partición base (un estado por bloque, como en ST-CC) y a la derecha las hojas de AST-CC.
    Los bloques sin cortes tienen el mismo color en ambos; los refinados llevan borde azul."""
    orden_g, orden_h, ancho, alto, x_de, y_de = vis.disposicion_global(O, rho, tau)
    nombre_g = {g: f"R{i + 1}" for i, g in enumerate(orden_g)}
    nombre_h = {h: f"T{i + 1}" for i, h in enumerate(orden_h)}
    fig, ejes = plt.subplots(1, 2, figsize=(15, 9.5), sharey=True)
    for ax, despues in zip(ejes, (False, True)):
        for raiz in res.raices:
            g, h = rho[raiz.filas[0]], tau[raiz.cols[0]]
            if despues:
                piezas = [(estado_de[id(hoja)], x, y, w, alt) for hoja, x, y, w, alt in vis.disponer(raiz, x_de[h], y_de[g])]
            else:
                piezas = [(estado_base[id(raiz)], x_de[h], y_de[g], ancho[h], alto[g])]
            for e, x, y, w, alt in piezas:
                ax.add_patch(Rectangle((x, y), w, alt, facecolor=COLORES[e], edgecolor="0.25", linewidth=0.8))
                ax.text(x + w / 2, y + alt / 2, SIGLAS[e], ha="center", va="center", fontsize=10, fontweight="bold")
        for raiz in sorted(res.raices, key=lambda r: not r.es_hoja):      # los refinados al final, encima
            g, h = rho[raiz.filas[0]], tau[raiz.cols[0]]
            ax.add_patch(Rectangle((x_de[h], y_de[g]), ancho[h], alto[g], fill=False,
                                   edgecolor="black" if raiz.es_hoja else "#1f4fd8",
                                   linewidth=1.4 if raiz.es_hoja else 3.5))
        ax.set_xlim(0, len(tau))
        ax.set_ylim(0, len(rho))
        ax.set_xticks([x_de[h] + ancho[h] / 2 for h in orden_h])
        ax.set_xticklabels([nombre_h[h] + "\n" + "\n".join(vis.FRANJAS[j] for j in np.where(tau == h)[0])
                            for h in orden_h], fontsize=7)
        ax.set_yticks([y_de[g] + alto[g] / 2 for g in orden_g])
        ax.set_yticklabels([f"{nombre_g[g]}\n({alto[g]} regiones)" for g in orden_g], fontsize=8)
        ax.tick_params(labelleft=True)
        ax.set_xlabel("grupos de intervalos (volumen creciente)")
        ax.set_title(f"AST-CC: {len(res.hojas())} hojas (δ = {vis.DELTA})" if despues else
                     f"Antes de refinar: rejilla base de {len(res.raices)} bloques, un estado por bloque (como ST-CC)",
                     fontsize=10)
    ejes[0].set_ylabel("grupos de regiones (volumen creciente hacia arriba)")
    ejes[1].legend(handles=[Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="0.25") for c in COLORES]
                   + [Rectangle((0, 0), 1, 1, fill=False, edgecolor="#1f4fd8", linewidth=2.5)],
                   labels=ESTADOS + ["bloque refinado"], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.suptitle(f"{vis.TITULO[periodo].capitalize()}: la misma rejilla antes y después de los cortes de AST-CC "
                 f"(misma escala de estados en ambos paneles)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURAS / f"antes_despues_{periodo}.png", dpi=150)
    plt.close(fig)
    return nombre_g, nombre_h


def main():
    L = []
    for periodo in ("weekday", "holiday"):
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        O, D = X.sum(0), X.shape[0]
        rho = np.load(DATOS / f"AST-CC/{periodo}_C5_rho.npy")
        tau = np.load(DATOS / f"AST-CC/{periodo}_C5_tau.npy")
        res = ASTCC(Parametros(delta=vis.DELTA), np.random.default_rng(vis.SEMILLA)).ajustar(X, rho, tau)
        hojas = res.hojas()
        Z = normalizar(atributos(O, D, hojas))

        L += [f"{periodo}: K-means sobre las {len(hojas)} hojas de AST-CC (media y desviación típica "
              f"de sus celdas, normalizadas a [0, 1])", "",
              "  Silhouette y SSE (K = 2 solo de referencia; el artículo lo excluye de la selección):"]
        ks, sil, sse = list(RANGO_K), [], []
        for k in ks:
            ajuste = km(Z, k)
            sil.append(silhouette_score(Z, ajuste.labels_))
            sse.append(ajuste.inertia_)
            L.append(f"    K={k}: silhouette = {sil[-1]:.4f}, SSE = {sse[-1]:.3f}")
        L.append(f"  Mejor silhouette entre K >= 3: K = {ks[1:][int(np.argmax(sil[1:]))]}; se usa K = {K_FINAL}, "
                 f"como ST-CC")
        figura_codo(periodo, ks, sil, sse)

        ajuste = km(Z, K_FINAL)
        etiqueta = ajuste.labels_
        V = O / D
        celdas = {c: np.concatenate([V[np.ix_(h.filas, h.cols)].ravel()
                                     for h, e in zip(hojas, etiqueta) if e == c]) for c in range(K_FINAL)}
        orden = sorted(celdas, key=lambda c: celdas[c].mean())
        rango = {c: r for r, c in enumerate(orden)}
        estado_de = {id(h): rango[e] for h, e in zip(hojas, etiqueta)}

        L += ["", f"  Con K = {K_FINAL}: silhouette = {silhouette_score(Z, etiqueta):.4f}", ""]
        filas_csv = []
        for c in orden:
            miembros = [h for h, e in zip(hojas, etiqueta) if e == c]
            bloques = {(rho[h.filas[0]], tau[h.cols[0]]) for h in miembros}
            L.append(f"  {ESTADOS[rango[c]]} (promedio {celdas[c].mean():.1f}, {len(miembros)} hojas de "
                     f"{len(bloques)} bloques base distintos, {sum(len(h.filas) * len(h.cols) for h in miembros)} celdas)")
        for h, e in zip(hojas, etiqueta):
            v = V[np.ix_(h.filas, h.cols)]
            filas_csv.append({"profundidad": h.profundidad, "tipo": h.estado, "n_regiones": len(h.filas),
                              "n_intervalos": len(h.cols), "media": v.mean(), "desv": v.std(),
                              "estado": ESTADOS[rango[e]]})
        pd.DataFrame(filas_csv).to_csv(SALIDA / f"kmeans_hojas_{periodo}.csv", index=False)
        figura_kmeans(periodo, res, O, rho, tau, estado_de)

        # ¿Cuántas hojas cambian de estado respecto del bloque base que las contiene?
        mezclan = 0
        for r in res.raices:
            estados = {estado_de[id(h)] for h in r.recorrer() if h.es_hoja}
            mezclan += len(estados) > 1
        refinados = sum(1 for r in res.raices if not r.es_hoja)
        L += ["", f"  De los {refinados} bloques base refinados, {mezclan} contienen hojas de estados distintos "
                  f"(el corte separa niveles de actividad distintos)", ""]

        # Antes y después con la misma escala: cada bloque base recibe el estado del centroide
        # más cercano del K-means de las hojas (misma normalización). Un bloque sin cortes es
        # él mismo una hoja, así que conserva su color.
        a_hojas, a_base = atributos(O, D, hojas), atributos(O, D, res.raices)
        mn, mx = a_hojas.min(0), a_hojas.max(0)
        pred = ajuste.predict((a_base - mn) / np.where(mx - mn != 0, mx - mn, 1))
        estado_base = {id(r): (estado_de[id(r)] if r.es_hoja else rango[p]) for r, p in zip(res.raices, pred)}
        nombre_g, nombre_h = figura_antes_despues(periodo, res, O, rho, tau, estado_de, estado_base)
        L += ["  Bloques refinados (figura antes_despues): estado del bloque entero -> estados de sus hojas",
              "  (media = drop-offs promedio por celda y día)"]
        for r, ab in zip(res.raices, a_base):
            if r.es_hoja:
                continue
            L.append(f"    {nombre_g[rho[r.filas[0]]]}-{nombre_h[tau[r.cols[0]]]} ({len(r.filas)} regiones, "
                     f"{' '.join(vis.FRANJAS[j] for j in r.cols)}): {SIGLAS[estado_base[id(r)]]} (media {ab[0]:.0f})")
            for h in r.recorrer():
                if h.es_hoja:
                    v = V[np.ix_(h.filas, h.cols)]
                    L.append(f"        -> {SIGLAS[estado_de[id(h)]]:<2} (media {v.mean():.0f}): {len(h.filas)} regiones x "
                             f"{len(h.cols)} intervalos")
        L.append("")
        print(f"{periodo}: listo", flush=True)

    texto = "\n".join(L)
    print("\n" + texto)
    (SALIDA / "kmeans_hojas.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
