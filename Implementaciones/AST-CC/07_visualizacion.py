"""
Representación de los resultados de AST-CC
==========================================
Cap_3.tex, sec:prop-representacion: los productos que acompañan al árbol, para cada
periodo, con la partición base de BBAC C5 (AST-CC/01) y delta = 0.005 (el árbol de
AST-CC/02, con la misma semilla):

  Matriz teselada      cada bloque base con sus cortes y, dentro de cada hoja, las celdas
                       reales (drop-offs promedio por día) con la paleta y el orden por
                       volumen del checkerboard de ST-CC (ST-CC/03); las líneas rojas
                       son los límites de las hojas. Las hojas producidas por un corte
                       llevan su frecuencia de soporte (AST-CC/06) y, si es menor que
                       0.5, borde discontinuo: no debe interpretarse como hallazgo.
                       Cada bloque se ordena por sus propios cortes, así que las filas
                       y columnas de dentro de un bloque no son comparables con las del
                       bloque vecino; los bloques sí están en el orden de volumen
                       creciente de los grupos base.
  Actividad relativa   la misma teselación con cada hoja de un solo color: log2 de su
                       nivel de actividad l_L (eq:prop-lift), rojo = más activa de lo
                       que corresponde al volumen de sus regiones e intervalos, azul =
                       menos. Quita el volumen, que domina la figura anterior.
  Mapas de profundidad profundidad máxima de las hojas que contienen a cada región
                       (mapa de Manhattan) y a cada intervalo de tiempo (barras).
  Bloques refinados    para cada corte, las regiones que separa y sus intervalos.
  Árbol anotado        cada nodo con su corte, G2 del corte, ganancia corregida y
                       p-valor, y las zonas de cada hijo (texto).

Salidas: Resultados/AST-CC/figuras/{teselada,actividad,profundidad,refinados}_{periodo}.png
         Resultados/AST-CC/arbol_{periodo}.txt
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc.arbol import ASTCC, Parametros  # noqa: E402
from astcc.estadisticos import esperados  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS = RAIZ / "DATA/procesados"
CRUDOS = RAIZ / "DATA/crudos"
SALIDA = RAIZ / "Resultados/AST-CC"
FIGURAS = SALIDA / "figuras"
FIGURAS.mkdir(parents=True, exist_ok=True)

SEMILLA = 20261004                 # la de 02_refinamiento.py
DELTA = 0.005
FRANJAS = [f"{h:02d}-{h + 2:02d}" for h in range(0, 24, 2)]
COLOR_PROFUNDIDAD = ["#d9d9d9", "#fdae6b", "#e6550d", "#a63603"]
COLOR_HIJOS = ["#2b8cbe", "#e34a33", "#31a354", "#fec44f"]
TITULO = {"weekday": "días laborables", "holiday": "periodo feriado"}


# ----------------------------------------------------------------------------
# Geometría del árbol
# ----------------------------------------------------------------------------

def disponer(nodo, x0, y0):
    """Rectángulos (hoja, x, y, ancho, alto) de un bloque, con el origen en (x0, y0) y y
    creciendo hacia arriba. Los hijos de un corte se colocan uno al lado del otro."""
    w, h = len(nodo.cols), len(nodo.filas)
    if nodo.es_hoja:
        return [(nodo, x0, y0, w, h)]
    c = nodo.hijos
    if nodo.corte == "fil":
        return disponer(c[0], x0, y0) + disponer(c[1], x0, y0 + len(c[0].filas))
    if nodo.corte == "col":
        return disponer(c[0], x0, y0) + disponer(c[1], x0 + len(c[0].cols), y0)
    h0, w0 = len(c[0].filas), len(c[0].cols)             # conjunto: hijo k = (r, c) con k = 2 r + c
    salida = []
    for k, hijo in enumerate(c):
        r, col = divmod(k, 2)
        salida += disponer(hijo, x0 + col * w0, y0 + r * h0)
    return salida


def actividad(nodo, E_global):
    """l_L = N_L / sum E_ij con la independencia de toda la matriz (eq:prop-lift)."""
    return nodo.N / E_global[np.ix_(nodo.filas, nodo.cols)].sum()


def profundidad_por_indice(res, m, n):
    pr, pc = np.zeros(m, int), np.zeros(n, int)
    for h in res.hojas():
        pr[h.filas] = np.maximum(pr[h.filas], h.profundidad)
        pc[h.cols] = np.maximum(pc[h.cols], h.profundidad)
    return pr, pc


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------

def disposicion_global(O, rho, tau):
    """Posición de cada bloque base en la matriz teselada: los grupos de regiones y de
    intervalos van en orden de volumen medio creciente (como el checkerboard de ST-CC).
    Devuelve (orden_g, orden_h, ancho, alto, x_de, y_de)."""
    gr, gt = np.unique(rho), np.unique(tau)
    orden_g = sorted(gr, key=lambda g: O.sum(1)[rho == g].mean())
    orden_h = sorted(gt, key=lambda h: O.sum(0)[tau == h].mean())
    ancho = {h: int((tau == h).sum()) for h in gt}
    alto = {g: int((rho == g).sum()) for g in gr}
    x_de = {h: sum(ancho[k] for k in orden_h[:i]) for i, h in enumerate(orden_h)}
    y_de = {g: sum(alto[k] for k in orden_g[:i]) for i, g in enumerate(orden_g)}
    return orden_g, orden_h, ancho, alto, x_de, y_de


def columnas_dibujadas(res, O, tau, h):
    """Intervalos del grupo base h en el orden en que se dibujan sus columnas: por volumen
    creciente, como el checkerboard de ST-CC (dentro de cada hoja las columnas se ordenan
    por su volumen, ver figura_teselada). Si algún bloque del grupo tiene un corte de
    columnas, los hijos se colocan uno junto al otro y el orden deja de ser común a todos
    los bloques; en ese caso se devuelve el orden cronológico, que solo indica los miembros."""
    idx = np.where(tau == h)[0]
    hay_corte_col = any(n.corte in ("col", "conj") for r in res.raices if tau[r.cols[0]] == h
                        for n in r.recorrer())
    return idx if hay_corte_col else idx[np.argsort(O.sum(0)[idx])]


def figura_teselada(periodo, res, O, D, rho, tau, soporte, modo):
    """modo = 'datos': dentro de cada hoja, las celdas reales (drop-offs promedio por día,
    la paleta y el orden por volumen del checkerboard de ST-CC, ST-CC/03). modo =
    'actividad': cada hoja de un solo color, log2 de su nivel de actividad l_L."""
    E_global = esperados(O)
    orden_g, orden_h, ancho, alto, x_de, y_de = disposicion_global(O, rho, tau)
    fig, ax = plt.subplots(figsize=(9, 10))
    datos = modo == "datos"
    if datos:
        norma, cmap = Normalize(0, float(O.max() / D)), plt.get_cmap("YlOrRd")
    else:
        norma, cmap = Normalize(-1, 1), plt.get_cmap("RdBu_r")
    for raiz in res.raices:
        g, h = rho[raiz.filas[0]], tau[raiz.cols[0]]
        for hoja, x, y, w, alt in disponer(raiz, x_de[h], y_de[g]):
            refinada = hoja.profundidad > 0
            sop = soporte.get(id(hoja)) if (soporte is not None and refinada) else None
            borde = dict(edgecolor="red" if datos else ("0.25" if refinada else "0.6"),
                         linewidth=(1.4 if refinada else 0.6) if datos else (1.3 if refinada else 0.5),
                         linestyle="--" if (sop is not None and sop < 0.5) else "-")
            if datos:
                filas = hoja.filas[np.argsort(O.sum(1)[hoja.filas])]
                cols = hoja.cols[np.argsort(O.sum(0)[hoja.cols])]
                ax.imshow(O[np.ix_(filas, cols)] / D, extent=(x, x + w, y, y + alt), origin="lower",
                          cmap=cmap, norm=norma, aspect="auto", interpolation="nearest")
                ax.add_patch(Rectangle((x, y), w, alt, fill=False, **borde))
            else:
                l = actividad(hoja, E_global)
                ax.add_patch(Rectangle((x, y), w, alt, facecolor=cmap(norma(np.log2(max(l, 1e-3)))), **borde))
            if refinada and sop is not None and w >= 1 and alt >= 2:
                ax.text(x + w / 2, y + alt / 2, f"{sop:.2f}", ha="center", va="center", fontsize=6,
                        bbox=dict(facecolor="white", alpha=0.7, pad=0.8, linewidth=0) if datos else None)
        ax.add_patch(Rectangle((x_de[h], y_de[g]), ancho[h], alto[g], fill=False,
                               edgecolor="black", linewidth=1.6))
    ax.set_xlim(0, len(tau))
    ax.set_ylim(0, len(rho))
    ax.set_xticks([x_de[h] + ancho[h] / 2 for h in orden_h])
    ax.set_xticklabels(["\n".join(FRANJAS[j] for j in columnas_dibujadas(res, O, tau, h)) for h in orden_h],
                       fontsize=6)
    ax.set_yticks([y_de[g] + alto[g] / 2 for g in orden_g])
    ax.set_yticklabels([f"{alto[g]} regiones" for g in orden_g], fontsize=7)
    ax.set_xlabel("grupos de intervalos de la partición base (orden de volumen creciente)\n"
                  "dentro de cada grupo, los intervalos van de arriba hacia abajo = de izquierda a derecha")
    ax.set_ylabel("grupos de regiones (volumen creciente hacia arriba)")
    nombre = "teselada" if datos else "actividad"
    ax.set_title(f"{'Matriz teselada de AST-CC' if datos else 'Actividad relativa de las hojas de AST-CC'}, "
                 f"{TITULO[periodo]} (δ = {DELTA})\n"
                 f"{len(res.hojas())} hojas; borde grueso = hoja producida por un corte; "
                 f"discontinuo = soporte < 0.5", fontsize=9)
    barra = fig.colorbar(plt.cm.ScalarMappable(norma, cmap), ax=ax, fraction=0.03, pad=0.02)
    if datos:
        barra.set_label("Drop-off promedio por día")
    else:
        barra.set_ticks([-1, -0.5, 0, 0.5, 1])
        barra.set_ticklabels(["0.5", "0.71", "1", "1.41", "2"])
        barra.set_label("nivel de actividad $\\ell_L$ (respecto de la independencia global)")
    fig.tight_layout()
    fig.savefig(FIGURAS / f"{nombre}_{periodo}.png", dpi=150)
    plt.close(fig)


def figura_profundidad(periodo, res, zonas_gdf, zonas):
    m, n = len(zonas), len(FRANJAS)
    pr, pc = profundidad_por_indice(res, m, n)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw={"width_ratios": [1, 1.1]})
    prof = pd.Series(pr, index=zonas)
    gdf = zonas_gdf.assign(prof=zonas_gdf["LocationID"].map(prof))
    gdf.plot(ax=a1, color=[COLOR_PROFUNDIDAD[min(int(p), 3)] for p in gdf["prof"]],
             edgecolor="white", linewidth=0.4)
    a1.set_axis_off()
    a1.set_title("profundidad máxima por región", fontsize=10)
    a1.legend(handles=[Rectangle((0, 0), 1, 1, color=COLOR_PROFUNDIDAD[d]) for d in range(max(pr.max(), 1) + 1)],
              labels=[f"{d}" + (" (solo la partición base)" if d == 0 else "") for d in range(max(pr.max(), 1) + 1)],
              loc="upper left", fontsize=8, title="profundidad", title_fontsize=8)
    a2.bar(range(n), pc, color=[COLOR_PROFUNDIDAD[min(int(p), 3)] for p in pc])
    a2.set_xticks(range(n))
    a2.set_xticklabels(FRANJAS, rotation=90, fontsize=8)
    a2.set_yticks(range(0, max(int(pc.max()), 1) + 1))
    a2.set_ylabel("profundidad máxima")
    a2.set_title("profundidad máxima por intervalo", fontsize=10)
    fig.suptitle(f"Dónde y cuándo la partición base resulta insuficiente, {TITULO[periodo]} (δ = {DELTA})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURAS / f"profundidad_{periodo}.png", dpi=150)
    plt.close(fig)
    return pr, pc


def figura_refinados(periodo, res, O, rho, tau, zonas_gdf, zonas):
    internos = res.internos()
    if not internos:
        return
    E_global = esperados(O)
    nc = min(3, len(internos))
    nf = int(np.ceil(len(internos) / nc))
    fig, ejes = plt.subplots(nf, nc, figsize=(4.6 * nc, 5.4 * nf), squeeze=False)
    pos = {z: i for i, z in enumerate(zonas)}
    for ax in ejes.ravel():
        ax.set_axis_off()
    for ax, nodo in zip(ejes.ravel(), internos):
        g, h = rho[nodo.filas[0]], tau[nodo.cols[0]]
        color = pd.Series("#f2f2f2", index=zonas)
        lineas = []
        if nodo.corte in ("fil", "conj"):
            grupos = ([nodo.hijos[0], nodo.hijos[1]] if nodo.corte == "fil" else nodo.hijos)
            for k, hijo in enumerate(grupos):
                color.iloc[hijo.filas] = COLOR_HIJOS[k]
        else:
            color.iloc[nodo.filas] = "#756bb1"
        for k, hijo in enumerate(nodo.hijos):
            et = (f"hijo {k + 1} ({['azul', 'rojo', 'verde', 'amarillo'][k]})" if nodo.corte != "col"
                  else f"hijo {k + 1}")
            lineas.append(f"{et}: {len(hijo.filas)} regiones, {len(hijo.cols)} intervalos, "
                          f"ℓ = {actividad(hijo, E_global):.2f}")
        gdf = zonas_gdf.assign(c=zonas_gdf["LocationID"].map(lambda z: color.iloc[pos[z]]))
        gdf.plot(ax=ax, color=gdf["c"], edgecolor="white", linewidth=0.4)
        ints = ", ".join(FRANJAS[j] for j in nodo.cols)
        if nodo.corte == "col":
            ints = " | ".join(", ".join(FRANJAS[j] for j in hijo.cols) for hijo in nodo.hijos)
        ax.set_title(f"bloque base ({g},{h}), corte de {dict(fil='filas', col='columnas', conj='filas y columnas')[nodo.corte]}\n"
                     f"g̃ = {nodo.g_corr:.4f}, p = {nodo.p_cortes[nodo.corte]:.3g}\nintervalos: {ints}",
                     fontsize=8)
        ax.text(0.5, -0.02, "\n".join(lineas), transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.suptitle(f"Bloques base refinados por AST-CC, {TITULO[periodo]} (δ = {DELTA}); "
                 f"gris = regiones que no pertenecen al bloque", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(FIGURAS / f"refinados_{periodo}.png", dpi=150)
    plt.close(fig)


def arbol_anotado(periodo, res, O, rho, tau, zonas, nombres):
    E_global = esperados(O)
    L = [f"Árbol de AST-CC, {TITULO[periodo]} (alfa = 0.05, delta = {DELTA}, nula de bootstrap de días)",
         "G2_s: estadístico del corte; g~: ganancia corregida por sesgo; p: p-valor min-P del corte;",
         "p_omn e I~: prueba ómnibus del nodo y su información mutua corregida; l: nivel de actividad", ""]

    def nombres_de(idx):
        return ", ".join(f"{nombres.get(int(zonas[i]), zonas[i])}" for i in idx)

    def escribir(nodo, sangria, etiqueta):
        base = (f"{sangria}{etiqueta}: {len(nodo.filas)} regiones x {len(nodo.cols)} intervalos, "
                f"N = {int(nodo.N):,}, alfa_B = {nodo.alfa:.4f}, l = {actividad(nodo, E_global):.2f}")
        omn = "" if np.isnan(nodo.p_omn) else f", p_omn = {nodo.p_omn:.3g}, I~ = {nodo.I_corr:.4f}"
        if nodo.es_hoja:
            L.append(base + f"  -> hoja {nodo.estado}{omn}")
            return
        L.append(base + f"  -> corte de {nodo.corte}: G2_s = {nodo.G2_corte:.1f}, g~ = {nodo.g_corr:.4f}, "
                        f"p = {nodo.p_cortes[nodo.corte]:.3g}{omn}")
        for k, hijo in enumerate(nodo.hijos):
            escribir(hijo, sangria + "    ", f"hijo {k + 1}")
            if hijo.es_hoja or True:
                if nodo.corte in ("fil", "conj"):
                    L.append(f"{sangria}        regiones: {nombres_de(hijo.filas)}")
                if nodo.corte in ("col", "conj"):
                    L.append(f"{sangria}        intervalos: {', '.join(FRANJAS[j] for j in hijo.cols)}")

    for raiz in res.raices:
        g, h = rho[raiz.filas[0]], tau[raiz.cols[0]]
        escribir(raiz, "", f"Bloque ({g},{h})")
    (SALIDA / f"arbol_{periodo}.txt").write_text("\n".join(L), encoding="utf-8")


def main():
    zonas = pd.read_csv(DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    shp = gpd.read_file(CRUDOS / "taxi_zones/taxi_zones.shp")
    zonas_gdf = shp[shp["LocationID"].isin(zonas)][["LocationID", "geometry"]].reset_index(drop=True)
    for periodo in ("weekday", "holiday"):
        X = np.load(DATOS / f"Calibracion/conteos_diarios_{periodo}.npy")
        O = X.sum(0)
        rho = np.load(DATOS / f"AST-CC/{periodo}_C5_rho.npy")
        tau = np.load(DATOS / f"AST-CC/{periodo}_C5_tau.npy")
        res = ASTCC(Parametros(delta=DELTA), np.random.default_rng(SEMILLA)).ajustar(X, rho, tau)
        previas = pd.read_csv(SALIDA / f"hojas_{periodo}_C5.csv")
        coincide = len(previas) == len(res.hojas()) and np.array_equal(previas["N"], [int(h.N) for h in res.hojas()])
        # El soporte de 06_estabilidad.py se calculó con otra semilla: el árbol es el mismo
        # pero los dos hijos de un corte pueden salir en otro orden, así que las hojas se
        # emparejan por su contenido y no por su posición.
        sop_csv = SALIDA / f"soporte_{periodo}.csv"
        soporte = None
        if sop_csv.exists():
            s = pd.read_csv(sop_csv)
            clave = lambda prof, nr, ni, N: (int(prof), int(nr), int(ni), int(N))  # noqa: E731
            por_clave = {clave(f.profundidad, f.n_regiones, f.n_intervalos, f.N): f.soporte
                         for f in s.itertuples()}
            claves = [clave(h.profundidad, len(h.filas), len(h.cols), h.N) for h in res.hojas()]
            if len(por_clave) == len(s) and all(k in por_clave for k in claves):
                soporte = {id(h): por_clave[k] for h, k in zip(res.hojas(), claves)}
        print(f"{periodo}: {len(res.hojas())} hojas, {len(res.internos())} cortes; "
              f"coincide con hojas_{periodo}_C5.csv: {coincide}; soporte: {'sí' if soporte is not None else 'no'}")
        for modo in ("datos", "actividad"):
            figura_teselada(periodo, res, O, X.shape[0], rho, tau, soporte, modo)
        pr, pc = figura_profundidad(periodo, res, zonas_gdf, zonas)
        figura_refinados(periodo, res, O, rho, tau, zonas_gdf, zonas)
        arbol_anotado(periodo, res, O, rho, tau, zonas, nombres)
        print(f"  profundidad máxima: regiones {pr.max()}, intervalos {pc.max()}")
    print(f"Figuras en {FIGURAS}")


if __name__ == "__main__":
    main()
