"""
Figuras estáticas de la segunda etapa (para la tesis)
=====================================================
Lee las salidas de AST-CC/14 y dibuja, en PNG y en PDF vectorial:

  eventos_fig_tiempo     línea de tiempo de los eventos de 2017
  eventos_fig_fichas     cuatro eventos verificados: huella en el mapa y perfil horario
  eventos_fig_corte      corte del cubo (días x intervalos) de una zona alrededor de un evento
  eventos_fig_umbral     eventos de ruido esperados y eventos observados según el pico exigido
  eventos_fig_amplitud   amplitud de cada día y umbral de los días de calendario
  eventos_fig_cubo       el cubo días x intervalos x regiones con las huellas de los eventos

Entradas: Resultados/AST-CC/eventos_agrupados{.csv,_huellas.json,_calibracion.csv,_dias.csv},
          DATA/procesados/AST-CC/conteos_diarios_2017.npy, DATA/crudos/taxi_zones/taxi_zones.shp
Salidas:  Resultados/AST-CC/figuras/eventos_fig_*.png y .pdf
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc import eventos as ev  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS, SALIDA, CRUDOS = RAIZ / "DATA/procesados", RAIZ / "Resultados/AST-CC", RAIZ / "DATA/crudos"
FIGURAS = SALIDA / "figuras"
ANCHO = 7.2                                  # pulgadas: ancho de texto de la tesis
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
C_POS, C_NEG, C_CAL = "#c2410c", "#1d4ed8", "0.6"
Z_TOPE = 20                                  # |z| que satura la escala de color
MAPA = plt.get_cmap("RdBu_r")

# Eventos de las fichas: (fecha de inicio, zona del pico, nombre, zona del perfil o None = la del pico)
FICHAS = [("2017-11-05", "Financial District South", "Maratón de Nueva York", None),
          ("2017-09-18", "Midtown Center", "Asamblea General de la ONU", None),
          ("2017-05-01", "Randalls Island", "Feria Frieze New York", None),
          ("2017-04-12", "West Chelsea/Hudson Yards", "New York Auto Show (días de prensa)", None)]
# Cortes del cubo: (zona, primer día, último día, título)
CORTES = [("Randalls Island", "2017-04-17", "2017-05-21", "Randalls Island: feria Frieze New York"),
          ("Midtown Center", "2017-09-04", "2017-10-08", "Midtown Center: Asamblea General de la ONU")]
CUBO_ROTULOS = [("2017-11-05", "Financial District South", "Maratón"), ("2017-09-18", "Midtown Center", "ONU"),
                ("2017-05-01", "Randalls Island", "Frieze"), ("2017-03-01", "Clinton West", "Armory Show")]
# Rótulos de la línea de tiempo: (fecha de inicio, zona del pico, texto, desplazamiento vertical en puntos)
ROTULOS = [("2017-11-05", "Financial District South", "Maratón", 5),
           ("2017-05-01", "Randalls Island", "Frieze", 5),
           ("2017-10-08", "Financial District South", "Media maratón de Staten Island", 5),
           ("2017-09-01", "Randalls Island", "Electric Zoo", 5),
           ("2017-03-19", "Seaport", "Media maratón", 5),
           ("2017-07-27", "Randalls Island", "Panorama", 5),
           ("2017-06-02", "Randalls Island", "Governors Ball", 5),
           ("2017-07-16", "Lincoln Square West", "Triatlón", -10),
           ("2017-05-15", "Morningside Heights", "Graduación de Columbia", 5),
           ("2017-01-21", "Sutton Place/Turtle Bay North", "Marcha de las Mujeres", 5),
           ("2017-04-12", "West Chelsea/Hudson Yards", "Auto Show", -10),
           ("2017-03-01", "Clinton West", "Armory Show", -10),
           ("2017-09-18", "Midtown Center", "Asamblea de la ONU", 7)]
CUBO_ZOOM = ("2017-09-18", "Midtown Center", "2017-09-15", "2017-09-25")

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 100})


def guardar(fig, nombre):
    for ext in ("png", "pdf"):
        fig.savefig(FIGURAS / f"{nombre}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {nombre}", flush=True)


def fecha_corta(f):
    f = pd.Timestamp(f)
    return f"{f.day} {MESES[f.month - 1]}"


def rango(r):
    return fecha_corta(r.inicio) if r.n_dias == 1 else f"{fecha_corta(r.inicio)} a {fecha_corta(r.fin)}"


def color_z(z):
    return MAPA(0.5 + 0.5 * np.sign(z) * np.sqrt(min(abs(z), Z_TOPE) / Z_TOPE))


def buscar(t, inicio, region):
    r = t[(t.clase == "agudo") & (t.inicio == inicio) & (t.region_pico == region)]
    assert len(r) == 1, (inicio, region)
    return r.iloc[0]


# ----------------------------------------------------------------------------
# 1. Línea de tiempo
# ----------------------------------------------------------------------------

def fig_tiempo(t, dias, t_s):
    fig, ax = plt.subplots(figsize=(ANCHO, 3.1))
    for f in dias.fecha[dias.calendario == 1]:
        ax.axvspan(pd.Timestamp(f) - pd.Timedelta(hours=12), pd.Timestamp(f) + pd.Timedelta(hours=12), color="0.86", lw=0)
    ag, rep = t[t.clase == "agudo"], t[t.clase == "repartido"]
    centro = pd.to_datetime(ag.inicio) + (pd.to_datetime(ag.fin) - pd.to_datetime(ag.inicio)) / 2
    ax.scatter(centro, ag.puntuacion, s=6 + 5 * np.sqrt(ag.n_celdas), c=[C_POS if s == "+" else C_NEG for s in ag.signo],
               alpha=0.75, lw=0, zorder=3)
    ax.scatter(pd.to_datetime(rep.inicio), rep.puntuacion, s=14, marker="s", facecolor="none", edgecolor="0.25", lw=0.8, zorder=3)
    for inicio, region, texto, dy in ROTULOS:
        r = buscar(t, inicio, region)
        x = pd.Timestamp(r.inicio) + (pd.Timestamp(r.fin) - pd.Timestamp(r.inicio)) / 2
        ax.annotate(texto, (x, r.puntuacion), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=6, zorder=4)
    ax.axhline(t_s, color="0.3", lw=0.7, ls=":")
    ax.text(pd.Timestamp("2017-12-31"), t_s, f" umbral {t_s:g}", va="center", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(3, 220)
    ax.set_yticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    ax.set_xlim(pd.Timestamp("2016-12-28"), pd.Timestamp("2018-01-04"))
    ax.set_xticks([pd.Timestamp(2017, m, 1) for m in range(1, 13)], MESES)
    ax.set_ylabel("Puntuación del evento")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_POS, label="agudo, más llegadas"),
                       Line2D([], [], marker="o", ls="", color=C_NEG, label="agudo, menos llegadas"),
                       Line2D([], [], marker="s", ls="", mfc="none", color="0.25", label="repartido (z de G)"),
                       Rectangle((0, 0), 1, 1, color="0.86", label="día de calendario")],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    guardar(fig, "eventos_fig_tiempo")


# ----------------------------------------------------------------------------
# 2. Fichas de eventos
# ----------------------------------------------------------------------------

def fig_fichas(t, huella, geo, nom, Y, Z, E, clave):
    fig = plt.figure(figsize=(ANCHO, 8.8))
    exterior = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    for celda, (inicio, region, titulo, _) in zip(exterior, FICHAS):
        r = buscar(t, inicio, region)
        c = huella[("agudo", r.puesto)]
        z = Z[c[:, 0], c[:, 1], c[:, 2]]
        por_region = pd.Series(z).groupby(c[:, 1]).agg(lambda v: v.iloc[np.abs(v.to_numpy()).argmax()])
        i = nom.index(region)
        por_dia = pd.Series(np.abs(z[c[:, 1] == i])).groupby(c[c[:, 1] == i, 0]).max()
        dias = sorted(por_dia.sort_values(ascending=False).index[:3])
        interior = celda.subgridspec(len(dias), 2, width_ratios=[1, 1.45], hspace=0.55, wspace=0.22)
        marco = fig.add_subplot(celda)
        marco.set_axis_off()
        marco.set_title(f"{titulo}\n{rango(r)} · {r.n_celdas} celdas en {r.n_regiones} zona{'s' if r.n_regiones > 1 else ''}", loc="left",
                        fontsize=8.5, fontweight="bold", pad=16)

        ax = fig.add_subplot(interior[:, 0])
        geo.plot(ax=ax, color="0.9", edgecolor="white", lw=0.3)
        geo.iloc[por_region.index].plot(ax=ax, color=[color_z(v) for v in por_region], edgecolor="0.25", lw=0.4)
        ax.set_axis_off()

        for k, d in enumerate(dias):
            ax = fig.add_subplot(interior[k, 1])
            en_huella = sorted(c[(c[:, 0] == d) & (c[:, 1] == i), 2])
            ax.plot(E[d, i], color="0.45", ls="--", lw=1.1)
            ax.plot(Y[d, i], color="black", lw=1.3)
            ax.plot(en_huella, Y[d, i, en_huella], ls="", marker="o", ms=4, color=C_POS if r.signo == "+" else C_NEG, zorder=3)
            ax.set_xticks(range(0, 12, 3), [f"{2 * j:02d}" for j in range(0, 12, 3)])
            ax.set_ylim(bottom=0)
            ax.set_title(f"{region if k == 0 else ''}{' · ' if k == 0 else ''}{fecha_corta(clave[d])}", fontsize=7, loc="left", pad=2)
            ax.tick_params(length=2)
            if k == len(dias) - 1:
                ax.set_xlabel("Hora de inicio del intervalo")
    fig.legend(handles=[Line2D([], [], color="black", lw=1.3, label="llegadas del día"),
                        Line2D([], [], color="0.45", ls="--", lw=1.1, label="esperadas"),
                        Line2D([], [], marker="o", ls="", color="0.3", label="intervalos del evento"),
                        Rectangle((0, 0), 1, 1, color=color_z(15), label="zona con más llegadas"),
                        Rectangle((0, 0), 1, 1, color=color_z(-15), label="zona con menos llegadas")],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.005), frameon=False)
    guardar(fig, "eventos_fig_fichas")


# ----------------------------------------------------------------------------
# 3. Corte del cubo
# ----------------------------------------------------------------------------

def fig_corte(t, huella, nom, Z, clave):
    fig, ejes = plt.subplots(len(CORTES), 1, figsize=(ANCHO, 2.25 * len(CORTES)), constrained_layout=True)
    agudas = np.concatenate([huella[("agudo", p)] for p in t.puesto[t.clase == "agudo"]])
    for ax, (region, ini, fin, titulo) in zip(ejes, CORTES):
        i, d0, d1 = nom.index(region), clave.index(ini), clave.index(fin)
        im = ax.imshow(Z[d0:d1 + 1, i].T, origin="lower", aspect="auto", cmap=MAPA, vmin=-Z_TOPE, vmax=Z_TOPE,
                       extent=(-0.5, d1 - d0 + 0.5, -0.5, 11.5), interpolation="nearest")
        for d, _, j in agudas[(agudas[:, 1] == i) & (agudas[:, 0] >= d0) & (agudas[:, 0] <= d1)]:
            ax.add_patch(Rectangle((d - d0 - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="black", lw=0.9))
        lunes = [k for k in range(d1 - d0 + 1) if pd.Timestamp(clave[d0 + k]).dayofweek == 0]
        ax.set_xticks(lunes, [fecha_corta(clave[d0 + k]) for k in lunes])
        ax.set_yticks(range(0, 12, 2), [f"{2 * j:02d}" for j in range(0, 12, 2)])
        ax.set_ylabel("Hora de inicio")
        ax.set_title(titulo, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
    ejes[-1].set_xlabel("Día (las marcas son lunes)")
    fig.colorbar(im, ax=ejes, fraction=0.025, pad=0.015, label="z de la celda (recortado a ±20); recuadro = celda de un evento")
    guardar(fig, "eventos_fig_corte")


# ----------------------------------------------------------------------------
# 4. Calibración del umbral
# ----------------------------------------------------------------------------

def fig_umbral(cal, t_s):
    cal = cal[cal.pico <= 25]
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.plot(cal.pico, cal.observados, color="black", lw=1.4, label="eventos observados")
    ax.plot(cal.pico, cal["nulos_por_año"].where(cal["nulos_por_año"] > 0), color="0.45", ls="--", lw=1.2,
            label="eventos de ruido esperados (nula simulada)")
    ax.axvline(t_s, color=C_POS, lw=1)
    n = int(cal.observados[np.isclose(cal.pico, t_s)].iloc[0])
    ax.text(t_s + 0.3, 130, f"umbral {t_s:g}: {n} eventos,\nruido estimado ≤ 5 %", fontsize=7, color=C_POS, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.05, 3000)
    ax.set_yticks([0.1, 1, 10, 100, 1000], ["0,1", "1", "10", "100", "1000"])
    ax.set_xlabel("Pico |z| exigido al evento")
    ax.set_ylabel("Eventos en el año")
    ax.legend(frameon=False, loc="upper right")
    guardar(fig, "eventos_fig_umbral")


# ----------------------------------------------------------------------------
# 5. Amplitud por día
# ----------------------------------------------------------------------------

def fig_amplitud(dias, umbral):
    fig, ax = plt.subplots(figsize=(ANCHO, 2.7))
    x = pd.to_datetime(dias.fecha)
    cal = dias.calendario == 1
    ax.vlines(x[~cal], 0, dias.amplitud[~cal], color="0.6", lw=0.9)
    ax.vlines(x[cal], 0, dias.amplitud[cal], color="black", lw=1.2)
    ax.axhline(umbral, color=C_POS, lw=0.9, ls="--")
    ax.text(pd.Timestamp("2018-01-02"), umbral, f" {umbral:g}", va="center", fontsize=7, color=C_POS)
    idx = np.flatnonzero(cal.to_numpy())
    rachas = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    arriba, ultimo = True, -99
    for r in rachas:
        if r[0] - ultimo < 12:
            continue
        ultimo = r[-1]
        k = r[dias.amplitud.to_numpy()[r].argmax()]
        rot = fecha_corta(dias.fecha[r[0]]) + (f"–{pd.Timestamp(dias.fecha[r[-1]]).day}" if len(r) > 1 and
                                                 pd.Timestamp(dias.fecha[r[0]]).month == pd.Timestamp(dias.fecha[r[-1]]).month
                                                 else (f" – {fecha_corta(dias.fecha[r[-1]])}" if len(r) > 1 else ""))
        ax.annotate(rot, (x[k], dias.amplitud[k]), xytext=(0, 3 if arriba else 11), textcoords="offset points",
                    ha="center", fontsize=6)
        arriba = not arriba
    ax.set_xlim(pd.Timestamp("2016-12-28"), pd.Timestamp("2018-01-06"))
    ax.set_xticks([pd.Timestamp(2017, m, 1) for m in range(1, 13)], MESES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Amplitud: fracción de hojas\ncon |z| ≥ 3")
    ax.legend(handles=[Line2D([], [], color="black", lw=1.2, label="día de calendario"),
                       Line2D([], [], color="0.6", lw=0.9, label="día normal")], frameon=False, loc="upper center", ncol=2)
    guardar(fig, "eventos_fig_amplitud")


# ----------------------------------------------------------------------------
# 6. El cubo
# ----------------------------------------------------------------------------

def ejes_cubo(ax, n_dias, n_reg, etiquetas_dia):
    ax.set_xlim(0, n_dias)
    ax.set_ylim(0, 12)
    ax.set_zlim(0, n_reg)
    ax.set_yticks(range(0, 13, 3), [f"{2 * j:02d}" for j in range(0, 13, 3)])
    ax.set_xticks(*etiquetas_dia)
    ax.set_zticks([])
    ax.set_xlabel("Día", labelpad=4)
    ax.set_ylabel("Hora", labelpad=-6)
    ax.set_zlabel("Regiones (sur → norte)", labelpad=-12)
    ax.tick_params(pad=-2)
    for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
        eje.pane.set_facecolor("white")
        eje.pane.set_edgecolor("0.7")
    ax.grid(False)


def fig_cubo(t, huella, geo, nom, Z, clave):
    orden = np.argsort(geo.centroid.y.to_numpy())                   # regiones de sur a norte
    rango_de = np.empty(len(nom), int)
    rango_de[orden] = np.arange(len(nom))
    fig = plt.figure(figsize=(ANCHO, 4.3))

    # (a) todo el año: las celdas de los eventos agudos dentro del cubo
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for r in t[t.clase == "agudo"].itertuples():
        c = huella[("agudo", r.puesto)]
        ax.scatter(c[:, 0] + 0.5, c[:, 2] + 0.5, rango_de[c[:, 1]] + 0.5, s=5, lw=0, alpha=0.8,
                   color=C_POS if r.signo == "+" else C_NEG, depthshade=False)
    for inicio, region, rotulo in CUBO_ROTULOS:
        r = buscar(t, inicio, region)
        c = huella[("agudo", r.puesto)]
        ax.text(c[:, 0].mean(), c[:, 2].max() + 1.2, rango_de[c[:, 1]].max() + 3, rotulo, fontsize=6.5, ha="center")
    ejes_cubo(ax, 365, len(nom), ([pd.Timestamp(2017, m, 1).dayofyear - 1 for m in (1, 4, 7, 10)], ["ene", "abr", "jul", "oct"]))
    ax.set_title("(a) Los 365 días: celdas de los eventos agudos", fontsize=8)
    ax.view_init(22, -62)
    ax.set_box_aspect((2.2, 1, 1.1))

    # (b) acercamiento: un evento como bloque de celdas
    inicio, region, ini, fin = CUBO_ZOOM
    r = buscar(t, inicio, region)
    c = huella[("agudo", r.puesto)]
    d0, d1 = clave.index(ini), clave.index(fin)
    regiones = np.unique(rango_de[c[:, 1]])
    r0, r1 = max(regiones.min() - 2, 0), min(regiones.max() + 3, len(nom))
    lleno = np.zeros((d1 - d0 + 1, 12, r1 - r0), bool)
    colores = np.zeros(lleno.shape + (4,))
    for d, i, j in c:
        if d0 <= d <= d1:
            lleno[d - d0, j, rango_de[i] - r0] = True
            colores[d - d0, j, rango_de[i] - r0] = to_rgba(color_z(Z[d, i, j]), 0.9)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.voxels(lleno, facecolors=colores, edgecolor=(0, 0, 0, 0.35), linewidth=0.3)
    marcas = [k for k in range(d1 - d0 + 1) if k % 2 == 1]
    ejes_cubo(ax, d1 - d0 + 1, r1 - r0, ([k + 0.5 for k in marcas], [fecha_corta(clave[d0 + k]) for k in marcas]))
    ax.tick_params(axis="x", labelsize=6)
    ax.set_title(f"(b) Asamblea General de la ONU: {r.n_celdas} celdas con menos llegadas,\n{rango(r)}, "
                 f"{r1 - r0} regiones alrededor de Midtown", fontsize=8)
    ax.view_init(22, -62)
    ax.set_box_aspect((1.6, 1, 1.1))
    fig.subplots_adjust(left=0, right=0.95, bottom=0.04, top=0.93, wspace=0.08)
    guardar(fig, "eventos_fig_cubo")


def main():
    zonas = pd.read_csv(DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    nom = [nombres[z] for z in zonas]
    shp = gpd.read_file(CRUDOS / "taxi_zones/taxi_zones.shp")
    geo = shp[shp["LocationID"].isin(zonas)].dissolve("LocationID").geometry.loc[list(zonas)]
    Y = np.load(DATOS / "AST-CC/conteos_diarios_2017.npy")
    fechas = pd.date_range("2017-01-01", "2017-12-31")
    clave = list(fechas.strftime("%Y-%m-%d"))
    Z, E = ev.puntuar_celdas(Y, fechas), ev.esperados(Y, fechas)
    t = pd.read_csv(SALIDA / "eventos_agrupados.csv").fillna("")
    h = json.loads((SALIDA / "eventos_agrupados_huellas.json").read_text(encoding="utf-8"))
    huella = {(e["clase"], e["puesto"]): np.array(e["celdas"]).reshape(-1, 3) for e in h["eventos"]}
    dias = pd.read_csv(SALIDA / "eventos_agrupados_dias.csv")
    cal = pd.read_csv(SALIDA / "eventos_agrupados_calibracion.csv")

    print("Figuras:")
    fig_tiempo(t, dias, h["t_s"])
    fig_fichas(t, huella, geo, nom, Y, Z, E, clave)
    fig_corte(t, huella, nom, Z, clave)
    fig_umbral(cal, h["t_s"])
    fig_amplitud(dias, 0.25)
    fig_cubo(t, huella, geo, nom, Z, clave)


if __name__ == "__main__":
    main()
