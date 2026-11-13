"""
Piloto de la segunda etapa: ¿qué días de 2017 se desvían del patrón recurrente?
===============================================================================
AST-CC describe el patrón que se repite entre días. Un evento de un solo día no puede
aparecer como corte, pero sí como desviación de ese día respecto de los demás. Aquí se
puntúa cada día de 2017 en cada unidad espacio-temporal y se compara qué unidades sirven
mejor de línea base:

  celda    cada región x intervalo (66 x 12 = 792 unidades)
  stcc     bloques de la rejilla de ST-CC (cgc)
  base     bloques de la partición base de AST-CC (BBAC C5)
  hojas    hojas del árbol de AST-CC (base C5, delta = 0.005)

Puntuación de la unidad u el día d: cuota s = N(d, u) / N(d) de las llegadas del día que
caen en la unidad, estandarizada de forma robusta con los demás días del mismo día de la
semana: z = (s - mediana) / max(1.4826 MAD, error típico de Poisson). Dividir entre N(d)
quita el nivel del día (una nevada baja toda la ciudad); queda el cambio de reparto.

Evaluación con fechas reales de EVENTOS (sin datos inyectados): puesto que ocupa el día
entre los de su periodo según max_u |z|, y unidad que da ese máximo.

Entradas: DATA/procesados/AST-CC/conteos_diarios_2017.npy, particiones de ST-CC y AST-CC,
          Resultados/AST-CC/hojas_{periodo}_C5.csv
Salidas:  Resultados/AST-CC/eventos_cubo.txt, eventos_cubo_{periodo}.csv (z de hojas x días)
          Resultados/AST-CC/figuras/eventos_cubo_{periodo}.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

RAIZ = Path(__file__).resolve().parents[1]
DATOS, SALIDA, CRUDOS = RAIZ / "DATA/procesados", RAIZ / "Resultados/AST-CC", RAIZ / "DATA/crudos"
FRANJAS = [f"{h:02d}-{h + 2:02d}" for h in range(0, 24, 2)]
DIAS_SEMANA = {"weekday": [1, 2, 3], "holiday": [5, 6]}          # lunes = 0
VENTANA = 91                                                       # días a cada lado para la línea base
Z_AMP = 3                                                          # |z| de unidad que cuenta para la amplitud
AMP_DIFUSO = 0.25                                                  # amplitud de hojas desde la que el día es difuso
Z_DET = 8                                                          # |z| de celda para listar una detección
Z_MAX = 8                                                          # tope de color de la figura

# Fechas con un hecho conocido. 'calendario' = cambia toda la ciudad; 'local' = zona concreta.
EVENTOS = {
    "2017-01-21": ("local", "Marcha de las Mujeres (Dag Hammarskjold Plaza, 2.ª Avenida)"),
    "2017-02-09": ("calendario", "Nevada"),
    "2017-03-14": ("calendario", "Nevada (Stella)"),
    "2017-03-19": ("local", "Media maratón de NYC"),
    "2017-05-07": ("local", "Five Boro Bike Tour"),
    "2017-05-18": ("local", "Atropello en Times Square"),
    "2017-06-11": ("local", "Desfile puertorriqueño (5.ª Avenida)"),
    "2017-06-25": ("local", "Marcha del Orgullo (5.ª Avenida, Village)"),
    "2017-07-04": ("calendario", "4 de julio"),
    "2017-09-19": ("local", "Asamblea General de la ONU"),
    "2017-09-20": ("local", "Asamblea General de la ONU"),
    "2017-09-21": ("local", "Asamblea General de la ONU"),
    "2017-10-31": ("local", "Halloween y atentado del West Side Highway"),
    "2017-11-05": ("local", "Maratón de NYC"),
    "2017-11-23": ("calendario", "Acción de Gracias (desfile de Macy's)"),
    "2017-11-29": ("local", "Encendido del árbol de Rockefeller Center"),
    "2017-12-31": ("calendario", "Nochevieja (Times Square)"),
}

# Días que señaló el propio análisis y cuyo evento se identificó después (no cuentan como
# validación a priori; solo anotan la lista de detecciones).
HALLADOS = {
    "2017-04-23": "9/11 Memorial 5K (salida 08:00, Pier 26, Tribeca)",
    "2017-05-04": "Frieze New York, preinauguración (Randalls Island)",
    "2017-05-06": "Frieze New York (Randalls Island)",
    "2017-05-17": "Graduación de Columbia University",
    "2017-06-03": "Governors Ball (Randalls Island)",
    "2017-06-04": "Governors Ball (Randalls Island)",
    "2017-07-16": "NYC Triathlon (salida 05:50, Riverside Park)",
    "2017-07-29": "Panorama Festival (Randalls Island)",
    "2017-07-30": "Panorama Festival (Randalls Island)",
    "2017-09-02": "Electric Zoo (Randalls Island)",
    "2017-09-03": "Electric Zoo (Randalls Island)",
}


def unidades(periodo, zonas):
    """Para cada línea base, lista de (filas, cols) que cubren la matriz 66 x 12."""
    def rejilla(rho, tau):
        return [(np.flatnonzero(rho == g), np.flatnonzero(tau == h)) for g in np.unique(rho) for h in np.unique(tau)]

    fila = {z: i for i, z in enumerate(zonas)}
    hojas = pd.read_csv(SALIDA / f"hojas_{periodo}_C5.csv")
    U = {
        "celda": [(np.array([i]), np.array([j])) for i in range(len(zonas)) for j in range(12)],
        "stcc": rejilla(np.load(DATOS / f"ST-CC/{periodo}_row_clusters.npy"),
                        np.load(DATOS / f"ST-CC/{periodo}_col_clusters.npy")),
        "base": rejilla(np.load(DATOS / f"AST-CC/{periodo}_C5_rho.npy"),
                        np.load(DATOS / f"AST-CC/{periodo}_C5_tau.npy")),
        "hojas": [(np.array([fila[int(z)] for z in r.split()]), np.array([FRANJAS.index(f) for f in c.split()]))
                  for r, c in zip(hojas["regiones"], hojas["intervalos"])],
    }
    for nombre, us in U.items():                                   # cada partición cubre todo una vez
        cubre = np.zeros((len(zonas), 12), int)
        for f, c in us:
            cubre[np.ix_(f, c)] += 1
        assert (cubre == 1).all(), nombre
    return U


def z_robusto(S, N, fechas):
    """z robusto de las cuotas S (días x unidades, con denominadores N del mismo tamaño)
    frente a los demás días del mismo día de la semana a menos de VENTANA días."""
    Z = np.zeros_like(S)
    dsem, dia = fechas.dayofweek.to_numpy(), fechas.dayofyear.to_numpy()
    for w in np.unique(dsem):
        k = np.flatnonzero(dsem == w)
        for a in k:                                                # dejar el día fuera
            resto = k[(k != a) & (np.abs(dia[k] - dia[a]) <= VENTANA)]
            med = np.median(S[resto], 0)
            mad = 1.4826 * np.median(np.abs(S[resto] - med), 0)
            poisson = np.sqrt(np.maximum(med, 1 / N[a]) / N[a])    # suelo para unidades escasas
            Z[a] = (S[a] - med) / np.maximum(mad, poisson)
    return Z


def puntuar(Y, fechas, us):
    """Nivel 1: z de la cuota de cada unidad en el total del día. Devuelve días x unidades."""
    N = Y.sum((1, 2)).astype(float)
    C = np.stack([Y[:, f][:, :, c].sum((1, 2)) for f, c in us], 1)
    return z_robusto(C / N[:, None], np.broadcast_to(N[:, None], C.shape), fechas)


def residuo(Y, fechas, us):
    """Nivel 2: z de la cuota de cada celda dentro de su unidad, N(d, celda) / N(d, unidad).
    Si la unidad entera sube o baja ese día, la cuota no cambia. Devuelve días x 792."""
    Nu = np.ones(Y.shape)
    for f, c in us:
        Nu[np.ix_(range(len(Y)), f, c)] = np.maximum(Y[:, f][:, :, c].sum((1, 2)), 1)[:, None, None]
    D = len(Y)
    return z_robusto((Y / Nu).reshape(D, -1), Nu.reshape(D, -1), fechas)


def describir(u, zonas, nombres, k=3):
    f, c = u
    reg = ", ".join(nombres[zonas[i]] for i in f[:k]) + (f" y {len(f) - k} más" if len(f) > k else "")
    return f"{reg} [{' '.join(FRANJAS[j] for j in c)}]"


def figura(periodo, Z, fechas, us):
    orden = np.argsort([c.min() * 100 + len(f) for f, c in us])     # hojas por franja horaria
    fig, ax = plt.subplots(figsize=(15, 0.17 * len(us) + 2.2))
    im = ax.imshow(np.clip(Z[:, orden].T, -Z_MAX, Z_MAX), aspect="auto", cmap="RdBu_r", vmin=-Z_MAX, vmax=Z_MAX,
                   interpolation="nearest")
    meses = [i for i in range(len(fechas)) if i == 0 or fechas[i].month != fechas[i - 1].month]
    ax.set_xticks(meses, [fechas[i].strftime("%b") for i in meses])
    ax.set_yticks(range(len(us)), [f"{' '.join(FRANJAS[j][:2] for j in us[o][1])} h · {len(us[o][0])} reg."
                                   for o in orden], fontsize=6)
    for i, d in enumerate(fechas):
        if d.strftime("%Y-%m-%d") in EVENTOS:
            ax.annotate("▼", (i, -0.5), ha="center", va="bottom", fontsize=7, annotation_clip=False)
    ax.set_xlabel(f"Días de 2017 ({'martes a jueves' if periodo == 'weekday' else 'sábados y domingos'}); ▼ = fecha de EVENTOS")
    ax.set_ylabel("Hojas de AST-CC")
    fig.colorbar(im, ax=ax, pad=0.01, label="z de la cuota de la hoja (recortado)")
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / f"eventos_cubo_{periodo}.png", dpi=170)
    plt.close(fig)


def fichas(periodo, Y, fechas, casos, zonas, nombres):
    """Perfil horario de la región el día del evento frente a los días de su línea base."""
    fig, ejes = plt.subplots(2, 3, figsize=(13, 6.2))
    dsem, dia = fechas.dayofweek.to_numpy(), fechas.dayofyear.to_numpy()
    for ax, (d, i, z) in zip(ejes.ravel(), casos):
        k = np.flatnonzero((dsem == dsem[d]) & (np.abs(dia - dia[d]) <= VENTANA) & (np.arange(len(dia)) != d))
        base = Y[k][:, i, :]
        ax.fill_between(range(12), np.percentile(base, 10, 0), np.percentile(base, 90, 0), color="0.8",
                        label="días de la línea base (P10-P90)")
        ax.plot(np.median(base, 0), color="0.35", lw=1.2, label="mediana")
        ax.plot(Y[d, i, :], color="C3", lw=1.8, marker="o", ms=3.5, label="día del evento")
        ax.set_xticks(range(0, 12, 2), [FRANJAS[j][:2] for j in range(0, 12, 2)])
        ax.set_title(f"{nombres[zonas[i]]}\n{fechas[d].strftime('%Y-%m-%d')} (z = {z:+.0f})", fontsize=9)
        ax.set_ylim(bottom=0)
    for ax in ejes[:, 0]:
        ax.set_ylabel("Llegadas por intervalo")
    for ax in ejes[-1]:
        ax.set_xlabel("Hora de inicio del intervalo")
    ejes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(SALIDA / "figuras" / f"eventos_cubo_fichas_{periodo}.png", dpi=170)
    plt.close(fig)


def main():
    zonas = pd.read_csv(DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    Y_todo = np.load(DATOS / "AST-CC/conteos_diarios_2017.npy")
    todas = pd.date_range("2017-01-01", "2017-12-31")

    L = ["Piloto de la segunda etapa: desviación de cada día de 2017 respecto de los demás",
         "z = cuota de la unidad en el día, estandarizada con los otros días del mismo día de la semana",
         "puesto = posición del día entre los de su periodo según max |z| (1 = el más atípico)", ""]
    for periodo in ("weekday", "holiday"):
        sel = np.isin(todas.dayofweek, DIAS_SEMANA[periodo])
        fechas, Y = todas[sel], Y_todo[sel]
        U = unidades(periodo, zonas)
        Z = {b: puntuar(Y, fechas, us) for b, us in U.items()}
        maxz = {b: np.abs(z).max(1) for b, z in Z.items()}
        puesto = {b: pd.Series(-m).rank(method="min").astype(int).to_numpy() for b, m in maxz.items()}

        L += [f"== {periodo}: {len(fechas)} días; unidades: " + ", ".join(f"{b} {len(us)}" for b, us in U.items()), "",
              f"  {'fecha':<11}{'tipo':<11}" + "".join(f"{b:>8}" for b in U) + "   evento  ->  unidad con mayor |z| (celda | hojas)"]
        for f, (tipo, desc) in EVENTOS.items():
            if pd.Timestamp(f) not in fechas:
                continue
            d = fechas.get_loc(pd.Timestamp(f))
            L.append(f"  {f:<11}{tipo:<11}" + "".join(f"{puesto[b][d]:>8}" for b in U) + f"   {desc}")
            for b in ("celda", "hojas"):
                u = int(np.abs(Z[b][d]).argmax())
                L.append(f"  {'':<22}{b:>6}: z = {Z[b][d, u]:+7.1f}  {describir(U[b][u], zonas, nombres)}")
        L.append("")
        for b in ("celda", "hojas"):
            L.append(f"  Los 15 días más atípicos según {b}:")
            for d in np.argsort(-maxz[b])[:15]:
                u = int(np.abs(Z[b][d]).argmax())
                marca = EVENTOS.get(fechas[d].strftime("%Y-%m-%d"), ("", ""))[1]
                L.append(f"    {fechas[d].strftime('%Y-%m-%d %a')}  z = {Z[b][d, u]:+7.1f}  "
                         f"{describir(U[b][u], zonas, nombres)}{'   <- ' + marca if marca else ''}")
            L.append("")

        # Nivel 2: residuo de cada celda dentro de su unidad ('celda' = dentro de toda la ciudad)
        celdas = U["celda"]
        R = {"ciudad": Z["celda"], **{b: residuo(Y, fechas, U[b]) for b in ("stcc", "base", "hojas")}}
        maxr = {b: np.abs(r).max(1) for b, r in R.items()}
        puesto_r = {b: pd.Series(-m).rank(method="min").astype(int).to_numpy() for b, m in maxr.items()}
        L += ["  Nivel 2: residuo de la celda dentro de su unidad (puesto del día según max |z| de celda)",
              f"  {'fecha':<11}{'tipo':<11}" + "".join(f"{b:>8}" for b in R) + "   evento  ->  celda con mayor |z| dentro de su hoja"]
        for f, (tipo, desc) in EVENTOS.items():
            if pd.Timestamp(f) not in fechas:
                continue
            d = fechas.get_loc(pd.Timestamp(f))
            u = int(np.abs(R["hojas"][d]).argmax())
            L.append(f"  {f:<11}{tipo:<11}" + "".join(f"{puesto_r[b][d]:>8}" for b in R)
                     + f"   {desc}  ->  z = {R['hojas'][d, u]:+.1f} {describir(celdas[u], zonas, nombres)}")
        for tipo in ("local", "calendario"):
            ds = [fechas.get_loc(pd.Timestamp(f)) for f, (t, _) in EVENTOS.items() if t == tipo and pd.Timestamp(f) in fechas]
            L.append(f"  {'mediana ' + tipo:<22}" + "".join(f"{np.median(puesto_r[b][ds]):>8.0f}" for b in R))
        L += ["", "  Los 15 días más atípicos según el residuo dentro de las hojas:"]
        for d in np.argsort(-maxr["hojas"])[:15]:
            u = int(np.abs(R["hojas"][d]).argmax())
            marca = EVENTOS.get(fechas[d].strftime("%Y-%m-%d"), ("", ""))[1]
            L.append(f"    {fechas[d].strftime('%Y-%m-%d %a')}  z = {R['hojas'][d, u]:+7.1f}  "
                     f"{describir(celdas[u], zonas, nombres)}{'   <- ' + marca if marca else ''}")
        L.append("")

        # Amplitud: cuántas unidades se desvían a la vez (|z| >= Z_AMP). Un día de calendario mueve
        # muchas; un evento local, pocas.
        amp = {b: (np.abs(Z[b]) >= Z_AMP).mean(1) for b in ("stcc", "base", "hojas")}
        L += [f"  Amplitud: fracción de unidades con |z| >= {Z_AMP} en el día "
              f"(mediana de todos los días: " + ", ".join(f"{b} {np.median(a):.2f}" for b, a in amp.items())
              + "; percentil 90: " + ", ".join(f"{b} {np.percentile(a, 90):.2f}" for b, a in amp.items()) + ")",
              f"  {'fecha':<11}{'tipo':<11}" + "".join(f"{b:>8}" for b in amp) + "   evento"]
        for f, (tipo, desc) in EVENTOS.items():
            if pd.Timestamp(f) in fechas:
                d = fechas.get_loc(pd.Timestamp(f))
                L.append(f"  {f:<11}{tipo:<11}" + "".join(f"{amp[b][d]:>8.2f}" for b in amp) + f"   {desc}")
        L.append("")

        # Detecciones: la mayor |z| de celda de cada (día, región), con conteo observado y esperado
        Zc = Z["celda"].reshape(len(fechas), len(zonas), 12)
        N = Y.sum((1, 2))
        filas = []
        for d, i in zip(*np.nonzero(np.abs(Zc).max(2) >= Z_DET)):
            j = int(np.abs(Zc[d, i]).argmax())
            k = np.flatnonzero((fechas.dayofweek == fechas[d].dayofweek) & (np.arange(len(fechas)) != d)
                               & (np.abs(fechas.dayofyear - fechas[d].dayofyear) <= VENTANA))
            esperado = np.median(Y[k, i, j] / N[k]) * N[d]
            filas.append({"fecha": fechas[d].strftime("%Y-%m-%d"), "dia": fechas[d].strftime("%a"),
                          "region": nombres[zonas[i]], "intervalo": FRANJAS[j], "z": round(float(Zc[d, i, j]), 1),
                          "observado": int(Y[d, i, j]), "esperado": round(float(esperado), 1),
                          "amplitud_hojas": round(float(amp["hojas"][d]), 2),
                          "clase": "difuso" if amp["hojas"][d] >= AMP_DIFUSO else "local",
                          "evento_conocido": EVENTOS.get(fechas[d].strftime("%Y-%m-%d"), ("", ""))[1],
                          "evento_hallado": HALLADOS.get(fechas[d].strftime("%Y-%m-%d"), ""),
                          "_d": d, "_i": i})
        det = pd.DataFrame(filas).sort_values("z", key=np.abs, ascending=False)
        det.drop(columns=["_d", "_i"]).to_csv(SALIDA / f"eventos_cubo_detecciones_{periodo}.csv", index=False)
        loc = det[det.clase == "local"]
        L += [f"  Detecciones con |z| >= {Z_DET} (una por día y región): {len(det)} en {det.fecha.nunique()} días; "
              f"{len(loc)} locales en {loc.fecha.nunique()} días (amplitud de hojas < {AMP_DIFUSO})",
              "  Las 25 detecciones locales más fuertes:"]
        for r in loc.head(25).itertuples():
            L.append(f"    {r.fecha} {r.dia}  z = {r.z:+7.1f}  {r.region} [{r.intervalo}]  obs {r.observado} / esp {r.esperado:.0f}"
                     f"  amp {r.amplitud_hojas:.2f}{'   <- ' + r.evento_conocido if r.evento_conocido else ''}"
                     f"{'   (hallado: ' + r.evento_hallado + ')' if r.evento_hallado else ''}")
        n25 = loc.head(25)
        L += [f"  De las 25: {(n25.evento_conocido != '').sum()} en fechas de EVENTOS, {(n25.evento_hallado != '').sum()} con evento "
              f"identificado después, {((n25.evento_conocido == '') & (n25.evento_hallado == '')).sum()} sin explicar", ""]
        casos = loc.drop_duplicates("region").head(6)
        fichas(periodo, Y, fechas, [(int(a), int(b), z) for a, b, z in zip(casos["_d"], casos["_i"], casos["z"])],
               zonas, nombres)

        pd.DataFrame(Z["hojas"].T, columns=fechas.strftime("%Y-%m-%d"),
                     index=[describir(u, zonas, nombres, k=99) for u in U["hojas"]]).to_csv(
            SALIDA / f"eventos_cubo_{periodo}.csv", float_format="%.2f")
        figura(periodo, Z["hojas"], fechas, U["hojas"])

    texto = "\n".join(L)
    print(texto)
    (SALIDA / "eventos_cubo.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
