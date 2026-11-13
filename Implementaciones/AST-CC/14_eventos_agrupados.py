"""
Segunda etapa: de celdas anómalas a eventos en el cubo regiones x intervalos x días
===================================================================================
AST-CC/11 puntúa cada celda del cubo (día, región, intervalo). Aquí las celdas anómalas se
unen en eventos y se juntan las tres puntuaciones en una sola tabla.

1. Umbral con criterio. La parte central de los z se toma como nula empírica N(0, s0^2)
   (Efron, 2004). El umbral t es el menor valor con tasa de falsos descubrimientos estimada
   n P(|N(0, s0^2)| >= t) / #{|z| >= t} <= Q. Lo mismo, a una cola, para el z de la ganancia
   G de cada hoja (AST-CC/13).
2. Días de calendario. Si la fracción de hojas desviadas (amplitud) llega a AMP_DIFUSO, el
   día entero es un solo evento de clase «calendario» (festivo, nevada): cambia toda la ciudad.
3. Eventos agudos. En los demás días, las celdas con |z| >= t se unen si tienen el mismo
   signo y son vecinas en el cubo: regiones colindantes en el mismo día e intervalo,
   intervalos contiguos de la misma región, o la misma región e intervalo en días
   consecutivos. Cada componente conexa es un evento con su huella. Solo se conservan los
   que tienen un pico |z| >= t_s (t extiende la huella; t_s decide). t_s sale de una nula
   simulada: cubos sin eventos, con la media esperada de cada celda y su sobredispersión
   (binomial negativa, celdas independientes), pasados por la misma puntuación y la misma
   agrupación. Es el menor pico con (eventos nulos esperados) / (eventos observados) <=
   FDR_EVENTO. La nula no incluye choques comunes a varias celdas (lluvia, cortes de
   tráfico), así que la tasa real de ruido puede ser mayor que la estimada.
4. Eventos repartidos. Pares (día, hoja) con G sobre su umbral que no tocan ningún evento
   agudo: la hoja entera cambia de reparto sin que ninguna celda destaque.

Se usan los 365 días de 2017. Cada día se compara con los del mismo día de la semana, así
que lunes y viernes entran sin cambios; para la amplitud y para G, los días de lunes a
viernes usan las hojas del árbol de laborables y los sábados y domingos las del de festivos.
Con todos los días en un solo cubo, un evento puede pasar de un viernes a un sábado.

Entradas: las de AST-CC/11, DATA/crudos/taxi_zones/taxi_zones.shp
Salidas:  Resultados/AST-CC/eventos_agrupados.txt, eventos_agrupados.csv,
          eventos_agrupados_huellas.json (celdas de cada evento, para AST-CC/15 y 16),
          eventos_agrupados_calibracion.csv (eventos nulos y observados según el pico) y
          eventos_agrupados_dias.csv (amplitud de cada día)
"""

import importlib.util
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc import eventos as ev  # noqa: E402
sys.stdout.reconfigure(encoding="utf-8")


def _cargar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "AST-CC" / archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cubo = _cargar("cubo11", "11_eventos_cubo.py")

Q = 0.01                    # tasa de falsos descubrimientos admitida
FDR_EVENTO = 0.05           # fracción admitida de eventos agudos que serían ruido
R_NULA = 100                # cubos simulados para la nula de los eventos agudos
R_NULA_G = 50               # cubos simulados para la nula de los eventos repartidos
SEMILLA = 20261004
CORTES_NULA = (4, 5, 6, 7, 8, 10, 12, 15)
CORTES_G = (3, 3.5, 4, 4.5, 5, 6)
MARGEN = 50                 # pies: holgura para decidir que dos zonas colindan
FRANJAS = cubo.FRANJAS
CORTES = (6, 8, 10, 15, 20)


def colindancia(zonas):
    """Matriz booleana de zonas que se tocan (con MARGEN de holgura)."""
    shp = gpd.read_file(cubo.CRUDOS / "taxi_zones/taxi_zones.shp")
    g = shp[shp["LocationID"].isin(zonas)].dissolve("LocationID").geometry
    g = g.loc[list(zonas)]
    amplio = g.buffer(MARGEN)
    A = np.array([amplio.iloc[i].intersects(g.iloc[k]) for i in range(len(g)) for k in range(len(g))])
    A = A.reshape(len(g), len(g))
    np.fill_diagonal(A, False)
    return A | A.T


def tramos(js):
    """Intervalos contiguos como «06-10 h»."""
    js, partes, ini = sorted(set(js)), [], None
    for k, j in enumerate(js):
        ini = j if ini is None else ini
        if k + 1 == len(js) or js[k + 1] != j + 1:
            partes.append(f"{FRANJAS[ini][:2]}-{FRANJAS[j][3:]}")
            ini = None
    return " ".join(partes) + " h"


def main():
    zonas = pd.read_csv(cubo.DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(cubo.CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    nom = [nombres[z] for z in zonas]
    vecinas = colindancia(zonas)
    Y = np.load(cubo.DATOS / "AST-CC/conteos_diarios_2017.npy")
    fechas = pd.date_range("2017-01-01", "2017-12-31")
    D, clave = len(fechas), list(fechas.strftime("%Y-%m-%d"))
    conocidos = {**{f: v[1] for f, v in cubo.EVENTOS.items()}, **cubo.HALLADOS}
    grupo = np.where(fechas.dayofweek < 5, "weekday", "holiday")

    Zc = ev.puntuar_celdas(Y, fechas)
    E = ev.esperados(Y, fechas)
    amp, zg_dia, G = np.zeros(D), np.zeros(D), {}
    for g in ("weekday", "holiday"):
        dias = np.flatnonzero(grupo == g)
        hojas = cubo.unidades(g, zonas)["hojas"]
        amp[dias] = ev.amplitud(Y[dias], fechas[dias], hojas, cubo.Z_AMP)
        ZG, validas = ev.ganancia(Y[dias], fechas[dias], hojas)
        zg_dia[dias] = ZG.max(1)
        hoja_de = -np.ones((len(zonas), 12), int)
        for k, (f, c) in enumerate(hojas):
            hoja_de[np.ix_(f, c)] = k
        G[g] = {"dias": dias, "local": {d: k for k, d in enumerate(dias)}, "hojas": hojas, "ZG": ZG, "validas": validas,
                "hoja_de": hoja_de, "tocada": np.zeros((len(dias), len(hojas)), bool),
                "t": ev.umbral_fdr(ZG, Q, una_cola=True)[0]}
    t_z, s0 = ev.umbral_fdr(Zc, Q)
    calendario = amp >= cubo.AMP_DIFUSO

    # Umbral de semilla: el menor |z| de pico con (eventos nulos esperados) / (eventos observados) <= FDR_EVENTO
    M = (np.abs(Zc) >= t_z) & ~calendario[:, None, None]
    pico_obs, nodos, etiq = ev.picos(Zc, M, fechas, vecinas)
    n_comp = len(pico_obs)
    phi = ev.dispersion(Y, E, ~calendario)
    pico_nulo, colas = ev.picos_nulos(E, phi, t_z, fechas, vecinas, np.random.default_rng(SEMILLA), R_NULA)
    normales = (~calendario).mean()
    t_s = ev.umbral_nulo(pico_obs, pico_nulo, R_NULA, FDR_EVENTO, normales, desde=t_z)
    # Umbral de los eventos repartidos: la misma regla con el z de G de cada hoja
    for g in G.values():
        rng_g = np.random.default_rng(SEMILLA + 1)
        g["nulo"] = ev.ganancias_nulas(E[g["dias"]], phi, fechas[g["dias"]], g["hojas"], rng_g, R_NULA_G)
        g["obs"] = g["ZG"][~calendario[g["dias"]]]
        g["t_normal"] = g["t"]
        g["t"] = ev.umbral_nulo(g["obs"], g["nulo"], R_NULA_G, FDR_EVENTO, normales)
    colas_obs = [(np.abs(Zc[~calendario]) >= u).mean() for u in (3, 4, 5, 6)]

    L = ["Segunda etapa: eventos en el cubo regiones x intervalos x días de 2017 (los 365 días)",
         f"umbrales con FDR estimada <= {Q} (nula empírica); zonas colindantes: {int(vecinas.sum() / 2)} pares, "
         f"{int((~vecinas.any(1)).sum())} zonas sin vecinas", "",
         f"  Umbral de celda: |z| >= {t_z:.1f} (s0 = {s0:.2f}); celdas marcadas: {(np.abs(Zc) >= t_z).sum()} de {Zc.size} "
         f"({(np.abs(Zc) >= t_z).mean():.2%}); extiende la huella",
         f"  Nula simulada ({R_NULA} cubos, binomial negativa por celda; sobredispersión mediana {np.median(phi):.2f}, "
         f"percentil 90 {np.percentile(phi, 90):.2f}, máxima {phi.max():.1f})",
         "    fracción de celdas con |z| >= 3, 4, 5, 6 en días normales: observada "
         + ", ".join(f"{v:.3%}" for v in colas_obs) + "; simulada " + ", ".join(f"{v:.3%}" for v in colas),
         f"    {'pico |z| >=':<14}" + "".join(f"{u:>8}" for u in CORTES_NULA),
         f"    {'nulos por año':<14}" + "".join(f"{(pico_nulo >= u).sum() / R_NULA * normales:>8.1f}" for u in CORTES_NULA),
         f"    {'observados':<14}" + "".join(f"{(pico_obs >= u).sum():>8}" for u in CORTES_NULA),
         f"    {'FDR estimada':<14}" + "".join(f"{(pico_nulo >= u).sum() / R_NULA * normales / max((pico_obs >= u).sum(), 1):>8.2f}"
                                               for u in CORTES_NULA),
         f"  Umbral de semilla: pico |z| >= {t_s:.1f} (FDR de eventos estimada <= {FDR_EVENTO})",
         f"  Eventos repartidos: z de G de cada hoja frente a la nula simulada ({R_NULA_G} cubos)",
         *[linea for g in G for linea in (
             f"    {g}: {'z de G >=':<14}" + "".join(f"{u:>8}" for u in CORTES_G),
             f"    {'':<{len(g) + 2}}{'nulos por año':<14}"
             + "".join(f"{(G[g]['nulo'] >= u).sum() / R_NULA_G * normales:>8.1f}" for u in CORTES_G),
             f"    {'':<{len(g) + 2}}{'observados':<14}" + "".join(f"{int((G[g]['obs'] >= u).sum()):>8}" for u in CORTES_G),
             f"    {'':<{len(g) + 2}}umbral: " + (f"z de G >= {G[g]['t']:.1f}" if np.isfinite(G[g]['t'])
                                                   else "ninguno cumple la FDR; no se listan eventos repartidos")
             + f" (con la nula normal era {G[g]['t_normal']:.1f})")],
         "  Amplitud por día: " + ", ".join(f"[{a:.2f}, {b:.2f}) {((amp >= a) & (amp < b)).sum()}"
                                            for a, b in [(0, .05), (.05, .15), (.15, .25), (.25, .4), (.4, .6), (.6, 1.01)]),
         f"  Días de calendario (amplitud >= {cubo.AMP_DIFUSO}): {int(calendario.sum())}; por día de la semana "
         "(lunes a domingo): " + " ".join(str(int(calendario[fechas.dayofweek == w].sum())) for w in range(7)), ""]

    filas = []
    # 2. Días de calendario: un evento por racha de días consecutivos
    d = 0
    while d < D:
        if not calendario[d]:
            d += 1
            continue
        fin = d
        while fin + 1 < D and calendario[fin + 1]:
            fin += 1
        dias = np.arange(d, fin + 1)
        m = np.abs(Zc[dias]) >= t_z
        pico = np.unravel_index(np.abs(Zc[dias]).argmax(), Zc[dias].shape)
        filas.append({"clase": "calendario", "inicio": clave[d], "fin": clave[fin], "n_dias": len(dias),
                      "signo": "", "n_celdas": int(m.sum()), "n_regiones": int(m.any((0, 2)).sum()),
                      "region_pico": nom[pico[1]], "intervalos": "", "puntuacion": float(np.abs(Zc[dias]).max()),
                      "z_pico": float(Zc[dias][pico]), "observado": int(Y[dias].sum()), "esperado": float(E[dias].sum()),
                      "amplitud": float(amp[dias].max()), "z_G": float(zg_dia[dias].max()), "regiones": "",
                      "_celdas": [[int(dias[a]), int(b), int(c)] for a, b, c in np.argwhere(m)]})
        d = fin + 1

    # 3. Eventos agudos: componentes conexas fuera de los días de calendario
    # Los fragmentos de un solo día que coinciden en día y hora se funden en un solo evento
    retenidos = [nodos[etiq == e] for e in np.flatnonzero(pico_obs >= t_s)]
    grupo_de = ev.fusionar(retenidos, [np.sign(Zc[tuple(c[0])]) for c in retenidos])
    for e in range(grupo_de.max() + 1 if len(retenidos) else 0):
        partes = [retenidos[k] for k in np.flatnonzero(grupo_de == e)]
        partes.sort(key=lambda c: -np.abs(Zc[tuple(c.T)]).max())
        c = np.concatenate(partes)
        dd, ii, jj = c.T
        z = Zc[dd, ii, jj]
        p = int(np.abs(z).argmax())
        zg = []
        for dia in np.unique(dd):
            g = G[grupo[dia]]
            hs = np.unique(g["hoja_de"][ii[dd == dia], jj[dd == dia]])
            g["tocada"][g["local"][dia], hs] = True
            hv = [g["validas"].index(h) for h in hs if h in g["validas"]]
            if hv:
                zg.append(g["ZG"][g["local"][dia], hv].max())
        orden = pd.Series(np.abs(z)).groupby(ii).max().sort_values(ascending=False).index
        filas.append({"clase": "agudo", "inicio": clave[dd.min()], "fin": clave[dd.max()],
                      "n_dias": len(np.unique(dd)), "signo": "+" if z[p] > 0 else "-", "n_celdas": len(c),
                      "n_regiones": len(orden), "region_pico": nom[ii[p]], "intervalos": tramos(jj),
                      "puntuacion": float(np.abs(z).max()), "z_pico": float(z[p]),
                      "observado": int(Y[dd, ii, jj].sum()), "esperado": float(E[dd, ii, jj].sum()),
                      "amplitud": float(amp[np.unique(dd)].max()), "z_G": float(max(zg)) if zg else np.nan,
                      "regiones": "; ".join(nom[i] for i in orden), "n_fragmentos": len(partes), "_celdas": c.tolist(),
                      "fragmentos": " | ".join(
                          f"{nom[q[np.abs(Zc[tuple(q.T)]).argmax()][1]]} {tramos(q[:, 2])} "
                          f"(z {Zc[tuple(q[np.abs(Zc[tuple(q.T)]).argmax()])]:+.0f})" for q in partes)})

    # 4. Eventos repartidos: hojas con G alta que ningún evento agudo toca
    for g in G.values():
        for ld, a in np.argwhere(g["ZG"] >= g["t"]):
            d, k = g["dias"][ld], g["validas"][a]
            if calendario[d] or g["tocada"][ld, k]:
                continue
            f, c = g["hojas"][k]
            zc = np.abs(Zc[d][np.ix_(f, c)])
            filas.append({"clase": "repartido", "inicio": clave[d], "fin": clave[d], "n_dias": 1, "signo": "",
                          "n_celdas": len(f) * len(c), "n_regiones": len(f), "region_pico": nom[f[zc.max(1).argmax()]],
                          "intervalos": tramos(c), "puntuacion": float(g["ZG"][ld, a]), "z_pico": float(zc.max()),
                          "observado": int(Y[d][np.ix_(f, c)].sum()), "esperado": float(E[d][np.ix_(f, c)].sum()),
                          "amplitud": float(amp[d]), "z_G": float(g["ZG"][ld, a]),
                          "regiones": "; ".join(nom[i] for i in f),
                          "_celdas": [[int(d), int(i), int(j)] for i in f for j in c]})

    t = pd.DataFrame(filas)
    t["exceso"] = t.observado - t.esperado
    t["n_fragmentos"] = t.n_fragmentos.fillna(1).astype(int)
    t["fragmentos"] = t.fragmentos.fillna("")
    t["dia_semana"] = pd.to_datetime(t.inicio).dt.strftime("%a")
    t["evento"] = [" / ".join(dict.fromkeys(
        conocidos[f] for f in clave[clave.index(a):clave.index(b) + 1]
        if f in conocidos and (c == "calendario" or cubo.coincide(f, r))))
        for a, b, c, r in zip(t.inicio, t.fin, t.clase, t.regiones)]
    t = t.sort_values(["clase", "puntuacion"], ascending=[True, False]).reset_index(drop=True)
    t.insert(0, "puesto", t.groupby("clase").cumcount() + 1)
    (cubo.SALIDA / "eventos_agrupados_huellas.json").write_text(json.dumps({
        "t_z": round(t_z, 1), "t_s": round(t_s, 1),
        "eventos": [{"clase": c, "puesto": int(q), "celdas": h} for c, q, h in zip(t.clase, t.puesto, t["_celdas"])]}),
        encoding="utf-8")
    t = t.drop(columns="_celdas")
    t.round({"puntuacion": 1, "z_pico": 1, "esperado": 0, "exceso": 0, "amplitud": 2, "z_G": 1}).to_csv(
        cubo.SALIDA / "eventos_agrupados.csv", index=False)
    u = np.round(np.arange(t_z, 40.05, 0.1), 1)
    pd.DataFrame({"pico": u, "nulos_por_año": [(pico_nulo >= v).sum() / R_NULA * normales for v in u],
                  "observados": [int((pico_obs >= v).sum()) for v in u]}).to_csv(
        cubo.SALIDA / "eventos_agrupados_calibracion.csv", index=False)
    pd.DataFrame({"fecha": clave, "dia_semana": fechas.strftime("%a"), "amplitud": amp.round(3),
                  "calendario": calendario.astype(int)}).to_csv(cubo.SALIDA / "eventos_agrupados_dias.csv", index=False)

    ag = t[t.clase == "agudo"]
    w_ini, w_fin = pd.to_datetime(ag.inicio).dt.dayofweek, pd.to_datetime(ag.fin).dt.dayofweek
    finde = (w_ini >= 5).to_numpy()
    cruza = ag[(ag.n_dias > 1) & ((w_ini < 5) != (w_fin < 5))]
    L += [f"  Eventos: {len(ag)} agudos ({n_comp} componentes de {int(M.sum())} celdas; {len(retenidos)} con pico sobre el "
          f"umbral, que la fusión por día y hora deja en {len(ag)}; {int((ag.n_fragmentos > 1).sum())} con más de un fragmento; "
          f"{int((ag.n_celdas == 1).sum())} de una sola celda, {int((ag.n_dias > 1).sum())} de más de un día, "
          f"{len(cruza)} que pasan de laborables a fin de semana o al revés), {int((t.clase == 'repartido').sum())} repartidos, "
          f"{int((t.clase == 'calendario').sum())} de calendario", "",
          "  Eventos agudos con explicación (fecha y zona de EVENTOS o HALLADOS), por |z| máximo:",
          f"  {'':<26}" + "".join(f"{'>= ' + str(u):>12}" for u in CORTES)]
    expl = (ag.evento != "").to_numpy()
    for nombre, mask in (("todos", np.ones(len(ag), bool)), ("inicio de lunes a viernes", ~finde),
                         ("inicio en fin de semana", finde)):
        celdas_txt = [f"{int(((ag.puntuacion >= u).to_numpy() & expl & mask).sum())} de "
                      f"{int(((ag.puntuacion >= u).to_numpy() & mask).sum())}" for u in CORTES]
        L.append(f"  {nombre:<26}" + "".join(f"{c:>12}" for c in celdas_txt))
    L += ["", "  Eventos que pasan de laborables a fin de semana o al revés:"]
    for r in cruza.itertuples():
        L.append(f"   {r.puesto:>3}. {r.inicio} {r.dia_semana} a {r.fin[5:]}  z = {r.z_pico:+7.1f}  {r.n_celdas} celdas, "
                 f"{r.intervalos}  {r.region_pico}{'   <- ' + r.evento if r.evento else ''}")
    L += ["", "  Eventos fundidos (fragmentos de un solo día que coinciden en día y hora):"]
    for r in ag[ag.n_fragmentos > 1].itertuples():
        L.append(f"   {r.puesto:>3}. {r.inicio} {r.dia_semana}{' a ' + r.fin[5:] if r.n_dias > 1 else '':<9} "
                 f"{r.n_fragmentos} fragmentos: {r.fragmentos}{'   <- ' + r.evento if r.evento else ''}")
    L += ["", "  Los 40 eventos agudos más fuertes:"]
    for r in ag.head(40).itertuples():
        L.append(f"   {r.puesto:>3}. {r.inicio} {r.dia_semana}{' a ' + r.fin[5:] if r.n_dias > 1 else '':<9} z = {r.z_pico:+7.1f}  "
                 f"{r.n_celdas:>3} celdas, {r.n_regiones:>2} reg., {r.intervalos:<14} obs {r.observado} / esp {r.esperado:.0f}  "
                 f"{r.region_pico}{'   <- ' + r.evento if r.evento else ''}")
    L += ["", "  Eventos repartidos:"]
    for r in t[t.clase == "repartido"].head(15).itertuples():
        L.append(f"   {r.puesto:>3}. {r.inicio} {r.dia_semana}  z de G = {r.z_G:5.1f}  {r.n_regiones:>2} reg., {r.intervalos:<14} "
                 f"max |z| de celda {r.z_pico:.1f}  {r.region_pico}...{'   <- ' + r.evento if r.evento else ''}")
    L += ["", "  Días de calendario:"]
    for r in t[t.clase == "calendario"].sort_values("inicio").itertuples():
        L.append(f"        {r.inicio} {r.dia_semana}{' a ' + r.fin[5:] if r.n_dias > 1 else '':<9} amplitud {r.amplitud:.2f}, "
                 f"{r.n_celdas} celdas marcadas{'   <- ' + r.evento if r.evento else ''}")
    L += ["", "  Fechas con evento real: en qué quedaron"]
    for f, desc in conocidos.items():
        dia = t[(t.inicio <= f) & (t.fin >= f)]
        m = dia[dia.evento.str.contains(desc, regex=False)]
        if m.empty:
            L.append(f"    {f}  NO DETECTADO   {desc}  ({len(dia)} evento(s) ese día en otras zonas)")
            continue
        mejor = (m[m.clase == "calendario"].iloc[0] if (m.clase == "calendario").any()
                 else m.sort_values("puntuacion", ascending=False).iloc[0])
        L.append(f"    {f}  {mejor.clase:<10} puesto {mejor.puesto:>3}  {len(m)} evento(s) en su zona; el mayor: "
                 f"{mejor.n_celdas} celdas, {mejor.n_regiones} reg., {mejor.n_dias} día(s), {mejor.region_pico} "
                 f"{mejor.intervalos}   {desc}")

    texto = "\n".join(L)
    print(texto)
    (cubo.SALIDA / "eventos_agrupados.txt").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    main()
