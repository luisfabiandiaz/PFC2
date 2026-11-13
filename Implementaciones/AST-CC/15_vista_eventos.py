"""
Vista interactiva de los eventos de la segunda etapa
====================================================
Genera una página HTML autónoma (sin dependencias ni conexión) para explorar la tabla de
AST-CC/14: línea de tiempo del año, lista filtrable de eventos y, para el evento elegido,
su huella en el mapa de zonas y el perfil horario observado frente al esperado.

El mapa se gira para que Manhattan quede horizontal (el norte a la derecha).

Entradas: Resultados/AST-CC/eventos_agrupados.csv y eventos_agrupados_huellas.json (AST-CC/14),
          DATA/procesados/AST-CC/conteos_diarios_2017.npy, DATA/crudos/taxi_zones/taxi_zones.shp
Salidas:  Resultados/AST-CC/vista_eventos.html
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import affinity

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from astcc import eventos as ev  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

DATOS, SALIDA, CRUDOS = RAIZ / "DATA/procesados", RAIZ / "Resultados/AST-CC", RAIZ / "DATA/crudos"
FRANJAS = [f"{h:02d}" for h in range(0, 24, 2)]
ANCHO = 900                 # ancho del mapa en unidades SVG
N_REGIONES = 3              # regiones con perfil horario por evento
N_DIAS = 8                  # días con perfil horario por evento


def mapa(zonas):
    """Contorno de cada zona como ruta SVG, con la isla en horizontal y el norte a la derecha."""
    shp = gpd.read_file(CRUDOS / "taxi_zones/taxi_zones.shp")
    g = shp[shp["LocationID"].isin(zonas)].dissolve("LocationID").geometry.loc[list(zonas)]
    c = np.array([(p.x, p.y) for p in g.centroid])
    eje = np.linalg.svd(c - c.mean(0))[2][0]                       # dirección principal de la isla
    if eje[1] < 0:
        eje = -eje                                                  # que apunte al norte
    angulo = -np.degrees(np.arctan2(eje[1], eje[0]))
    origen = tuple(c.mean(0))
    g = g.apply(lambda p: affinity.rotate(p, angulo, origin=origen)).simplify(120)
    x0, y0, x1, y1 = g.total_bounds
    k = ANCHO / (x1 - x0)
    rutas = []
    for p in g:
        partes = []
        for pol in getattr(p, "geoms", [p]):
            xy = np.array(pol.exterior.coords)
            pts = np.c_[(xy[:, 0] - x0) * k, (y1 - xy[:, 1]) * k].round(1)
            partes.append("M" + "L".join(f"{a:g},{b:g}" for a, b in pts) + "Z")
        rutas.append("".join(partes))
    return rutas, round((y1 - y0) * k, 1)


def main():
    zonas = pd.read_csv(DATOS / "ST-CC/zone_order.csv")["LocationID"].to_numpy()
    nombres = dict(pd.read_csv(CRUDOS / "taxi_zone_lookup.csv")[["LocationID", "Zone"]].values)
    Y = np.load(DATOS / "AST-CC/conteos_diarios_2017.npy")
    fechas = pd.date_range("2017-01-01", "2017-12-31")
    clave = fechas.strftime("%Y-%m-%d")
    Z = ev.puntuar_celdas(Y, fechas)
    E = ev.esperados(Y, fechas)
    t = pd.read_csv(SALIDA / "eventos_agrupados.csv").fillna("")
    h = json.loads((SALIDA / "eventos_agrupados_huellas.json").read_text(encoding="utf-8"))
    huella = {(e["clase"], e["puesto"]): np.array(e["celdas"]).reshape(-1, 3) for e in h["eventos"]}

    eventos = []
    for r in t.itertuples():
        c = huella[(r.clase, r.puesto)]
        dias = np.arange(list(clave).index(r.inicio), list(clave).index(r.fin) + 1)
        if len(c) == 0:                                            # día de calendario sin celdas marcadas
            c = np.array([[dias[0], i, j] for i in range(len(zonas)) for j in range(12)])
        z = Z[c[:, 0], c[:, 1], c[:, 2]]
        pico = pd.Series(z).groupby(c[:, 1]).agg(lambda v: v.iloc[np.abs(v.to_numpy()).argmax()])
        orden = pico.abs().sort_values(ascending=False).index
        perfiles = []
        for i in orden[:N_REGIONES]:
            perfiles.append({"r": int(i), "d": [
                {"f": clave[d], "o": Y[d, i].tolist(), "e": E[d, i].round(1).tolist(),
                 "m": sorted(int(j) for j in c[(c[:, 0] == d) & (c[:, 1] == i), 2])} for d in dias[:N_DIAS]]})
        eventos.append({
            "id": f"{r.clase}-{r.puesto}", "c": r.clase, "p": int(r.puesto), "ini": r.inicio, "fin": r.fin,
            "nd": int(r.n_dias), "ds": r.dia_semana, "pt": float(r.puntuacion), "zp": float(r.z_pico),
            "nc": int(r.n_celdas), "nr": int(r.n_regiones), "rp": r.region_pico, "iv": r.intervalos,
            "obs": int(r.observado), "esp": float(r.esperado), "amp": float(r.amplitud),
            "zg": None if r.z_G == "" else float(r.z_G), "ev": r.evento, "nf": int(r.n_fragmentos),
            "doy": int(dias[0]), "reg": [[int(i), round(float(v), 1)] for i, v in pico.items()], "perf": perfiles})

    rutas, alto = mapa(zonas)
    datos = {"zonas": [{"n": nombres[z], "d": d} for z, d in zip(zonas, rutas)], "ancho": ANCHO, "alto": alto,
             "franjas": FRANJAS, "eventos": eventos, "t_z": h["t_z"], "t_s": h["t_s"],
             "meses": [int(fechas.get_loc(f"2017-{m:02d}-01")) for m in range(1, 13)]}
    html = PLANTILLA.replace("__DATOS__", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    (SALIDA / "vista_eventos.html").write_text(html, encoding="utf-8")
    print(f"{len(eventos)} eventos; {len(html) / 1e6:.2f} MB -> {SALIDA / 'vista_eventos.html'}")


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eventos de 2017 en el cubo de taxis</title>
<style>
:root {
  --fondo: #f6f5f1; --panel: #ffffff; --tinta: #1d1f23; --tenue: #666b74; --linea: #dcd9d0;
  --agudo: #c2410c; --repartido: #1d4ed8; --calendario: #7c7f87; --sel: #fff3d6; --zona: #e7e4db;
  --pos: 194, 65, 12; --neg: 29, 78, 216;
}
@media (prefers-color-scheme: dark) {
  :root { --fondo: #15171b; --panel: #1e2126; --tinta: #e9e7e1; --tenue: #9a9fa8; --linea: #343840;
          --agudo: #fb923c; --repartido: #7aa2ff; --calendario: #8b9099; --sel: #3a3220; --zona: #2c3037;
          --pos: 251, 146, 60; --neg: 122, 162, 255; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--fondo); color: var(--tinta);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 18px 20px 6px; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 650; }
header p { margin: 0; color: var(--tenue); max-width: 95ch; }
.panel { background: var(--panel); border: 1px solid var(--linea); border-radius: 8px; }
#tiempo { margin: 12px 20px; padding: 8px 10px 2px; }
#tiempo svg { display: block; width: 100%; height: auto; }
main { display: grid; grid-template-columns: minmax(300px, 440px) 1fr; gap: 12px; padding: 0 20px 24px; align-items: start; }
@media (max-width: 900px) { main { grid-template-columns: 1fr; } header, #tiempo, main { padding-left: 16px; padding-right: 16px; }
                            #tiempo { margin-left: 16px; margin-right: 16px; } }
#filtros { padding: 10px 12px; display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; border-bottom: 1px solid var(--linea); }
#filtros label { display: inline-flex; gap: 5px; align-items: center; cursor: pointer; }
#filtros input[type=search] { flex: 1 1 140px; min-width: 0; padding: 5px 8px; border: 1px solid var(--linea); border-radius: 6px;
                              background: var(--fondo); color: var(--tinta); font: inherit; }
#cuenta { color: var(--tenue); font-size: 12px; width: 100%; }
#lista { max-height: 72vh; overflow: auto; }
.fila { display: grid; grid-template-columns: 54px 1fr auto; gap: 2px 10px; padding: 8px 12px; border-bottom: 1px solid var(--linea);
        cursor: pointer; }
.fila:hover { background: var(--fondo); }
.fila.sel { background: var(--sel); }
.fila .pt { font-variant-numeric: tabular-nums; font-weight: 650; text-align: right; }
.fila .sub { grid-column: 2 / 4; color: var(--tenue); font-size: 12.5px; }
.fila .ev { grid-column: 2 / 4; font-size: 12.5px; }
.marca { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
.agudo { background: var(--agudo); } .repartido { background: var(--repartido); } .calendario { background: var(--calendario); }
#detalle { padding: 14px 16px 18px; min-width: 0; }
#detalle h2 { margin: 0 0 2px; font-size: 18px; }
#detalle .que { color: var(--tenue); margin: 0 0 10px; }
.cifras { display: flex; flex-wrap: wrap; gap: 8px 22px; margin: 10px 0 12px; }
.cifras div { min-width: 90px; }
.cifras b { display: block; font-size: 18px; font-variant-numeric: tabular-nums; }
.cifras span { color: var(--tenue); font-size: 12px; }
h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--tenue); margin: 16px 0 6px; font-weight: 600; }
#mapa svg { width: 100%; height: auto; display: block; }
#mapa path { stroke: var(--panel); stroke-width: 1; }
#mapa path:hover { stroke: var(--tinta); stroke-width: 1.6; }
.escala { display: flex; align-items: center; gap: 8px; color: var(--tenue); font-size: 12px; margin-top: 4px; flex-wrap: wrap; }
.escala i { display: inline-block; width: 140px; height: 9px; border-radius: 4px;
            background: linear-gradient(90deg, rgb(var(--neg)), var(--zona), rgb(var(--pos))); }
#pestanas { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
#pestanas button { font: inherit; padding: 4px 10px; border-radius: 14px; border: 1px solid var(--linea); background: var(--fondo);
                   color: var(--tinta); cursor: pointer; }
#pestanas button.sel { background: var(--tinta); color: var(--panel); border-color: var(--tinta); }
#perfiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; }
#perfiles figure { margin: 0; border: 1px solid var(--linea); border-radius: 6px; padding: 6px 8px 2px; }
#perfiles figcaption { font-size: 12px; color: var(--tenue); }
#perfiles svg { width: 100%; height: auto; display: block; }
.leyenda { color: var(--tenue); font-size: 12px; margin-top: 6px; }
.leyenda b { font-weight: 600; }
text { fill: var(--tenue); font-size: 10px; }
</style>
</head>
<body>
<header>
  <h1>Eventos de 2017 en el cubo de llegadas de taxi</h1>
  <p id="resumen"></p>
</header>
<section id="tiempo" class="panel"></section>
<main>
  <section class="panel">
    <div id="filtros">
      <label><input type="checkbox" data-c="agudo" checked><span class="marca agudo"></span>Agudos</label>
      <label><input type="checkbox" data-c="repartido" checked><span class="marca repartido"></span>Repartidos</label>
      <label><input type="checkbox" data-c="calendario" checked><span class="marca calendario"></span>Calendario</label>
      <label><input type="checkbox" id="soloexp">Solo identificados</label>
      <input type="search" id="buscar" placeholder="Buscar zona, fecha o evento">
      <div id="cuenta"></div>
    </div>
    <div id="lista"></div>
  </section>
  <section class="panel" id="detalle"></section>
</main>
<script>
const D = __DATOS__;
const $ = (s) => document.querySelector(s);
const NS = "http://www.w3.org/2000/svg";
const NOMBRE = { agudo: "Evento agudo", repartido: "Evento repartido", calendario: "Día de calendario" };
const MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
const DIAS = { Mon: "lun", Tue: "mar", Wed: "mié", Thu: "jue", Fri: "vie", Sat: "sáb", Sun: "dom" };
let actual = null, region = 0;

function fecha(f) { const [, m, d] = f.split("-"); return +d + " " + MESES[+m - 1]; }
function rango(e) { return e.nd > 1 ? fecha(e.ini) + " – " + fecha(e.fin) : DIAS[e.ds] + " " + fecha(e.ini); }
function num(v) { return Math.round(v).toLocaleString("es"); }
function color(z) {
  const a = Math.min(1, Math.sqrt(Math.abs(z) / 20));
  return "rgba(" + getComputedStyle(document.documentElement).getPropertyValue(z >= 0 ? "--pos" : "--neg") + "," + (0.18 + 0.82 * a).toFixed(2) + ")";
}
function visibles() {
  const clases = [...document.querySelectorAll("#filtros input[data-c]")].filter((c) => c.checked).map((c) => c.dataset.c);
  const q = $("#buscar").value.trim().toLowerCase(), solo = $("#soloexp").checked;
  return D.eventos.filter((e) => clases.includes(e.c) && (!solo || e.ev) &&
    (!q || (e.rp + " " + e.ev + " " + e.ini + " " + rango(e) + " " + e.reg.map((r) => D.zonas[r[0]].n).join(" ")).toLowerCase().includes(q)));
}

function pintarTiempo(lista) {
  const W = 1200, H = 150, m = { l: 34, r: 8, t: 8, b: 20 };
  const x = (d) => m.l + (d / 364) * (W - m.l - m.r);
  const y = (p) => H - m.b - (Math.log10(Math.max(p, 3)) - Math.log10(3)) / (Math.log10(150) - Math.log10(3)) * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Línea de tiempo de eventos">`;
  D.eventos.filter((e) => e.c === "calendario").forEach((e) => {
    s += `<rect x="${x(e.doy) - 1.5}" y="${m.t}" width="${Math.max(3, x(e.doy + e.nd) - x(e.doy))}" height="${H - m.t - m.b}" fill="var(--calendario)" opacity="${lista.includes(e) ? 0.28 : 0.08}" data-id="${e.id}" style="cursor:pointer"><title>${rango(e)}: día de calendario${e.ev ? " (" + e.ev + ")" : ""}</title></rect>`;
  });
  [5, 10, 20, 50, 100].forEach((v) => { s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--linea)"/><text x="${m.l - 5}" y="${y(v) + 3}" text-anchor="end">${v}</text>`; });
  D.meses.forEach((d, i) => { s += `<text x="${x(d) + 2}" y="${H - 6}">${MESES[i]}</text><line x1="${x(d)}" x2="${x(d)}" y1="${H - m.b}" y2="${H - m.b + 4}" stroke="var(--tenue)"/>`; });
  lista.filter((e) => e.c !== "calendario").forEach((e) => {
    s += `<circle cx="${x(e.doy + (e.nd - 1) / 2)}" cy="${y(e.pt)}" r="${3 + Math.sqrt(e.nc) * 0.6}" fill="var(--${e.c})" opacity="${e === actual ? 1 : 0.6}" stroke="${e === actual ? "var(--tinta)" : "none"}" stroke-width="2" data-id="${e.id}" style="cursor:pointer"><title>${rango(e)} · ${e.rp}${e.ev ? " · " + e.ev : ""}</title></circle>`;
  });
  $("#tiempo").innerHTML = s + "</svg>";
  $("#tiempo").querySelectorAll("[data-id]").forEach((n) => n.addEventListener("click", () => elegir(n.dataset.id, true)));
}

function pintarLista() {
  const lista = visibles();
  $("#cuenta").textContent = lista.length + " de " + D.eventos.length + " eventos";
  $("#lista").innerHTML = lista.map((e) => `<div class="fila${e === actual ? " sel" : ""}" data-id="${e.id}">
      <span><span class="marca ${e.c}"></span>${e.p}</span><span><b>${rango(e)}</b> · ${e.rp}</span>
      <span class="pt" title="${e.c === "repartido" ? "z de G" : "|z| máximo"}">${e.pt.toFixed(1)}</span>
      <span class="sub">${e.c === "calendario" ? "amplitud " + e.amp.toFixed(2) + " · " + e.nc + " celdas marcadas" : e.nc + " celdas · " + e.nr + " zonas · " + e.iv}</span>
      ${e.ev ? `<span class="ev">${e.ev}</span>` : ""}</div>`).join("");
  $("#lista").querySelectorAll(".fila").forEach((n) => n.addEventListener("click", () => elegir(n.dataset.id, false)));
  pintarTiempo(lista);
}

function pintarMapa(e) {
  const z = new Map(e.reg);
  let s = `<svg viewBox="-4 -4 ${D.ancho + 8} ${D.alto + 8}" role="img" aria-label="Huella del evento por zona">`;
  D.zonas.forEach((zo, i) => {
    const v = z.get(i);
    s += `<path d="${zo.d}" fill="${v === undefined ? "var(--zona)" : color(v)}"><title>${zo.n}${v === undefined ? "" : ": z = " + (v > 0 ? "+" : "") + v}</title></path>`;
  });
  return s + `</svg><div class="escala"><span>menos llegadas</span><i></i><span>más llegadas</span><span>· norte a la derecha · pasa el cursor por una zona para ver su z</span></div>`;
}

function grafico(dia) {
  const W = 230, H = 120, m = { l: 30, r: 6, t: 6, b: 16 };
  const max = Math.max(...dia.o, ...dia.e, 1) * 1.08;
  const x = (j) => m.l + (j / 11) * (W - m.l - m.r), y = (v) => H - m.b - (v / max) * (H - m.t - m.b);
  const linea = (a) => a.map((v, j) => (j ? "L" : "M") + x(j).toFixed(1) + "," + y(v).toFixed(1)).join("");
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Perfil horario del ${dia.f}">`;
  [0, 0.5, 1].forEach((f) => { const v = max / 1.08 * f; s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--linea)"/><text x="${m.l - 4}" y="${y(v) + 3}" text-anchor="end">${num(v)}</text>`; });
  [0, 3, 6, 9].forEach((j) => { s += `<text x="${x(j)}" y="${H - 3}" text-anchor="middle">${D.franjas[j]}</text>`; });
  s += `<path d="${linea(dia.e)}" fill="none" stroke="var(--tenue)" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  s += `<path d="${linea(dia.o)}" fill="none" stroke="var(--${actual.c})" stroke-width="2"/>`;
  dia.o.forEach((v, j) => { s += `<circle cx="${x(j)}" cy="${y(v)}" r="${dia.m.includes(j) ? 3.6 : 1.6}" fill="var(--${actual.c})"><title>${D.franjas[j]} h: ${num(v)} llegadas, ${num(dia.e[j])} esperadas</title></circle>`; });
  return s + "</svg>";
}

function pintarPerfiles() {
  const e = actual, p = e.perf[region];
  $("#pestanas").innerHTML = e.perf.map((q, k) => `<button class="${k === region ? "sel" : ""}" data-k="${k}">${D.zonas[q.r].n}</button>`).join("");
  $("#pestanas").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { region = +b.dataset.k; pintarPerfiles(); }));
  $("#perfiles").innerHTML = p.d.map((dia) => `<figure>${grafico(dia)}<figcaption>${fecha(dia.f)}</figcaption></figure>`).join("");
}

function pintarDetalle() {
  const e = actual;
  if (!e) { $("#detalle").innerHTML = "<p class='que'>Elige un evento de la lista o de la línea de tiempo.</p>"; return; }
  const razon = e.esp > 0 ? (e.obs / e.esp) : 0;
  $("#detalle").innerHTML = `<h2>${rango(e)} · ${e.rp}</h2>
    <p class="que"><span class="marca ${e.c}"></span>${NOMBRE[e.c]} n.º ${e.p}${e.ev ? " · " + e.ev : " · sin evento identificado"}${e.nf > 1 ? " · " + e.nf + " fragmentos fundidos" : ""}</p>
    <div class="cifras">
      <div><b>${(e.zp > 0 ? "+" : "") + e.zp.toFixed(1)}</b><span>z de la celda pico</span></div>
      <div><b>${num(e.obs)}</b><span>llegadas observadas</span></div>
      <div><b>${num(e.esp)}</b><span>esperadas</span></div>
      <div><b>× ${razon.toFixed(2)}</b><span>observado / esperado</span></div>
      <div><b>${e.nc}</b><span>celdas en ${e.nr} zonas</span></div>
      <div><b>${e.amp.toFixed(2)}</b><span>amplitud del día</span></div>
      ${e.zg === null ? "" : `<div><b>${e.zg.toFixed(1)}</b><span>z de G de la hoja</span></div>`}
    </div>
    <h3>Huella por zona</h3><div id="mapa">${pintarMapa(e)}</div>
    <h3>Perfil horario: observado frente a esperado</h3><div id="pestanas"></div><div id="perfiles"></div>
    <p class="leyenda"><b>Línea continua</b>: llegadas del día. <b>Línea discontinua</b>: esperadas según los demás días del mismo día de la semana. <b>Puntos grandes</b>: intervalos que forman parte del evento. Eje horizontal: hora de inicio del intervalo de 2 h.</p>`;
  pintarPerfiles();
}

function elegir(id, desplazar) {
  actual = D.eventos.find((e) => e.id === id) || null; region = 0;
  pintarLista(); pintarDetalle();
  if (desplazar) { const n = $("#lista .sel"); if (n) n.scrollIntoView({ block: "nearest" }); }
}

const n = (c) => D.eventos.filter((e) => e.c === c).length;
$("#resumen").textContent = `Cada día de 2017 se compara con los del mismo día de la semana. ${n("agudo")} eventos agudos (celdas vecinas con |z| ≥ ${D.t_z} y un pico ≥ ${D.t_s}), ${n("repartido")} repartidos (una hoja de AST-CC cambia de reparto) y ${n("calendario")} días de calendario (cambia toda la ciudad). El eje vertical de la línea de tiempo es la puntuación del evento en escala logarítmica.`;
document.querySelectorAll("#filtros input").forEach((i) => i.addEventListener("input", pintarLista));
elegir(D.eventos.find((e) => e.c === "agudo").id, false);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
