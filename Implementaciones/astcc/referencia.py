"""
Configuración de referencia de AST-CC sobre los datos reales (AST-CC/11_seleccion_delta.py).

Base. El árbol refina la partición de ST-CC tal cual (ST-CC/01), la que promete el título:
así toda mejora se atribuye al árbol. Las particiones de BBAC C5 (AST-CC/01) y la rejilla
optimizada para C6 (AST-CC/05) quedan como análisis de sensibilidad. En los días laborables
el árbol completo llega al mismo ajuste fuera de muestra con las tres bases.

Delta. La validación en el resto de 2017 no puede fijarlo: la pérdida fuera de muestra baja
de forma monótona hasta el modelo saturado, y los cortes se replican y son estables para
todo delta de la rejilla. Por eso delta es un umbral de resolución con dos niveles:
  completo  DELTA_COMPLETO, donde el árbol deja de cambiar; es el modelo que se compara
            con ST-CC y con las rejillas
  resumen   DELTA_RESUMEN, del orden de 10 a 15 cortes por periodo, para interpretar y
            dibujar (los bloques de ST-CC son más heterogéneos en el fin de semana)
"""

from pathlib import Path

import numpy as np

DATOS = Path(__file__).resolve().parents[1] / "DATA/procesados"

BASE = "ST"
BASES = ("ST", "C5", "C6")
NOMBRE_BASE = {"ST": "ST-CC", "C2": "BBAC C2", "C5": "BBAC C5", "C6": "rejilla C6"}
DELTA_COMPLETO = 0.00025
DELTA_RESUMEN = {"weekday": 0.005, "holiday": 0.02}
# Semilla de todo ajuste del árbol sobre los datos reales: con ella, el árbol completo y el
# resumido son los mismos en todos los scripts (las réplicas nulas dependen del azar, y
# unos pocos cortes cercanos al umbral cambian con otra semilla).
SEMILLA_ARBOL = 20261004


def cargar_base(periodo, nombre=BASE):
    """(rho, tau) de la partición base: 'ST' (ST-CC/01), 'C2' o 'C5' (AST-CC/01), 'C6' (AST-CC/05)."""
    if nombre == "ST":
        return (np.load(DATOS / f"ST-CC/{periodo}_row_clusters.npy"),
                np.load(DATOS / f"ST-CC/{periodo}_col_clusters.npy"))
    return (np.load(DATOS / f"AST-CC/{periodo}_{nombre}_rho.npy"),
            np.load(DATOS / f"AST-CC/{periodo}_{nombre}_tau.npy"))
