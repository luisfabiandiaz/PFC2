"""
Estabilidad de las hojas y ajuste fuera de muestra (Cap_3.tex, sec:prop-representacion,
"Frecuencia de soporte", y sec:prop-parada, "Elección de delta").

Una hoja es un rectángulo I x J de celdas, así que el índice de Jaccard entre dos hojas
se calcula sin enumerar celdas: |I ∩ I'| |J ∩ J'| es el tamaño de la intersección.
"""

import numpy as np
from scipy.special import xlogy

KAPPA = 0.5


def jaccard(filas1, cols1, filas2, cols2):
    """Índice de Jaccard, medido sobre celdas, entre dos rectángulos."""
    inter = len(np.intersect1d(filas1, filas2)) * len(np.intersect1d(cols1, cols2))
    union = len(filas1) * len(cols1) + len(filas2) * len(cols2) - inter
    return inter / union if union else 0.0


def hay_hoja_similar(hoja, hojas, kappa=KAPPA):
    """True si alguna hoja de la lista tiene Jaccard >= kappa con `hoja`."""
    return any(jaccard(hoja.filas, hoja.cols, h.filas, h.cols) >= kappa for h in hojas)


def frecuencia_de_soporte(hojas_ref, ejecuciones, kappa=KAPPA):
    """Para cada hoja de referencia, la proporción de ejecuciones (listas de hojas) en
    que aparece una hoja con Jaccard >= kappa respecto de ella."""
    return np.array([np.mean([hay_hoja_similar(h, hojas, kappa) for hojas in ejecuciones])
                     for h in hojas_ref])


def concordancia(hojas_a, hojas_b, kappa=KAPPA):
    """Proporción de hojas de `hojas_a` que tienen una similar en `hojas_b`."""
    return float(np.mean([hay_hoja_similar(h, hojas_b, kappa) for h in hojas_a]))


def divergencia_fuera_de_muestra(O_prueba, ajustado_entrenamiento, suavizado=0.5):
    """D_I(O_prueba || prediccion). La predicción es el ajuste del modelo sobre los
    días de entrenamiento, con `suavizado` viajes añadidos a cada celda para que una
    celda vacía en el entrenamiento no produzca divergencia infinita, y reescalado a
    la masa de la matriz de prueba (la matriz es la suma de días, y los dos conjuntos
    tienen distinta cantidad de ellos)."""
    A = np.asarray(ajustado_entrenamiento, float) + suavizado
    P = A * (O_prueba.sum() / A.sum())
    return float(np.sum(xlogy(O_prueba, O_prueba) - xlogy(O_prueba, P) - O_prueba + P))
