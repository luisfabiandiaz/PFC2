"""
Visualización tipo checkerboard (Figuras 4 y 5 de Liu et al., 2021)
=======================================================================
Reordena la matriz de co-ocurrencia según los clusters obtenidos por
BBAC_I y dibuja el heatmap "checkerboard" resultante, con líneas rojas
delgadas marcando los límites entre region-clusters y timestamp-clusters,
igual que las Figuras 4/5 del paper (regiones y timestamps arreglados de
menor a mayor volumen, de abajo-izquierda hacia arriba-derecha).

Usa los archivos ya guardados por 01_matriz_y_coclustering.py:
  - {label}_matrix.npy
  - {label}_row_clusters.npy / {label}_col_clusters.npy
  - zone_order.csv

Requiere el entorno conda 'stcc' activo.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]      # Implementaciones/
MATRICES_DIR = RAIZ / "DATA/procesados/ST-CC"
FIGURES_DIR = RAIZ / "Resultados/ST-CC/figuras"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TIME_BIN_LABELS = [f"{h:02d}-{h + 2:02d}" for h in range(0, 24, 2)]  # 12


def reorder_by_cluster(values_1d: np.ndarray, cluster_labels: np.ndarray):
    """
    Índices que agrupan por cluster (clusters ordenados ascendente según
    su promedio) y, dentro de cada cluster, ordena los elementos
    individuales ascendente según su propio valor -- tal como describe
    el paper. Devuelve también las posiciones divisorias y un resumen
    por cluster (id, promedio, tamaño).
    """
    unique_clusters = np.unique(cluster_labels)
    cluster_avg = {c: values_1d[cluster_labels == c].mean()
                   for c in unique_clusters}
    clusters_sorted = sorted(unique_clusters, key=lambda c: cluster_avg[c])

    order = []
    boundaries = []
    cluster_info = []
    for c in clusters_sorted:
        idx_in_cluster = np.where(cluster_labels == c)[0]
        idx_in_cluster = idx_in_cluster[np.argsort(values_1d[idx_in_cluster])]
        order.extend(idx_in_cluster.tolist())
        boundaries.append(len(order))
        cluster_info.append((c, cluster_avg[c], len(idx_in_cluster)))
    boundaries = boundaries[:-1]  # no hace falta línea después del último
    return np.array(order), boundaries, cluster_info


def plot_checkerboard(matrix: np.ndarray, row_clusters: np.ndarray,
                       col_clusters: np.ndarray, zone_ids: list, label: str):
    row_avg = matrix.mean(axis=1)  # promedio por región, para ordenar
    col_avg = matrix.mean(axis=0)  # promedio por bloque horario

    row_order, row_boundaries, row_cluster_info = reorder_by_cluster(
        row_avg, row_clusters)
    col_order, col_boundaries, col_cluster_info = reorder_by_cluster(
        col_avg, col_clusters)

    reordered = matrix[np.ix_(row_order, col_order)]

    fig, ax = plt.subplots(figsize=(8, 10))
    # origin='lower' para que el volumen aumente de abajo hacia arriba,
    # como describe el paper (bottom-left -> top-right)
    im = ax.imshow(reordered, aspect="auto", cmap="YlOrRd", origin="lower")

    for b in row_boundaries:
        ax.axhline(b - 0.5, color="red", linewidth=1)
    for b in col_boundaries:
        ax.axvline(b - 0.5, color="red", linewidth=1)

    col_labels = [TIME_BIN_LABELS[i] for i in col_order]
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)

    row_labels = [str(zone_ids[i]) for i in row_order]
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_labels, fontsize=5)

    ax.set_xlabel("Bloques de 2h (reordenados por timestamp-cluster)")
    ax.set_ylabel("LocationID (reordenado por region-cluster)")
    ax.set_title(f"{label}: checkerboard "
                 f"({len(row_cluster_info)} region-clusters x "
                 f"{len(col_cluster_info)} timestamp-clusters)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Drop-off promedio por día")

    fig.tight_layout()
    out_path = FIGURES_DIR / f"checkerboard_{label.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada en {out_path}")

    print(f"\n  Timestamp-clusters de {label} (orden ascendente de volumen):")
    idx = 0
    for c, avg, n in col_cluster_info:
        bins_in_cluster = col_labels[idx:idx + n]
        print(f"    cluster {c}: {n} bloques -> {bins_in_cluster}, "
              f"promedio={avg:.1f}")
        idx += n

    print(f"\n  Region-clusters de {label} (orden ascendente de volumen):")
    for c, avg, n in row_cluster_info:
        print(f"    cluster {c}: {n} zonas, promedio={avg:.1f}")


def main():
    zone_order = pd.read_csv(MATRICES_DIR / "zone_order.csv")
    zone_ids = zone_order["LocationID"].tolist()

    for label in ["weekday", "holiday"]:
        print(f"\n{'=' * 60}\n{label.upper()}\n{'=' * 60}")
        matrix = np.load(MATRICES_DIR / f"{label}_matrix.npy")
        row_clusters = np.load(MATRICES_DIR / f"{label}_row_clusters.npy")
        col_clusters = np.load(MATRICES_DIR / f"{label}_col_clusters.npy")
        plot_checkerboard(matrix, row_clusters, col_clusters, zone_ids,
                           label=label.capitalize())

    print("\n✅ Listo. Revisa las figuras checkerboard en "
          f"'{FIGURES_DIR}/' y compáralas con las Figuras 4/5 del paper.")


if __name__ == "__main__":
    main()