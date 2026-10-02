"""
Refinamiento K-means "non-checkerboard" (Sección IV-D, Liu et al. 2021)
===========================================================================
Regrupa los K_R x K_T bloques checkerboard usando K-means sobre (media,
desviación estándar) de cada bloque -- tal como describe el paper:

    "The mean and standard deviation of each checkerboard co-cluster
    were employed as the attribute dimensions of the samples in K-means."

Esto puede juntar bloques que NO son vecinos en la rejilla original
(de ahí "non-checkerboard"): dos co-clusters con volumen/variabilidad
parecidos terminan en el mismo grupo aunque estén en region-clusters o
timestamp-clusters distintos.

Usa cgc.kmeans.KMeans restringido a SOLO (mean, std) -- por defecto cgc
usa 6 estadísticos (mean, std, min, max, p5, p95), aquí lo acotamos a
los 2 que usa el paper.

Usa los archivos ya guardados por 01_matriz_y_coclustering.py:
  - {label}_matrix.npy
  - {label}_row_clusters.npy / {label}_col_clusters.npy

Requiere el entorno conda 'stcc' activo.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # evita el warning de
# memory leak de sklearn+MKL en Windows (ver nota al final del archivo)

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn.cluster._kmeans")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from cgc.kmeans import KMeans

RAIZ = Path(__file__).resolve().parents[1]      # Implementaciones/
MATRICES_DIR = RAIZ / "DATA/procesados/ST-CC"
FIGURES_DIR = RAIZ / "Resultados/ST-CC/figuras"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K_CONFIG = {
    "weekday": {"k_row": 8, "k_col": 4},
    "holiday": {"k_row": 6, "k_col": 4},
}

# SOLO mean y std, como especifica el paper (no los 6 default de cgc)
STATISTICS = [(np.mean, None), (np.std, None)]

# El paper prueba K de 3 a 8 (excluye K=2 a propósito) y combina el
# codo de SSE con el silhouette para elegir K=4. El rango de diagnóstico
# incluye K=2 solo para poder graficar la curva completa de SSE, tal
# como la Figura 8 del paper, pero K=2 NO es un candidato válido.
DIAGNOSTIC_K_RANGE = range(2, 9)   # 2 a 8, solo para la curva
FINAL_K = 4                         # valor que el paper reporta como óptimo

STATE_LABELS = ["Very Low", "Low", "Medium", "High"]
STATE_COLORS = ["#2ca02c", "#f2e341", "#ff8c00", "#d62728"]  # verde/amarillo/naranja/rojo, como el paper


def block_stats(matrix: np.ndarray, row_clusters: np.ndarray,
                 col_clusters: np.ndarray, k_row: int, k_col: int):
    """Calcula (media, std) de cada bloque checkerboard, solo para
    imprimir/anotar -- cgc.kmeans recalcula esto internamente igual."""
    mean_grid = np.full((k_row, k_col), np.nan)
    std_grid = np.full((k_row, k_col), np.nan)
    for i in range(k_row):
        for j in range(k_col):
            block = matrix[np.ix_(row_clusters == i, col_clusters == j)]
            if block.size:
                mean_grid[i, j] = block.mean()
                std_grid[i, j] = block.std()
    return mean_grid, std_grid


def run_kmeans_refinement(matrix, row_clusters, col_clusters, k_row, k_col,
                           label: str):
    print(f"\n--- Refinamiento K-means {label} "
          f"(bloques checkerboard: {k_row}x{k_col}={k_row * k_col}) ---")

    kmeans_kwargs = {"init": "random", "n_init": 50, "random_state": 0}

    # --- Paso 1: diagnóstico, K=2 a 8, para ver silhouette + SSE juntos,
    #     igual que la Figura 8 del paper ---
    km_diag = KMeans(
        matrix,
        clusters=(row_clusters, col_clusters),
        nclusters=(k_row, k_col),
        statistics=STATISTICS,
        k_range=DIAGNOSTIC_K_RANGE,
        kmeans_kwargs=kmeans_kwargs,
    )
    diag = km_diag.compute()

    print("Curva silhouette + SSE (K=2 se muestra solo de referencia, "
          "el paper lo EXCLUYE de la selección):")
    for k, s, sse in zip(km_diag.k_range, diag.measure_list, diag.inertia):
        flag = " (excluido de la selección)" if k == 2 else ""
        print(f"    K={k:2d} -> silhouette={s:.4f}, SSE={sse:.2f}{flag}")

    # --- Paso 2: forzar K=4, tal como concluye el paper combinando
    #     el codo de SSE con el silhouette (no es simplemente el argmax
    #     de silhouette -- cgc.KMeans.k_value sí sería solo eso) ---
    km_final = KMeans(
        matrix,
        clusters=(row_clusters, col_clusters),
        nclusters=(k_row, k_col),
        statistics=STATISTICS,
        k_range=[FINAL_K],
        kmeans_kwargs=kmeans_kwargs,
    )
    results = km_final.compute()
    print(f"\nUsando K={FINAL_K} (valor que reporta el paper), "
          f"silhouette={results.measure_list[0]:.4f}")

    mean_grid, std_grid = block_stats(
        matrix, row_clusters, col_clusters, k_row, k_col)

    # cluster_averages tiene la MISMA forma que la grilla (K_R, K_T):
    # cada celda trae repetido el promedio de SU cluster refinado.
    cluster_avgs = {}
    for refined_id in range(FINAL_K):
        members = np.argwhere(results.labels == refined_id)
        i0, j0 = members[0]
        cluster_avgs[refined_id] = results.cluster_averages[i0, j0]

    # Etiqueta cada cluster como Very Low/Low/Medium/High según su
    # promedio ascendente, tal como hace el paper (Sección IV-D)
    order = sorted(cluster_avgs, key=lambda c: cluster_avgs[c])
    id_to_state = {old_id: STATE_LABELS[rank]
                   for rank, old_id in enumerate(order)}

    print("\nComposición de cada cluster refinado "
          "(region-cluster, timestamp-cluster) -> (media, std):")
    for refined_id in order:
        members = np.argwhere(results.labels == refined_id)
        print(f"  {id_to_state[refined_id]} (promedio="
              f"{cluster_avgs[refined_id]:.1f}, {len(members)} bloques):")
        for (i, j) in members:
            print(f"      region-cluster {i}, timestamp-cluster {j} "
                  f"-> media={mean_grid[i, j]:.1f}, std={std_grid[i, j]:.1f}")

    return results, mean_grid, std_grid, id_to_state, km_diag, diag


def plot_refinement(results, id_to_state: dict, mean_grid: np.ndarray,
                     k_row: int, k_col: int, label: str):
    # Mismo criterio que visualizar_checkerboard.py: ordenar region- y
    # timestamp-clusters de menor a mayor volumen promedio, y usar
    # origin='lower', para que este gráfico sea comparable a simple
    # vista con el checkerboard (si no, los IDs crudos de BBAC_I salen
    # en orden arbitrario y el eje sale invertido).
    row_order = np.argsort(mean_grid.mean(axis=1))
    col_order = np.argsort(mean_grid.mean(axis=0))

    state_to_rank = {s: i for i, s in enumerate(STATE_LABELS)}
    rank_grid = np.vectorize(lambda x: state_to_rank[id_to_state[x]])(
        results.labels)
    rank_grid = rank_grid[np.ix_(row_order, col_order)]
    labels_reordered = results.labels[np.ix_(row_order, col_order)]

    cmap = ListedColormap(STATE_COLORS)
    norm = BoundaryNorm(range(len(STATE_LABELS) + 1), cmap.N)

    fig, ax = plt.subplots(figsize=(1.3 * k_col + 2, 1.3 * k_row + 2))
    im = ax.imshow(rank_grid, cmap=cmap, norm=norm, origin="lower")

    for new_i, orig_i in enumerate(row_order):
        for new_j, orig_j in enumerate(col_order):
            state = id_to_state[labels_reordered[new_i, new_j]]
            abbrev = "".join(w[0] for w in state.split())  # VL/L/M/H
            ax.text(new_j, new_i, abbrev, ha="center", va="center",
                    fontsize=11, color="black", fontweight="bold")

    ax.set_xticks(range(k_col))
    ax.set_xticklabels([f"T{j}" for j in col_order])
    ax.set_yticks(range(k_row))
    ax.set_yticklabels([f"R{i}" for i in row_order])
    ax.set_xlabel("Timestamp-cluster (reordenado por volumen)")
    ax.set_ylabel("Region-cluster (reordenado por volumen)")
    ax.set_title(f"{label}: {len(STATE_LABELS)} clusters refinados "
                 f"(non-checkerboard)\nsobre grilla {k_row}x{k_col}, "
                 "ejes reordenados como el checkerboard")

    cbar = fig.colorbar(im, ax=ax, ticks=[i + 0.5 for i in range(4)])
    cbar.ax.set_yticklabels(STATE_LABELS)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"refinamiento_kmeans_{label.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigura guardada en {out_path}")


def plot_elbow(km_diag, diag, label: str):
    """Silhouette + SSE lado a lado, replicando la Figura 8 del paper."""
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ks = list(km_diag.k_range)

    color1 = "black"
    ax1.plot(ks, diag.measure_list, marker="s", color=color1,
              label="Silhouette")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Silhouette promedio", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axvline(FINAL_K, color="gray", linestyle="--", linewidth=1)

    ax2 = ax1.twinx()
    color2 = "red"
    ax2.plot(ks, diag.inertia, marker="o", linestyle=":", color=color2,
              label="SSE")
    ax2.set_ylabel("SSE", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"{label}: silhouette + SSE vs K\n"
                 f"(línea punteada = K={FINAL_K}, el elegido)")
    fig.tight_layout()
    out_path = FIGURES_DIR / f"elbow_{label.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figura de codo guardada en {out_path}")


def main():
    for label in ["weekday", "holiday"]:
        print(f"\n{'=' * 60}\n{label.upper()}\n{'=' * 60}")
        matrix = np.load(MATRICES_DIR / f"{label}_matrix.npy")
        row_clusters = np.load(MATRICES_DIR / f"{label}_row_clusters.npy")
        col_clusters = np.load(MATRICES_DIR / f"{label}_col_clusters.npy")
        k_row = K_CONFIG[label]["k_row"]
        k_col = K_CONFIG[label]["k_col"]

        results, mean_grid, std_grid, id_to_state, km_diag, diag = \
            run_kmeans_refinement(
                matrix, row_clusters, col_clusters, k_row, k_col, label)
        plot_refinement(results, id_to_state, mean_grid, k_row, k_col,
                         label=label.capitalize())
        plot_elbow(km_diag, diag, label=label.capitalize())

        np.save(MATRICES_DIR / f"{label}_refined_labels.npy", results.labels)

    print("\n✅ Listo. Revisa las figuras de refinamiento en "
          f"'{FIGURES_DIR}/'. Cada color (Very Low/Low/Medium/High) es un "
          "cluster non-checkerboard; si dos celdas lejanas en la rejilla "
          "comparten color, es la señal que el paper busca en la "
          "Sección IV-D.")


if __name__ == "__main__":
    main()