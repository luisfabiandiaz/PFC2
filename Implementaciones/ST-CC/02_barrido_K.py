"""
Barrido de K_R (region-clusters) y K_T (timestamp-clusters)
=============================================================
Replica la Sección IV-B y la Figura 3 de Liu et al. (2021): barre el
número de clusters buscando dónde la función objetivo (I-divergencia)
deja de mejorar significativamente, para justificar la elección de
K_R y K_T en vez de fijarlos a ciegas.

Usa las matrices YA GUARDADAS por 01_matriz_y_coclustering.py
(weekday_matrix.npy, holiday_matrix.npy) -- no vuelve a tocar los
parquet crudos.

Metodología:
  - Barrido 1: se fija K_T=4 (valor que el paper reporta estable) y
    se barre K_R en [4, 15], igual que la Figura 3 del paper.
  - Barrido 2: se fija K_R en el valor reportado por el paper
    (8 para weekday, 6 para holiday) y se barre K_T en [3, 8].
  - Cada punto (K_R, K_T) corre BBAC_I con nruns inicializaciones
    aleatorias, quedándose con la de menor I-divergencia real.

Requiere el entorno conda 'stcc' activo (incluye matplotlib).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # guardar figuras sin necesitar display interactivo
import matplotlib.pyplot as plt
from pathlib import Path
from cgc.coclustering import Coclustering

RAIZ = Path(__file__).resolve().parents[1]      # Implementaciones/
MATRICES_DIR = RAIZ / "DATA/procesados/ST-CC"
FIGURES_DIR = RAIZ / "Resultados/ST-CC/figuras"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---- Configuración del barrido ----
K_R_RANGE = range(4, 16)   # 4 a 15, como el paper
K_T_RANGE = range(3, 9)    # 3 a 8, como el paper
K_T_FIXED = 4               # valor que el paper reporta como óptimo/estable
K_R_FIXED = {"weekday": 8, "holiday": 6}  # valores reportados por el paper

NRUNS_SWEEP = 50  # inicializaciones aleatorias por cada (K_R,K_T) probado


def true_i_divergence_constant(Z: np.ndarray) -> float:
    """
    Constante O*log(O) - O que cgc omite de su 'error' interno (ver
    01_matriz_y_coclustering.py para la explicación completa).
    Sumada al error de cgc, da la I-divergencia real D_I(O || Ô).
    """
    nonzero = Z > 0
    return float(np.sum(Z[nonzero] * np.log(Z[nonzero]) - Z[nonzero]))


def divergence_for_k(matrix: np.ndarray, k_row: int, k_col: int,
                      nruns: int = NRUNS_SWEEP) -> float:
    cc = Coclustering(
        matrix,
        nclusters_row=k_row,
        nclusters_col=k_col,
        max_iterations=100,
        conv_threshold=1e-5,
        nruns=nruns,
    )
    results = cc.run_with_threads(nthreads=4)
    constant = true_i_divergence_constant(matrix)
    return constant + results.error


def sweep_k_row(matrix: np.ndarray, k_col_fixed: int, k_range) -> dict:
    print(f"  Barriendo K_R en {list(k_range)} (K_T fijo = {k_col_fixed})...")
    out = {}
    for k in k_range:
        d = divergence_for_k(matrix, k_row=k, k_col=k_col_fixed)
        out[k] = d
        print(f"    K_R={k:2d} -> D_I = {d:.2f}")
    return out


def sweep_k_col(matrix: np.ndarray, k_row_fixed: int, k_range) -> dict:
    print(f"  Barriendo K_T en {list(k_range)} (K_R fijo = {k_row_fixed})...")
    out = {}
    for k in k_range:
        d = divergence_for_k(matrix, k_row=k_row_fixed, k_col=k)
        out[k] = d
        print(f"    K_T={k:2d} -> D_I = {d:.2f}")
    return out


def suggest_elbow(results: dict, drop_threshold: float = 0.05) -> int:
    """
    Heurística simple: primer K donde la mejora relativa respecto al
    K anterior cae por debajo de drop_threshold (5% por defecto).
    Es solo una SUGERENCIA -- la inspección visual de la curva (como
    hace el propio paper) sigue siendo el criterio principal.
    """
    ks = sorted(results.keys())
    for i in range(1, len(ks)):
        prev_d, curr_d = results[ks[i - 1]], results[ks[i]]
        if prev_d == 0:
            continue
        improvement = (prev_d - curr_d) / prev_d
        if improvement < drop_threshold:
            return ks[i - 1]
    return ks[-1]


def plot_sweep(row_results: dict, col_results: dict, label: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ks = sorted(row_results.keys())
    ds = [row_results[k] for k in ks]
    axes[0].plot(ks, ds, marker="o")
    axes[0].set_xlabel("K_R (region-clusters)")
    axes[0].set_ylabel("I-divergencia real  D_I(O || Ô)")
    axes[0].set_title(f"{label}: barrido K_R  (K_T={K_T_FIXED} fijo)")
    axes[0].grid(alpha=0.3)

    ks2 = sorted(col_results.keys())
    ds2 = [col_results[k] for k in ks2]
    axes[1].plot(ks2, ds2, marker="o", color="darkorange")
    axes[1].set_xlabel("K_T (timestamp-clusters)")
    axes[1].set_ylabel("I-divergencia real  D_I(O || Ô)")
    axes[1].set_title(
        f"{label}: barrido K_T  (K_R={K_R_FIXED[label.lower()]} fijo)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"barrido_K_{label.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada en {out_path}")


def main():
    for label in ["weekday", "holiday"]:
        print(f"\n{'=' * 60}\n{label.upper()}\n{'=' * 60}")
        matrix = np.load(MATRICES_DIR / f"{label}_matrix.npy")

        row_results = sweep_k_row(
            matrix, k_col_fixed=K_T_FIXED, k_range=K_R_RANGE)
        col_results = sweep_k_col(
            matrix, k_row_fixed=K_R_FIXED[label], k_range=K_T_RANGE)

        elbow_r = suggest_elbow(row_results)
        elbow_t = suggest_elbow(col_results)
        print(f"\n  Sugerencia (heurística, revisa la gráfica): "
              f"K_R≈{elbow_r}, K_T≈{elbow_t}")
        print(f"  Valor reportado por el paper: "
              f"K_R={K_R_FIXED[label]}, K_T={K_T_FIXED}")

        plot_sweep(row_results, col_results, label=label.capitalize())

    print("\n✅ Barrido completo. Revisa las figuras en "
          f"'{FIGURES_DIR}/' y compáralas con la Figura 3 del paper.")


if __name__ == "__main__":
    main()