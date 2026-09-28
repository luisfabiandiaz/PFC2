"""
Pipeline de reconstrucción de ST-CC (Liu et al., 2021, IEEE Access)
=====================================================================
Construye la matriz de co-ocurrencia región x timestamp (66 x 12)
para 'weekday' y 'holiday' en noviembre 2017, y corre BBAC_I
(vía el paquete cgc) sobre cada una, usando los K_R/K_T que el
paper reporta como óptimos.

Metodología replicada del paper (Sección III-A y IV-A/B):
  - Se usan viajes de DROP-OFF (destino), yellow + green combinados.
  - Se filtra a las 66 zonas de Manhattan (excluyendo las 3 zonas-isla).
  - El día se divide en 12 bloques de 2h: [0-2), [2-4), ..., [22-24).
  - El valor de cada celda es el PROMEDIO de viajes por día en ese
    bloque (no la suma total), para que weekday (12 días) y holiday
    (9 días) queden en la misma escala física.
  - weekday = martes/miércoles/jueves de nov-2017, excluyendo el
    miércoles previo a Thanksgiving y Thanksgiving -> 12 días.
  - holiday = todos los sábados, domingos y Thanksgiving -> 9 días.
  - K óptimos reportados por el paper: weekday K_R=8,K_T=4;
    holiday K_R=6,K_T=4 (Sección IV-B, Figura 3).

Requiere el entorno conda 'stcc' activo (duckdb, pandas, numpy, cgc).
"""

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from cgc.coclustering import Coclustering

# ----------------------------------------------------------------
# 0. Configuración de rutas (relativas a Implementaciones/)
# ----------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]      # Implementaciones/
CRUDOS_DIR = RAIZ / "DATA/crudos"
YELLOW_PATH = CRUDOS_DIR / "yellow_tripdata_2017-11.parquet"
GREEN_PATH = CRUDOS_DIR / "green_tripdata_2017-11.parquet"
ZONES_CSV = CRUDOS_DIR / "taxi_zone_lookup.csv"
OUTPUT_DIR = RAIZ / "DATA/procesados/ST-CC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------
# 1. Días weekday / holiday, exactamente como en el paper
# ----------------------------------------------------------------
WEEKDAY_DAYS = [1, 2, 7, 8, 9, 14, 15, 16, 21, 28, 29, 30]
HOLIDAY_DAYS = [4, 5, 11, 12, 18, 19, 23, 25, 26]

assert len(WEEKDAY_DAYS) == 12, "Deben ser 12 días weekday, según el paper"
assert len(HOLIDAY_DAYS) == 9, "Deben ser 9 días holiday, según el paper"

N_TIME_BINS = 12  # bloques de 2 horas

# K óptimos reportados en el paper (Sección IV-B)
K_CONFIG = {
    "weekday": {"k_row": 8, "k_col": 4},
    "holiday": {"k_row": 6, "k_col": 4},
}

# ----------------------------------------------------------------
# 2. Zonas válidas de Manhattan (66, excluyendo las 3 zonas-isla)
# ----------------------------------------------------------------
con = duckdb.connect()

zones_df = con.execute(f"SELECT * FROM '{ZONES_CSV}'").fetchdf()
manhattan = zones_df[zones_df["Borough"] == "Manhattan"]
islands_mask = manhattan["Zone"].str.contains(
    "Governor|Ellis|Liberty", case=False, na=False)
valid_zones = sorted(manhattan.loc[~islands_mask, "LocationID"].tolist())

assert len(valid_zones) == 66, f"Se esperaban 66 zonas, se obtuvieron {len(valid_zones)}"
print(f"Zonas de Manhattan válidas: {len(valid_zones)}")

zone_id_to_idx = {zid: i for i, zid in enumerate(valid_zones)}

# ----------------------------------------------------------------
# 3. Query combinada yellow + green, dropoff, noviembre 2017
# ----------------------------------------------------------------
zones_sql_list = ",".join(str(z) for z in valid_zones)

query = f"""
WITH trips AS (
    SELECT
        DOLocationID AS zone_id,
        tpep_dropoff_datetime AS dropoff_ts
    FROM '{YELLOW_PATH}'
    WHERE tpep_dropoff_datetime >= TIMESTAMP '2017-11-01'
      AND tpep_dropoff_datetime <  TIMESTAMP '2017-12-01'

    UNION ALL

    SELECT
        DOLocationID AS zone_id,
        lpep_dropoff_datetime AS dropoff_ts
    FROM '{GREEN_PATH}'
    WHERE lpep_dropoff_datetime >= TIMESTAMP '2017-11-01'
      AND lpep_dropoff_datetime <  TIMESTAMP '2017-12-01'
)
SELECT
    zone_id,
    EXTRACT(day  FROM dropoff_ts) AS day,
    EXTRACT(hour FROM dropoff_ts) AS hour
FROM trips
WHERE zone_id IN ({zones_sql_list})
"""

print("Ejecutando query combinada yellow+green (puede tardar uno o dos minutos)...")
trips_df = con.execute(query).fetchdf()
print(f"Viajes con drop-off en zonas de Manhattan válidas: {len(trips_df):,}")

trips_df["time_bin"] = (trips_df["hour"] // 2).astype(int)  # 0..11


# ----------------------------------------------------------------
# 4. Construcción de las dos matrices 66x12 (promedio diario)
# ----------------------------------------------------------------
def build_matrix(df: pd.DataFrame, day_list: list) -> np.ndarray:
    subset = df[df["day"].isin(day_list)]
    mat = np.zeros((len(valid_zones), N_TIME_BINS), dtype=float)
    counts = subset.groupby(["zone_id", "time_bin"]).size()
    for (zid, tbin), n in counts.items():
        mat[zone_id_to_idx[zid], tbin] = n
    mat /= len(day_list)  # promedio por día, como en el paper
    return mat


weekday_matrix = build_matrix(trips_df, WEEKDAY_DAYS)
holiday_matrix = build_matrix(trips_df, HOLIDAY_DAYS)

print(f"\nMatriz weekday: shape={weekday_matrix.shape}, "
      f"suma total={weekday_matrix.sum():.1f}, "
      f"min={weekday_matrix.min():.1f}, max={weekday_matrix.max():.1f}")
print(f"Matriz holiday: shape={holiday_matrix.shape}, "
      f"suma total={holiday_matrix.sum():.1f}, "
      f"min={holiday_matrix.min():.1f}, max={holiday_matrix.max():.1f}")

# Guardamos las matrices y el orden de zonas, para reproducibilidad
np.save(OUTPUT_DIR / "weekday_matrix.npy", weekday_matrix)
np.save(OUTPUT_DIR / "holiday_matrix.npy", holiday_matrix)
pd.Series(valid_zones, name="LocationID").to_csv(
    OUTPUT_DIR / "zone_order.csv", index_label="matrix_row")

print(f"\nMatrices guardadas en {OUTPUT_DIR}/")


# ----------------------------------------------------------------
# 5. Co-clustering BBAC_I con cgc, usando los K del paper
# ----------------------------------------------------------------
def true_i_divergence_constant(Z: np.ndarray) -> float:
    """
    cgc solo reporta la parte de la I-divergencia que depende de la
    partición (Ô - O*log(Ô)). Le falta la constante O*log(O) - O, que
    depende únicamente de la matriz de datos original (no del K_R/K_T
    ni del resultado del clustering). Esta función calcula esa
    constante para poder recuperar D_I(O || Ô) completa:

        D_I(O || Ô) = constante + cc.error

    Usa la convención estándar 0*log(0) = 0 (límite), así que las
    celdas con valor 0 no aportan nada.
    """
    nonzero = Z > 0
    return float(np.sum(Z[nonzero] * np.log(Z[nonzero]) - Z[nonzero]))


def run_coclustering(matrix: np.ndarray, k_row: int, k_col: int,
                      label: str, nruns: int = 200) -> object:
    print(f"\n--- Co-clustering {label} (K_R={k_row}, K_T={k_col}, "
          f"nruns={nruns}) ---")
    cc = Coclustering(
        matrix,
        nclusters_row=k_row,
        nclusters_col=k_col,
        max_iterations=100,
        conv_threshold=1e-5,
        nruns=nruns,  # el paper usa 200 inicializaciones aleatorias
    )
    results = cc.run_with_threads(nthreads=4)
    constant = true_i_divergence_constant(matrix)
    true_divergence = constant + results.error
    print(f"Error interno de cgc (parcial, puede ser negativo): "
          f"{results.error:.4f}")
    print(f"I-divergencia real D_I(O || Ô): {true_divergence:.4f}")
    print(f"Runs completados: {results.nruns_completed}, "
          f"convergidos: {results.nruns_converged}")
    return results


weekday_results = run_coclustering(
    weekday_matrix, **K_CONFIG["weekday"], label="WEEKDAY")
holiday_results = run_coclustering(
    holiday_matrix, **K_CONFIG["holiday"], label="HOLIDAY")

# Guardamos las asignaciones de cluster
np.save(OUTPUT_DIR / "weekday_row_clusters.npy", weekday_results.row_clusters)
np.save(OUTPUT_DIR / "weekday_col_clusters.npy", weekday_results.col_clusters)
np.save(OUTPUT_DIR / "holiday_row_clusters.npy", holiday_results.row_clusters)
np.save(OUTPUT_DIR / "holiday_col_clusters.npy", holiday_results.col_clusters)

print("\n✅ Listo. Matrices + resultados de co-clustering guardados en "
      f"'{OUTPUT_DIR}/'.")
print("\nSiguiente paso natural: barrido de K_R/K_T con silhouette/CH/DBI "
      "para reproducir la Figura 3 del paper y validar que K=8/6 es "
      "realmente el óptimo con tus datos.")