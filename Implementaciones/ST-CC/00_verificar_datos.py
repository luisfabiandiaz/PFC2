"""
Verificación rápida de los datos descargados para ST-CC.
Requiere: pip install duckdb geopandas --break-system-packages
(duckdb lee parquet sin cargarlo completo en RAM; geopandas necesita fiona/pyogrio)

Las rutas se calculan a partir de la ubicación de este archivo.
"""

import duckdb
import geopandas as gpd
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]      # Implementaciones/
DATA_DIR = RAIZ / "DATA/crudos"

# ---- 1. Yellow taxi parquet ----
print("=" * 60)
print("YELLOW TAXI")
print("=" * 60)
yellow_path = DATA_DIR / "yellow_tripdata_2017-11.parquet"
con = duckdb.connect()
schema = con.execute(f"DESCRIBE SELECT * FROM '{yellow_path}'").fetchdf()
print(schema)
n_rows = con.execute(f"SELECT COUNT(*) FROM '{yellow_path}'").fetchone()[0]
print(f"\nFilas totales: {n_rows:,}")
sample = con.execute(f"SELECT * FROM '{yellow_path}' LIMIT 3").fetchdf()
print("\nMuestra:")
print(sample)

# ---- 2. Green taxi parquet ----
print("\n" + "=" * 60)
print("GREEN TAXI")
print("=" * 60)
green_path = DATA_DIR / "green_tripdata_2017-11.parquet"
schema_g = con.execute(f"DESCRIBE SELECT * FROM '{green_path}'").fetchdf()
print(schema_g)
n_rows_g = con.execute(f"SELECT COUNT(*) FROM '{green_path}'").fetchone()[0]
print(f"\nFilas totales: {n_rows_g:,}")

# ---- 3. Taxi zone lookup CSV ----
print("\n" + "=" * 60)
print("TAXI ZONE LOOKUP")
print("=" * 60)
zones_csv = DATA_DIR / "taxi_zone_lookup.csv"
zones_df = con.execute(f"SELECT * FROM '{zones_csv}'").fetchdf()
print(zones_df.head(10))
manhattan_zones = zones_df[zones_df["Borough"] == "Manhattan"]
print(f"\nZonas en Manhattan (antes de excluir islas): {len(manhattan_zones)}")
islas = manhattan_zones[manhattan_zones["Zone"].str.contains(
    "Governor|Ellis|Liberty", case=False, na=False)]
print("Zonas-isla detectadas (deben excluirse, según Liu et al.):")
print(islas[["LocationID", "Zone"]])
print(f"\nZonas finales esperadas (66 según el paper): "
      f"{len(manhattan_zones) - len(islas)}")

# ---- 4. Shapefile ----
print("\n" + "=" * 60)
print("SHAPEFILE DE ZONAS")
print("=" * 60)
shp_path = DATA_DIR / "taxi_zones" / "taxi_zones.shp"
gdf = gpd.read_file(shp_path)
print(gdf.head())
print(f"\nCRS: {gdf.crs}")
print(f"Total de polígonos: {len(gdf)}")

print("\n✅ Si no hubo errores arriba, los 4 archivos están sanos y listos.")