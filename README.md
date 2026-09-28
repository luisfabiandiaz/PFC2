# Implementaciones — PFC2

Código de la tesis. El documento LaTeX está en `../Latex`.

## Estructura

```
Implementaciones/
├── environment.yml            Entorno conda "stcc"
├── DATA/
│   ├── crudos/                Datos originales, no se modifican
│   │   ├── yellow_tripdata_2017-11.parquet
│   │   ├── green_tripdata_2017-11.parquet
│   │   ├── taxi_zone_lookup.csv
│   │   └── taxi_zones/        Shapefile de zonas TLC
│   └── procesados/
│       └── ST-CC/             Matrices 66x12 y etiquetas de clusters (.npy)
├── ST-CC/                     Reproducción de Liu et al. (2021)
│   ├── 00_verificar_datos.py
│   ├── 01_matriz_y_coclustering.py
│   ├── 02_barrido_K.py
│   ├── 03_visualizacion.py
│   └── 04_refinamiento_kmeans.py
├── Verificacion_teorica/      Comprobación numérica del marco teórico y la propuesta
│   └── identidades.py
└── Resultados/                Todo lo que generan los scripts (figuras, tablas)
    ├── ST-CC/figuras/
    └── Verificacion_teorica/
```

Convenciones:

- Los scripts calculan sus rutas a partir de su propia ubicación, así que
  funcionan desde cualquier directorio.
- `DATA/crudos` es de solo lectura; lo derivado va a `DATA/procesados/<módulo>`
  y las figuras o tablas a `Resultados/<módulo>`.
- Los scripts de un módulo se numeran en el orden en que se ejecutan.

## Datos crudos

`DATA/crudos/` no está en git (140 MB). Se descargan de la página de la NYC Taxi
and Limousine Commission (TLC Trip Record Data):

- `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2017-11.parquet`
- `https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2017-11.parquet`
- `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`
- `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip` (descomprimir en `taxi_zones/`)

## Uso

```
conda activate stcc
python ST-CC/01_matriz_y_coclustering.py
python ST-CC/02_barrido_K.py
python ST-CC/03_visualizacion.py
python ST-CC/04_refinamiento_kmeans.py
python Verificacion_teorica/identidades.py
```

`02` a `04` leen las matrices que guarda `01`, así que `01` debe ejecutarse primero.
