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
│       ├── ST-CC/             Matrices 66x12 y etiquetas de clusters (.npy)
│       └── Calibracion/       Conteos diarios D x 66 x 12 por periodo (.npy)
├── ST-CC/                     Reproducción de Liu et al. (2021)
│   ├── 00_verificar_datos.py
│   ├── 01_matriz_y_coclustering.py
│   ├── 02_barrido_K.py
│   ├── 03_visualizacion.py
│   └── 04_refinamiento_kmeans.py
├── astcc/                     Biblioteca compartida: estadísticos, modelos nulos, cortes
│   ├── estadisticos.py
│   ├── nulas.py
│   └── cortes.py
├── tests/                     Pruebas de astcc (python -m pytest tests)
├── Verificacion_teorica/      Comprobación numérica del marco teórico y la propuesta
│   └── identidades.py
├── Calibracion/               Supuestos estadísticos de las pruebas (sobredispersión, H0)
│   ├── 01_sobredispersion.py       φ̂ y función de varianza con las réplicas diarias
│   ├── 02_exactitud_omnibus.py     E1: nivel de Patefield y de la χ² bajo Poisson
│   ├── 03_seleccion_minP.py        E2: mejor corte, ingenuo frente a min-P
│   ├── 04_nula_semirreal.py        E3: tipo I con la variabilidad real de los días
│   └── 05_bloques_base_BBAC.py     E4: efecto de elegir los bloques con BBAC
└── Resultados/                Todo lo que generan los scripts (figuras, tablas)
    ├── ST-CC/figuras/
    ├── Verificacion_teorica/
    └── Calibracion/
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
python Calibracion/01_sobredispersion.py
python Calibracion/02_exactitud_omnibus.py
python Calibracion/03_seleccion_minP.py
python Calibracion/04_nula_semirreal.py
python Calibracion/05_bloques_base_BBAC.py
python -m pytest tests
```

En `ST-CC`, `02` a `04` leen las matrices que guarda `01`. En `Calibracion`, `01`
guarda los conteos diarios que usan los demás. Los scripts de `Calibracion`
importan `astcc` añadiendo la raíz del proyecto a `sys.path`, así que no hace
falta instalar nada.
