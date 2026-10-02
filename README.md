# PFC2 — AST-CC

Proyecto de fin de carrera (UCSP): refinamiento jerárquico de ST-CC guiado por
pruebas de hipótesis sobre matrices espacio-temporales de conteos.

| Carpeta | Contenido |
|---|---|
| `Latex/` | Plan de tesis (`Tesis.tex`). Compilar con `latexmk -pdf Tesis.tex`. |
| `Implementaciones/` | Código, experimentos y resultados (ver su `README.md`). |

## Puesta en marcha en otra máquina

```bash
git clone <url-del-repo> PFC2
cd PFC2/Implementaciones
conda env create -f environment.yml
conda activate stcc
```

No se versionan:

- `Implementaciones/DATA/crudos/`: datos de la TLC (noviembre de 2017), se
  descargan según `Implementaciones/README.md`.
- `Latex/Papers/`: PDFs de terceros.
- El entorno de conda: se recrea con `environment.yml`.
