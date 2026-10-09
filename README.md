# Recreación con datos simulados: predicción del rendimiento de papa con Sentinel-2 y aprendizaje de máquina

**Trabajo final — Machine Learning II**
Maestría en Analítica Aplicada e Inteligencia Artificial, Universidad de La Sabana
Profesor: Jesús Antonio Villarraga Palomino
Autora: Isabel Cristina Morales Parra

---

## 1. Descripción

Este repositorio recrea, mediante datos simulados, el análisis de un artículo científico sobre predicción del rendimiento de papa con imágenes satelitales y aprendizaje de máquina. El propósito no es reproducir exactamente los resultados originales, sino aproximarse a su procedimiento, comparar los resultados obtenidos con los reportados, analizar las diferencias y evaluar, mediante una extensión experimental, cómo cambia el desempeño de los modelos al modificar una condición del experimento.

## 2. Artículo de referencia

Imtiaz, F., Farooque, A. A., Randhawa, G. S., Wang, X., Esau, T. J., Hashemi Garmdareh, S. E. y Acharya, B. (2025). Optimizing potato yield mapping and prediction: Integrating satellite-based remote sensing and machine learning for sustainable agriculture. *Computers and Electronics in Agriculture, 237*, 110636. https://doi.org/10.1016/j.compag.2025.110636

Una copia del artículo se encuentra en `references/`, distribuida bajo la licencia CC BY-NC-ND 4.0 de sus autores.

**Síntesis del estudio original:** cuatro parcelas de papa en la Isla del Príncipe Eduardo (Canadá), temporadas 2021 y 2022. Predictores: cuatro bandas espectrales (azul, verde, rojo e infrarrojo cercano) y cuatro índices de vegetación (NDVI, GNDVI, EVI y SAVI) obtenidos de Sentinel-2A y PlanetScope. Modelos: árbol de clasificación y regresión (CART), bosque aleatorio (RF) y Gradient Tree Boosting (GTB). Métricas: R², RMSE y MAE.

## 3. Relación con el trabajo de grado

El trabajo de grado de la autora busca predecir el rendimiento y la producción de papa en el corredor Bogotá–Tunja (Cundinamarca y Boyacá) a partir de imágenes satelitales e información histórica. Este ejercicio sirve como piloto metodológico de esa línea de investigación: emplea el mismo cultivo, el mismo sensor (Sentinel-2) y una formulación equivalente del problema como regresión.

## 4. Estructura del repositorio

```
proyecto_ml_final/
├── README.md                 # Descripción, supuestos e instrucciones
├── environment.yml           # Entorno reproducible (versiones fijadas)
├── data/                     # Datos simulados (CSV)
├── notebooks/                # Notebooks principales, en orden de ejecución
├── src/                      # Funciones de apoyo en Python
├── results/                  # Tablas de resultados y métricas
├── figures/                  # Gráficas generadas
└── references/               # Artículo y ficha de extracción
```

## 5. Instalación

Requisitos previos: [Git](https://git-scm.com/) y [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o Anaconda.

```bash
git clone https://github.com/isabelmorales2108-ux/proyecto_ml_final.git
cd proyecto_ml_final
conda env create -f environment.yml
conda activate ml2_papa
python -m ipykernel install --user --name ml2_papa --display-name "Python (ml2_papa)"
```

El último comando registra el entorno para que aparezca como opción de kernel en Jupyter y VS Code.

## 6. Orden de ejecución

Ejecutar los notebooks en orden, seleccionando el kernel **Python (ml2_papa)**:

| Notebook | Propósito | Produce |
|---|---|---|
| `01_simulacion_datos.ipynb` | Construcción de los datos simulados | Archivos en `data/` |
| `02_analisis_articulo.ipynb` | Recreación del análisis y comparación con el artículo | Tablas en `results/` y gráficas en `figures/` |
| `03_extension_experimental.ipynb` | Extensión experimental | Tablas en `results/` y gráficas en `figures/` |

*[Pendiente: se completará cuando los notebooks estén construidos.]*

## 7. Reproducibilidad

- Versiones de paquetes fijadas en `environment.yml`.
- Semillas aleatorias centralizadas en `src/config.py`.
- Los datos simulados pueden regenerarse ejecutando el notebook 01.

## 8. Supuestos de la simulación

Toda característica de los datos que el artículo no reporta y que es necesaria para la simulación se declara como supuesto.

*[Pendiente: se completará tras validar las decisiones de simulación con el profesor. La ficha de extracción en `references/` documenta qué información fue reportada, cuál se estimó a partir de figuras y cuál no está disponible.]*

## 9. Resultados principales

*[Pendiente.]*

## 10. Conclusiones y aprendizajes

*[Pendiente.]*

## Licencia y uso

Código desarrollado con fines académicos. El artículo de referencia es propiedad de sus autores y se distribuye bajo la licencia [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
