# Ficha de extracción: Imtiaz et al. (2025)

Imtiaz, F., Farooque, A. A., Randhawa, G. S., Wang, X., Esau, T. J., Hashemi Garmdareh, S. E. y Acharya, B. (2025). Optimizing potato yield mapping and prediction: Integrating satellite-based remote sensing and machine learning for sustainable agriculture. *Computers and Electronics in Agriculture, 237*, 110636. https://doi.org/10.1016/j.compag.2025.110636

## Convenciones de origen

| Marca | Significado |
|---|---|
| **[R]** | Reportado explícitamente en el texto o en una tabla |
| **[F]** | Estimado a partir de una figura (lectura visual, con incertidumbre) |
| **[D]** | Derivado por cálculo a partir de resultados reportados |
| **[NR]** | No reportado; requiere un supuesto |

## 1. Objetivos y tarea

- Comparar Sentinel-2A y PlanetScope para predecir y mapear el rendimiento de papa a escala de campo. [R]
- Evaluar tres modelos de aprendizaje de máquina con datos satelitales multiespectrales. [R]
- Comparar dos métodos de medición del rendimiento: cosechadora y excavación manual. [R]
- Tarea: regresión del rendimiento en t/ha. [R]

## 2. Área de estudio y diseño

| Elemento | Contenido | Origen |
|---|---|---|
| Región | Isla del Príncipe Eduardo (Canadá), agricultura de secano, ~890 mm de precipitación anual | [R] |
| Parcelas | A y B (2021), C y D (2022), de 15–16 ha cada una | [R] |
| Ciclo | Siembra a finales de mayo, cosecha a inicios de octubre | [R] |
| Medición del rendimiento | A y C: cosechadora con GPS y monitor de rendimiento. B y D: excavación manual de una franja de 10 pies en 30 puntos (3 zonas × 10 puntos) | [R] |
| Variedad, fertilización, distancia entre surcos | No informados | [NR] |

## 3. Datos satelitales

| Elemento | Sentinel-2 | PlanetScope | Origen |
|---|---|---|---|
| Producto | Nivel 2A armonizado (`COPERNICUS/S2_SR_HARMONIZED`), con corrección atmosférica | Imágenes de 4 bandas, instrumento PSB.SD | [R] |
| Resolución espacial | 10 m | 3 m | [R] |
| Imágenes (junio–septiembre) | 5 en 2021, 7 en 2022 (nubosidad < 10 %) | 17 en 2021, 19 en 2022 | [R] |
| Tratamiento de nubes | Máscara con la banda QA60 | Solo escenas sin nubes | [R] |
| Agregación temporal | Un compuesto por la media de la temporada | Igual | [R] |

## 4. Construcción de la base tabular

| Paso | Contenido | Origen |
|---|---|---|
| Limpieza (A y C) | Eliminación de rendimientos cero y de atípicos por umbrales del cultivo y rango intercuartílico | [R]; umbrales exactos [NR] |
| Interpolación | Distancia inversa ponderada (IDW) a grillas de 3 m y 10 m | [R]; parámetros [NR] |
| Muestreo | 300 puntos aleatorios por parcela | [R] |
| Tamaño total | 4 parcelas × 300 puntos = 1.200 registros por sensor | [R] (derivado) |
| Partición | 70/30 aleatoria dentro de cada parcela (210 entrenamiento / 90 prueba) | [R]; semilla [NR] |

## 5. Variables

- **Respuesta:** rendimiento de papa (t/ha). [R]
- **Bandas:** azul, verde, rojo e infrarrojo cercano (NIR). [R]
- **Índices:** NDVI, GNDVI, EVI y SAVI (L = 0,5), con fórmulas explícitas (ecuaciones 1–4). [R]
- **Total de predictores asumido:** 8. La sección 2.5 omite EVI y repite NIR; la discusión y la conclusión confirman 4 bandas y 4 índices.

## 6. Correlaciones de Pearson (Figura 4) [F]

| Índice vs. rendimiento | Sentinel-2 | PlanetScope |
|---|---|---|
| EVI | 0,89 | 0,83 |
| GNDVI | 0,88 | 0,76 |
| NDVI | 0,90 | 0,85 |
| SAVI | 0,88 | 0,86 |

| Entre índices | Sentinel-2 | PlanetScope |
|---|---|---|
| GNDVI–EVI | 0,80 | 0,66 |
| NDVI–EVI | 0,79 | 0,68 |
| NDVI–GNDVI | 0,78 | 0,63 |
| SAVI–EVI | 0,80 | 0,71 |
| SAVI–GNDVI | 0,77 | 0,65 |
| SAVI–NDVI | 0,80 | 0,73 |

Correlaciones de las bandas y nivel de cálculo (por parcela o con las cuatro parcelas juntas): [NR].

## 7. Rendimiento por parcela

| Parcela | Método | Rango observado [F] | Media aprox. [F] | Desv. estándar implícita [D] |
|---|---|---|---|---|
| A | Cosechadora | ~20–105 | ~46 | 8,5–12,7 (inconsistente entre modelos) |
| B | Excavación | ~70–105 | ~88 | 6,2–6,6 |
| C | Cosechadora | ~35–57 | ~47 | 4,7–5,3 |
| D | Excavación | ~75–110 | ~92 | 5,2–8,7 |

La desviación estándar implícita se obtiene de Var(y) = RMSE² / (1 − R²) con los resultados de prueba de las Tablas 4 y 5. Las medias son lecturas visuales con incertidumbre aproximada de ±3 t/ha.

## 8. Modelos e hiperparámetros (Tabla 3)

| Modelo | Reportado | Equivalente en scikit-learn | No reportado |
|---|---|---|---|
| CART | maxNodes 100–150 | `DecisionTreeRegressor(max_leaf_nodes=...)` | Profundidad, mínimo por hoja |
| RF | nTrees 100–200; minLeafPop 1–7 | `RandomForestRegressor(n_estimators=..., min_samples_leaf=...)` | Variables por división, fracción de muestreo |
| GTB | nTrees 100–150; shrinkage 0,05–0,1 | `GradientBoostingRegressor(n_estimators=..., learning_rate=...)` | Función de pérdida, submuestreo, nodos máximos |

Ajuste manual por ensayo y error en Google Earth Engine, que no tiene herramientas automáticas de optimización. [R]

## 9. Evaluación

- Métricas: R² (ecuación 5), RMSE y MAE (ecuaciones 6 y 7). [R]
- Resultados de entrenamiento y prueba por parcela y modelo (Tablas 4 y 5). [R]
- Bootstrap con 1.000 iteraciones y evaluación fuera de bolsa, con intervalos de confianza del 95 % (Tabla 6). [R]; sensor usado en la Tabla 6 [NR]
- Modelo con mejor desempeño según los autores: GTB, seguido de CART y RF. [R]

## 10. Hallazgos de la lectura crítica

1. **R² y RMSE incompatibles en la parcela A (Sentinel-2).** CART tiene menor RMSE y menor R² que GTB, lo que es imposible con la ecuación 5 sobre un mismo conjunto de prueba. Explicaciones posibles: particiones distintas por modelo, o un R² reportado como cuadrado de la correlación entre observado y predicho. En B y C las varianzas implícitas son coherentes, así que el problema se concentra en la parcela A.
2. **Correlaciones globales altas frente al desempeño de los modelos.** Una correlación de 0,90 implicaría R² ≈ 0,81 con NDVI solo, más que cualquier modelo reportado. Con las medias por parcela, cerca del 89 % de la varianza total del rendimiento corresponde a diferencias entre parcelas, lo que infla la correlación si la matriz se calculó con las cuatro parcelas juntas.
3. **Correlación NDVI–SAVI baja (0,80).** Con L = 0,5, ambos índices calculados sobre las mismas bandas suelen correlacionar cerca de 1.
4. **Leyendas de las Figuras 6 y 7 invertidas.** Los valores de la Figura 6 corresponden a la Tabla 5 (PlanetScope) y los de la Figura 7 a la Tabla 4 (Sentinel-2). Las figuras anotan n = 300, pero los puntos graficados corresponden al conjunto de prueba.
5. **Tamaño efectivo de muestra menor que 300.** En B y D, los 300 puntos provienen de una superficie interpolada a partir de 30 mediciones. La partición aleatoria mezcla puntos vecinos casi idénticos entre entrenamiento y prueba (posible fuga de información; los autores reconocen la autocorrelación espacial como limitación).
6. **Lista de predictores ambigua** (sección 2.5): omite EVI y repite NIR.
7. **Rendimientos de 70–110 t/ha en B y D**, frente al promedio canadiense de ~41,3 t/ha citado por los autores. Explicaciones compatibles con el diseño: valores puntuales y no promedios de campo, rendimiento bruto en la excavación manual, conversión de franja a hectárea con distancia entre surcos no reportada. La parcela A, medida con cosechadora, también alcanza ~100 t/ha. No se trata como anomalía.
8. **Fórmula del EVI** (ecuación 3): usa 7 × Azul, mientras que la formulación estándar usa 7,5 × Azul.
9. **Conclusión sobre sensores no respaldada por la Tabla 7.** Sentinel-2 gana en todas las métricas solo en la parcela C; PlanetScope gana en todas en la parcela B; en A y D las métricas se contradicen. Los conjuntos de prueba de ambos sensores no son idénticos, porque el rendimiento se interpoló en grillas distintas.
10. **Confusión en el diseño experimental.** El método de medición coincide con el nivel de rendimiento (cosechadora: A y C, ~46 t/ha; excavación: B y D, ~90 t/ha), con una parcela por combinación. No es posible separar el efecto del método del efecto de la parcela.
11. **Una sola partición por combinación**, sin réplicas con distintas semillas, lo que impide estimar la variabilidad de los resultados debida a la partición.

## 11. Decisiones de simulación

**Variable de respuesta:** rendimiento de papa (t/ha) por punto. **Predictores:** 4 bandas y 4 índices de vegetación. Parcela, año, método de medición y sensor son variables de diseño, no predictores.

1. **Simulación univariada (una contra una).** Se simula el rendimiento por parcela y cada banda se genera a partir de él con un ruido propio e independiente. Solo se especifica la relación de cada predictor con la respuesta. Análisis: correlación y dispersión de cada predictor frente al rendimiento, y modelos con un predictor a la vez.
2. **Simulación multivariada (todas contra todas).** El rendimiento y las bandas se generan conjuntamente, con ruidos compartidos entre bandas, de modo que se especifica la matriz completa de relaciones. Análisis: matriz de correlación completa, diagramas de pares y modelos con los 8 predictores.
3. **Ambos sensores.** Sentinel-2 y PlanetScope observan los mismos puntos y el mismo rendimiento, con un error de medición propio de cada sensor, y se evalúan sobre el mismo conjunto de prueba.
4. **Índices calculados.** Las bandas se simulan como reflectancias (escala 0–1) y los índices se calculan con sus fórmulas. EVI con la formulación estándar (7,5 × Azul); el artículo escribe 7. Las correlaciones entre índices pueden diferir de la Figura 4. Las correlaciones índice–rendimiento y los R² por parcela se calibran con los resultados reportados.