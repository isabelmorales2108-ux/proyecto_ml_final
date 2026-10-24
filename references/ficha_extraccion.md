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

**Valores adoptados en la simulación.** Media y límites tomados de la figura [F]; desviación estándar elegida dentro del rango implícito [D]: A = 10 (centro del rango 8,5–12,7, inconsistente entre modelos), B = 6,4, C = 5,0 y D = 6,9 (centro del rango 5,2–8,7). En A se agregó asimetría positiva (parámetro 4), porque la Figura 10 muestra una cola hacia rendimientos altos.

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

1. **R² y RMSE incompatibles entre modelos.** Con la ecuación 5 sobre un mismo conjunto de prueba, la varianza implícita Var(y) = RMSE² / (1 − R²) debe ser idéntica para los tres modelos. No lo es en ninguna de las ocho combinaciones de parcela y sensor (sección 7). El caso más visible es la parcela A con Sentinel-2, donde CART tiene menor RMSE y menor R² que GTB. Explicaciones posibles: particiones distintas por modelo, o un R² calculado como cuadrado de la correlación entre observado y predicho. *Verificado en el notebook 02, sección 4.*
2. **Correlaciones globales altas frente al desempeño de los modelos.** Una correlación de 0,90 implicaría R² ≈ 0,81 con NDVI solo, más que cualquier modelo reportado. Con las medias por parcela, cerca del 89 % de la varianza total del rendimiento corresponde a diferencias entre parcelas, lo que infla la correlación si la matriz se calculó con las cuatro parcelas juntas. *Matiz tras la simulación:* con los parámetros calibrados, las correlaciones índice–rendimiento dentro de cada parcela alcanzan valores del orden de la Figura 4. La figura es compatible tanto con un cálculo dentro de cada parcela como con uno de las cuatro parcelas juntas; el nivel de cálculo sigue sin reportarse. *Notebook 01, sección 4.*
3. **Correlación NDVI–SAVI baja (0,80).** Con L = 0,5, ambos índices calculados sobre las mismas bandas suelen correlacionar cerca de 1. *Matiz tras la simulación:* con ruido de iluminación y brillo del suelo, la correlación NDVI–SAVI dentro de una parcela baja a ≈ 0,88. Un valor de 0,80 es poco probable, pero no imposible, si hay fuentes de variación que afectan de forma distinta a ambos índices. *Notebook 01, sección 5.*
4. **Leyendas de las Figuras 6 y 7 invertidas.** Los valores de la Figura 6 corresponden a la Tabla 5 (PlanetScope) y los de la Figura 7 a la Tabla 4 (Sentinel-2). Las figuras anotan n = 300, pero los puntos graficados corresponden al conjunto de prueba.
5. **Tamaño efectivo de muestra menor que 300.** En B y D, los 300 puntos provienen de una superficie interpolada a partir de 30 mediciones. La partición aleatoria mezcla puntos vecinos casi idénticos entre entrenamiento y prueba (posible fuga de información; los autores reconocen la autocorrelación espacial como limitación). *Cuantificación:* con la respuesta interpolada, la validación aleatoria infla el R² en ~0,10 frente a la validación por bloques. Con mediciones directas, la diferencia es despreciable. *Notebook 03, sección 2.*
6. **Lista de predictores ambigua** (sección 2.5): omite EVI y repite NIR.
7. **Rendimientos de 70–110 t/ha en B y D**, frente al promedio canadiense de ~41,3 t/ha citado por los autores. Explicaciones compatibles con el diseño: valores puntuales y no promedios de campo, rendimiento bruto en la excavación manual, conversión de franja a hectárea con distancia entre surcos no reportada. La parcela A, medida con cosechadora, también alcanza ~100 t/ha. No se trata como anomalía.
8. **Fórmula del EVI** (ecuación 3): usa 7 × Azul, mientras que la formulación estándar usa 7,5 × Azul.
9. **Conclusión sobre sensores no respaldada por la Tabla 7.** Sentinel-2 gana en todas las métricas solo en la parcela C; PlanetScope gana en todas en la parcela B; en A y D las métricas se contradicen. Los conjuntos de prueba de ambos sensores no son idénticos, porque el rendimiento se interpoló en grillas distintas. *Cuantificación:* entre 30 réplicas, el R² de una misma combinación varía hasta ±0,16 según la partición, por lo que las diferencias entre sensores de la Tabla 7 son compatibles con el azar de una sola partición. *Notebook 02, sección 6.*
10. **Confusión en el diseño experimental.** El método de medición coincide con el nivel de rendimiento (cosechadora: A y C, ~46 t/ha; excavación: B y D, ~90 t/ha), con una parcela por combinación. No es posible separar el efecto del método del efecto de la parcela.
11. **Una sola partición por combinación**, sin réplicas con distintas semillas, lo que impide estimar la variabilidad de los resultados debida a la partición. *Cuantificación:* la variación entre réplicas (hasta ±0,16 de R²) es del mismo orden que las diferencias entre modelos y sensores que el artículo interpreta. *Notebook 02.*
12. **El procedimiento declarado no reproduce los R² de B y D.** Con la respuesta interpolada por IDW desde 30 mediciones, los tres modelos obtienen R² de prueba cercanos a cero o negativos. Los R² del artículo (0,60–0,69) solo se alcanzan con mediciones directas en cada punto. El resultado se mantiene con manchas de rendimiento de hasta 150 m (notebook 03, sección 5). Hipótesis restante, no simulada: que los predictores tengan una suavidad espacial (por remuestreo o compuesto temporal) parecida a la de la superficie interpolada. *Notebook 02, sección 7.*
13. **R² de entrenamiento no reproducibles.** Con los hiperparámetros de la Tabla 3, los R² de entrenamiento por resustitución son sistemáticamente mayores que los reportados. Esto sugiere que los valores de entrenamiento se calcularon de otra forma (por ejemplo, fuera de bolsa) o con otra configuración no reportada. *Notebook 02, sección 5.*

## 11. Decisiones de simulación

**Variable de respuesta:** rendimiento de papa (t/ha) por punto. **Predictores:** 4 bandas y 4 índices de vegetación. Parcela, año, método de medición y sensor son variables de diseño, no predictores.

1. **Simulación univariada (una contra una).** Se simula el rendimiento por parcela y cada banda se genera a partir de él con un ruido propio e independiente. Solo se especifica la relación de cada predictor con la respuesta. Análisis: correlación y dispersión de cada predictor frente al rendimiento, y modelos con un predictor a la vez.
2. **Simulación multivariada (todas contra todas).** El rendimiento y las bandas se generan conjuntamente, con ruidos compartidos entre bandas, de modo que se especifica la matriz completa de relaciones. Análisis: matriz de correlación completa, diagramas de pares y modelos con los 8 predictores.
3. **Ambos sensores.** Sentinel-2 y PlanetScope observan los mismos puntos y el mismo rendimiento, con un error de medición propio de cada sensor, y se evalúan sobre el mismo conjunto de prueba.
4. **Índices calculados.** Las bandas se simulan como reflectancias (escala 0–1) y los índices se calculan con sus fórmulas. EVI con la formulación estándar (7,5 × Azul); el artículo escribe 7. Las correlaciones entre índices pueden diferir de la Figura 4. Las correlaciones índice–rendimiento y los R² por parcela se calibran con los resultados reportados.
5. **Estructura espacial.** Coordenadas en un lote de 390 m; campo gaussiano con correlación exponencial (alcance de 40 m, pepita de 0,20), transformado por cuantiles a la distribución de rendimiento de cada parcela. Sensibilidad con otros alcances (notebook 03).
6. **Excavación e interpolación.** 30 mediciones (3 franjas × grilla de 5 × 2, separadas 65 m) e IDW con potencia 2 en B y D. Se conserva el rendimiento real de cada punto para medir el efecto de la interpolación.
7. **Calibración del ruido.** Desacople dosel–rendimiento, ruido del sensor, brillo del suelo e iluminación, calibrados con A y C frente a las Tablas 4 y 5 (diferencia media de R² de 0,038).
8. **Hiperparámetros no reportados.** Valores por defecto de Google Earth Engine, verificados en su documentación.
9. **Línea base.** Regresión Ridge con estandarización, que no está en el artículo, para juzgar si la complejidad de los árboles se justifica.

## 12. Limitaciones de la recreación

- Los predictores no tienen suavidad espacial propia, lo que podría explicar el hallazgo 12.
- A y C comparten una misma curva de cobertura, aunque corresponden a años distintos.
- La zona de exclusión de 40 m es insuficiente para el alcance de 150 m.
- La extensión se acotó a Sentinel-2, estrategia multivariada y 5–10 réplicas por costo de cómputo.
- Las medias y los rangos leídos de figuras tienen una incertidumbre de ±3 t/ha.