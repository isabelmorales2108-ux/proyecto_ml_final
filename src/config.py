"""
Configuración central del proyecto.

Reúne en un solo lugar las rutas, las semillas y los parámetros extraídos
del artículo de referencia (Imtiaz et al., 2025). Las marcas de origen
siguen la ficha de extracción (references/ficha_extraccion.md):
[R] reportado, [F] estimado de figura, [D] derivado por cálculo, [S] supuesto.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# 1. Rutas del proyecto
# ---------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "results"
DIR_FIGURAS = RAIZ / "figures"

# ---------------------------------------------------------------------
# 2. Reproducibilidad
# ---------------------------------------------------------------------
SEMILLA = 42             # [S] semilla base de todo el proyecto
N_REPLICAS = 30          # [S] repeticiones con semillas distintas (hallazgo 11)

# ---------------------------------------------------------------------
# 3. Diseño del estudio
# ---------------------------------------------------------------------
PARCELAS = ["A", "B", "C", "D"]
SENSORES = ["Sentinel-2", "PlanetScope"]
MODELOS = ["CART", "RF", "GTB"]
N_PUNTOS = 300           # [R] puntos por parcela
PROP_PRUEBA = 0.30       # [R] partición 70/30 dentro de cada parcela

VARIABLE_RESPUESTA = "rendimiento"               # t/ha
BANDAS = ["azul", "verde", "rojo", "nir"]        # [R]
INDICES = ["ndvi", "gndvi", "evi", "savi"]       # [R]
PREDICTORES = BANDAS + INDICES                   # 8 predictores (hallazgo 6)
SAVI_L = 0.5                                     # [R] factor de ajuste del suelo

# ---------------------------------------------------------------------
# 4. Rendimiento por parcela (t/ha)
#    media, minimo, maximo: [F] Figuras 6, 7 y 10
#    desv: [D] Var(y) = RMSE^2 / (1 - R^2), Tablas 4 y 5
#    asimetria: [S] la parcela A presenta cola larga hacia valores altos
# ---------------------------------------------------------------------
PARCELAS_INFO = {
    "A": {"anio": 2021, "metodo": "cosechadora", "media": 46.0, "desv": 10.0,
          "minimo": 20.0, "maximo": 105.0, "asimetria": 4.0},
    "B": {"anio": 2021, "metodo": "excavacion", "media": 88.0, "desv": 6.4,
          "minimo": 70.0, "maximo": 105.0, "asimetria": 0.0},
    "C": {"anio": 2022, "metodo": "cosechadora", "media": 47.0, "desv": 5.0,
          "minimo": 35.0, "maximo": 57.0, "asimetria": 0.0},
    "D": {"anio": 2022, "metodo": "excavacion", "media": 92.0, "desv": 6.9,
          "minimo": 75.0, "maximo": 110.0, "asimetria": 0.0},
}

# ---------------------------------------------------------------------
# 5. Correlación de Pearson índice-rendimiento (Figura 4) [F]
#    Calculada probablemente con las cuatro parcelas juntas (hallazgo 2)
# ---------------------------------------------------------------------
CORR_INDICE_RENDIMIENTO = {
    "Sentinel-2":  {"evi": 0.89, "gndvi": 0.88, "ndvi": 0.90, "savi": 0.88},
    "PlanetScope": {"evi": 0.83, "gndvi": 0.76, "ndvi": 0.85, "savi": 0.86},
}

# ---------------------------------------------------------------------
# 6. Hiperparámetros por sensor y parcela (Tabla 3) [R]
#    Equivalencias en scikit-learn:
#    CART maxNodes   -> DecisionTreeRegressor(max_leaf_nodes)
#    RF nTrees       -> RandomForestRegressor(n_estimators)
#    RF minLeafPop   -> RandomForestRegressor(min_samples_leaf)
#    GTB nTrees      -> GradientBoostingRegressor(n_estimators)
#    GTB shrinkage   -> GradientBoostingRegressor(learning_rate)
# ---------------------------------------------------------------------
def _hp(cart_nodos, rf_arboles, rf_hoja, gtb_arboles, gtb_tasa):
    """Arma el diccionario de hiperparámetros de una combinación sensor-parcela."""
    return {
        "CART": {"max_leaf_nodes": cart_nodos},
        "RF": {"n_estimators": rf_arboles, "min_samples_leaf": rf_hoja},
        "GTB": {"n_estimators": gtb_arboles, "learning_rate": gtb_tasa},
    }

HIPERPARAMETROS = {
    "Sentinel-2": {
        "A": _hp(100, 150, 3, 100, 0.10),
        "B": _hp(100, 200, 1, 100, 0.10),
        "C": _hp(150, 100, 5, 150, 0.05),
        "D": _hp(100, 200, 1, 100, 0.10),
    },
    "PlanetScope": {
        "A": _hp(100, 150, 3, 100, 0.05),
        "B": _hp(100, 200, 3, 100, 0.10),
        "C": _hp(120, 150, 7, 150, 0.07),
        "D": _hp(100, 200, 1, 100, 0.10),
    },
}

# ---------------------------------------------------------------------
# 7. Métricas de prueba reportadas (Tablas 4 y 5) [R]
#    Formato: (RMSE t/ha, MAE t/ha, R2)
# ---------------------------------------------------------------------
METRICAS_ARTICULO = {
    "Sentinel-2": {
        "A": {"CART": (4.57, 3.47, 0.76), "RF": (5.79, 4.07, 0.72), "GTB": (5.96, 4.20, 0.78)},
        "B": {"CART": (4.20, 2.78, 0.60), "RF": (3.73, 2.84, 0.65), "GTB": (3.82, 2.67, 0.65)},
        "C": {"CART": (2.92, 2.50, 0.62), "RF": (2.94, 2.47, 0.63), "GTB": (2.82, 2.33, 0.71)},
        "D": {"CART": (3.87, 3.04, 0.69), "RF": (3.27, 2.83, 0.61), "GTB": (3.67, 2.95, 0.62)},
    },
    "PlanetScope": {
        "A": {"CART": (4.23, 3.26, 0.77), "RF": (5.74, 4.54, 0.68), "GTB": (4.48, 3.73, 0.72)},
        "B": {"CART": (4.11, 3.42, 0.60), "RF": (3.89, 3.26, 0.61), "GTB": (3.48, 2.74, 0.68)},
        "C": {"CART": (3.22, 2.47, 0.63), "RF": (3.07, 2.60, 0.61), "GTB": (3.29, 2.58, 0.62)},
        "D": {"CART": (4.86, 3.41, 0.69), "RF": (4.44, 3.45, 0.62), "GTB": (4.25, 3.38, 0.67)},
    },
}
# ---------------------------------------------------------------------
# 8. Modelo generador: mezcla lineal de follaje y suelo [S]
#    Reflectancia de la banda = cobertura * follaje + (1 - cobertura) * suelo
#    Valores típicos de la literatura, validados con criterio agronómico
#    para papa sobre suelo de la serie Charlottetown (Isla del Príncipe Eduardo).
# ---------------------------------------------------------------------
REFLECTANCIA_FOLLAJE = {"azul": 0.02, "verde": 0.06, "rojo": 0.03, "nir": 0.45}
REFLECTANCIA_SUELO_HUMEDO = {"azul": 0.06, "verde": 0.09, "rojo": 0.14, "nir": 0.22}
REFLECTANCIA_SUELO_SECO = {"azul": 0.08, "verde": 0.12, "rojo": 0.18, "nir": 0.27}

# Cobertura promedio de temporada (junio-septiembre) en función del
# rendimiento. Curva logística con saturación moderada: el compuesto
# estacional promedia fases de cobertura parcial y dosel cerrado.
COBERTURA = {
    "minima": 0.20,
    "maxima": 0.95,
    "rendimiento_medio": 50.0,   # t/ha en el punto de inflexión
    "pendiente": 0.053,          # calibrada: B y D entre 0,85 y 0,88
}
# ---------------------------------------------------------------------
# 9. Estructura espacial [S]
#    Necesaria para diagnosticar y corregir la fuga de información
#    espacial (hallazgo 5).
# ---------------------------------------------------------------------
LOTE_LADO_M = 390.0                  # lote cuadrado de ~15,2 ha [R: 15-16 ha]

AUTOCORRELACION = {
    "modelo": "exponencial",         # correlación = (1 - pepita) * exp(-3 * d / alcance)
    "alcance_m": 40.0,               # [S] manchas de rendimiento de 30 a 50 m (criterio agronómico)
    "pepita": 0.20,                  # [S] variación local sin estructura espacial
}
ALCANCES_SENSIBILIDAD_M = [30.0, 40.0, 50.0]   # análisis de sensibilidad del alcance

N_MEDICIONES_EXCAVACION = 30         # [R] puntos medidos a mano en B y D
IDW_POTENCIA = 2.0                   # [S] potencia habitual de IDW; no reportada

# Validación espacial
N_BLOQUES_LADO = 4                   # grilla de 4 x 4 bloques de 97,5 m (> alcance)
DISTANCIA_EXCLUSION_M = 40.0         # zona de exclusión igual al alcance

# ---------------------------------------------------------------------
# 10. Fuentes de ruido [S] (valores iniciales; se calibran con las
#     métricas de prueba de las Tablas 4 y 5)
#
#     Ambas estrategias: desacople dosel-rendimiento y ruido del sensor.
#     Solo multivariada: brillo del suelo e iluminación, compartidos
#     entre las cuatro bandas.
# ---------------------------------------------------------------------
RUIDO = {
    # Fuente 1. Parte del rendimiento que no se expresa en el follaje
    # (nematodos, Rhizoctonia, compactación). En desviaciones estándar de
    # la parcela; con manchas del mismo alcance que el rendimiento.
    "desacople": 0.30,

    # Fuente 2. Error radiométrico relativo de cada banda, independiente
    # por punto y por banda.
    "sensor": {"Sentinel-2": 0.025, "PlanetScope": 0.040},

    # Fuente 3. Brillo del suelo entre húmedo y seco, con estructura
    # espacial (alcance de AUTOCORRELACION). Sin parámetro adicional:
    # usa REFLECTANCIA_SUELO_HUMEDO y REFLECTANCIA_SUELO_SECO.

    # Fuente 4. Variación relativa de iluminación o calibración que
    # escala las cuatro bandas en la misma proporción.
    "iluminacion": 0.03,
}

ESTRATEGIAS = ["univariada", "multivariada"]


# ---------------------------------------------------------------------
# 11. Hiperparámetros no reportados en la Tabla 3 [S]
#     Valores por defecto de Google Earth Engine (verificados en la
#     documentación oficial de smileCart, smileRandomForest y
#     smileGradientTreeBoost) y su equivalente en scikit-learn.
#     Se combinan con HIPERPARAMETROS (bloque 6) al construir cada modelo.
# ---------------------------------------------------------------------
HIPERPARAMETROS_FIJOS = {
    "CART": {
        "min_samples_leaf": 1,           # minLeafPopulation = 1
    },
    "RF": {
        "max_features": "sqrt",          # variablesPerSplit = raíz cuadrada de 8 -> 2
        "bootstrap": True,
        "max_samples": 0.5,              # bagFraction = 0,5
        "max_leaf_nodes": None,          # maxNodes = sin límite
    },
    "GTB": {
        "subsample": 0.7,                # samplingRate = 0,7
        "max_depth": None,               # maxNodes = sin límite (scikit-learn usa 3 por defecto)
        "max_leaf_nodes": None,
        "loss": "absolute_error",        # loss = LeastAbsoluteDeviation
    },
}
