"""
Construcción de los modelos del artículo de referencia y de la línea base.

Combina los hiperparámetros reportados en la Tabla 3 (config.HIPERPARAMETROS)
con los valores por defecto de Google Earth Engine para los no reportados
(config.HIPERPARAMETROS_FIJOS).

Equivalencias Earth Engine -> scikit-learn:
    smileCart              -> DecisionTreeRegressor
    smileRandomForest      -> RandomForestRegressor
    smileGradientTreeBoost -> GradientBoostingRegressor

Línea base de la extensión (no está en el artículo):
    Ridge -> estandarización + regresión lineal con penalización L2,
             dentro de un Pipeline para que la estandarización se ajuste
             solo con los datos de entrenamiento de cada partición.
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config

CLASES_MODELO = {
    "CART": DecisionTreeRegressor,
    "RF": RandomForestRegressor,
    "GTB": GradientBoostingRegressor,
}


def obtener_hiperparametros(nombre, sensor, parcela):
    """
    Devuelve la configuración completa de un modelo para un sensor y una
    parcela: Tabla 3 [R] más los valores por defecto de Earth Engine [S].

    Parámetros
    ----------
    nombre : str
        "CART", "RF" o "GTB".
    sensor : str
        "Sentinel-2" o "PlanetScope".
    parcela : str
        "A", "B", "C" o "D".

    Retorna
    -------
    dict
        Hiperparámetros con los nombres de scikit-learn.
    """
    return {
        **config.HIPERPARAMETROS[sensor][parcela][nombre],
        **config.HIPERPARAMETROS_FIJOS[nombre],
    }


def construir_modelo(nombre, sensor, parcela, semilla, hiperparametros=None):
    """
    Construye un modelo de scikit-learn sin entrenar.

    Para "CART", "RF" y "GTB" usa la configuración del artículo. Para
    "Ridge" construye la línea base lineal de la extensión.

    Parámetros
    ----------
    nombre : str
        "CART", "RF", "GTB" o "Ridge".
    sensor : str
        "Sentinel-2" o "PlanetScope".
    parcela : str
        "A", "B", "C" o "D".
    semilla : int
        Semilla del modelo (random_state), para reproducibilidad.
    hiperparametros : dict, opcional
        Si se entrega, reemplaza la configuración por defecto. Se usará con
        hiperparámetros elegidos por validación cruzada anidada.

    Retorna
    -------
    estimador de scikit-learn
        Modelo listo para entrenar con .fit(X, y).
    """
    if nombre == "Ridge":
        alpha = config.RIDGE_ALPHA if hiperparametros is None else hiperparametros["alpha"]
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if hiperparametros is None:
        hiperparametros = obtener_hiperparametros(nombre, sensor, parcela)
    return CLASES_MODELO[nombre](**hiperparametros, random_state=semilla)
