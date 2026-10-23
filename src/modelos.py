"""
Construcción de los modelos del artículo de referencia.

Combina los hiperparámetros reportados en la Tabla 3 (config.HIPERPARAMETROS)
con los valores por defecto de Google Earth Engine para los no reportados
(config.HIPERPARAMETROS_FIJOS).

Equivalencias Earth Engine -> scikit-learn:
    smileCart              -> DecisionTreeRegressor
    smileRandomForest      -> RandomForestRegressor
    smileGradientTreeBoost -> GradientBoostingRegressor
"""

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    Construye un modelo de scikit-learn sin entrenar, configurado como en
    el artículo.

    Parámetros
    ----------
    nombre : str
        "CART", "RF" o "GTB".
    sensor : str
        "Sentinel-2" o "PlanetScope".
    parcela : str
        "A", "B", "C" o "D".
    semilla : int
        Semilla del modelo (random_state), para reproducibilidad.
    hiperparametros : dict, opcional
        Si se entrega, reemplaza la configuración del artículo. Se usará en
        la extensión, con hiperparámetros elegidos por validación cruzada.

    Retorna
    -------
    estimador de scikit-learn
        Modelo listo para entrenar con .fit(X, y).
    """
    if hiperparametros is None:
        hiperparametros = obtener_hiperparametros(nombre, sensor, parcela)
    return CLASES_MODELO[nombre](**hiperparametros, random_state=semilla)