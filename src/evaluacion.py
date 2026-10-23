"""
Evaluación del desempeño de los modelos.

Incluye las métricas del artículo (ecuaciones 5 a 7) y el r² (cuadrado de
la correlación entre observado y predicho), para analizar el hallazgo 1:
la posible confusión entre ambas versiones del R².
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import config
from .modelos import construir_modelo
from .simulacion import generar_base


def calcular_metricas(observado, predicho):
    """
    Calcula las métricas de desempeño de una regresión.

    - r2: coeficiente de determinación según la ecuación 5 del artículo,
      1 - SCE / SCT. Penaliza el sesgo y puede ser negativo.
    - r2_corr: cuadrado de la correlación de Pearson entre observado y
      predicho. Mide alineación con cualquier recta; no penaliza el sesgo.
    - rmse: raíz del error cuadrático medio (t/ha), ecuación 6.
    - mae: error absoluto medio (t/ha), ecuación 7.

    Parámetros
    ----------
    observado, predicho : array-like
        Valores observados y predichos de la respuesta.

    Retorna
    -------
    dict
        {"r2": ..., "r2_corr": ..., "rmse": ..., "mae": ...}
    """
    observado = np.asarray(observado, dtype=float)
    predicho = np.asarray(predicho, dtype=float)
    return {
        "r2": float(r2_score(observado, predicho)),
        "r2_corr": float(np.corrcoef(observado, predicho)[0, 1] ** 2),
        "rmse": float(np.sqrt(mean_squared_error(observado, predicho))),
        "mae": float(mean_absolute_error(observado, predicho)),
    }


def particion_aleatoria(n_filas, semilla, prop_prueba=None):
    """
    Partición aleatoria del artículo: 70 % entrenamiento y 30 % prueba,
    sin tener en cuenta la ubicación de los puntos.

    Parámetros
    ----------
    n_filas : int
        Número de observaciones.
    semilla : int
        Semilla de la partición.
    prop_prueba : float, opcional
        Proporción de prueba. Por defecto, config.PROP_PRUEBA (0,30).

    Retorna
    -------
    tuple de numpy.ndarray
        (índices de entrenamiento, índices de prueba).
    """
    prop_prueba = config.PROP_PRUEBA if prop_prueba is None else prop_prueba
    indices = np.arange(n_filas)
    return train_test_split(indices, test_size=prop_prueba, random_state=semilla)


def evaluar_como_articulo(datos, nombre_modelo, semilla, respuesta=None):
    """
    Recrea el procedimiento del artículo para una combinación de parcela,
    sensor y estrategia: partición aleatoria 70/30 de los 300 puntos de
    muestreo, entrenamiento con los 8 predictores y métricas en
    entrenamiento y en prueba.

    Parámetros
    ----------
    datos : pandas.DataFrame
        Filas de una sola parcela, un solo sensor y una sola estrategia.
    nombre_modelo : str
        "CART", "RF" o "GTB".
    semilla : int
        Semilla de la partición y del modelo.
    respuesta : str, opcional
        Columna respuesta. Por defecto, config.VARIABLE_RESPUESTA
        ("rendimiento", el real). Para recrear B y D como el artículo se
        usa "rendimiento_interpolado".

    Retorna
    -------
    dict
        Identificación del experimento y métricas con prefijo "ent_"
        (entrenamiento) y "pru_" (prueba).
    """
    respuesta = config.VARIABLE_RESPUESTA if respuesta is None else respuesta
    muestreo = datos[datos["tipo_punto"] == "muestreo"]
    parcela = muestreo["parcela"].iloc[0]
    sensor = muestreo["sensor"].iloc[0]

    X = muestreo[config.PREDICTORES].to_numpy()
    y = muestreo[respuesta].to_numpy()
    ent, pru = particion_aleatoria(len(y), semilla)

    modelo = construir_modelo(nombre_modelo, sensor, parcela, semilla)
    modelo.fit(X[ent], y[ent])

    resultado = {
        "parcela": parcela,
        "sensor": sensor,
        "estrategia": muestreo["estrategia"].iloc[0],
        "modelo": nombre_modelo,
        "respuesta": respuesta,
        "semilla": semilla,
    }
    for prefijo, idx in (("ent_", ent), ("pru_", pru)):
        metricas = calcular_metricas(y[idx], modelo.predict(X[idx]))
        resultado.update({prefijo + k: v for k, v in metricas.items()})
    return resultado


def ejecutar_recreacion(n_replicas=None, archivo="02_replicas.csv", recalcular=False):
    """
    Recrea el análisis del artículo con varias réplicas y guarda los resultados.

    Cada réplica genera una base simulada con una semilla distinta
    (config.SEMILLA + número de réplica) y aplica el procedimiento del
    artículo a todas las combinaciones de parcela, sensor, estrategia y
    modelo. En B y D se evalúan dos respuestas: el rendimiento interpolado
    (recreación fiel del artículo) y el rendimiento real (contraste).

    Si el archivo de resultados ya existe y recalcular es False, se lee el
    archivo en lugar de volver a entrenar los modelos.

    Parámetros
    ----------
    n_replicas : int, opcional
        Número de réplicas. Por defecto, config.N_REPLICAS (30).
    archivo : str
        Nombre del archivo CSV en la carpeta results/.
    recalcular : bool
        Si es True, vuelve a calcular aunque el archivo exista.

    Retorna
    -------
    pandas.DataFrame
        Una fila por réplica, parcela, sensor, estrategia, modelo y respuesta,
        con las métricas de entrenamiento y de prueba.
    """
    ruta = config.DIR_RESULTADOS / archivo
    if ruta.exists() and not recalcular:
        return pd.read_csv(ruta)

    n_replicas = config.N_REPLICAS if n_replicas is None else n_replicas
    filas = []
    inicio = time.time()
    for replica in range(n_replicas):
        semilla = config.SEMILLA + replica
        base = generar_base(semilla=semilla, guardar=False)
        for (parcela, sensor, estrategia), datos in base.groupby(
            ["parcela", "sensor", "estrategia"]
        ):
            respuestas = [config.VARIABLE_RESPUESTA]
            if datos["rendimiento_interpolado"].notna().all():
                respuestas.append("rendimiento_interpolado")
            for respuesta in respuestas:
                for modelo in config.MODELOS:
                    resultado = evaluar_como_articulo(datos, modelo, semilla, respuesta)
                    resultado["replica"] = replica
                    filas.append(resultado)
        minutos = (time.time() - inicio) / 60
        print(f"Réplica {replica + 1}/{n_replicas} terminada ({minutos:.1f} min)")

    resultados = pd.DataFrame(filas)
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(ruta, index=False)
    return resultados
