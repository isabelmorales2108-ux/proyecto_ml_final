"""
Esquemas de validación para estudiar la fuga de información espacial.

Todos los esquemas devuelven una lista de particiones (índices de
entrenamiento, índices de prueba), con la misma interfaz, para que puedan
compararse modificando una sola condición del experimento.

- Aleatoria (K particiones): como el artículo, ignora la ubicación.
- Por bloques: el lote se divide en una grilla y cada bloque es una prueba.
- Por bloques con zona de exclusión: además, se retiran del entrenamiento
  los puntos a menos de una distancia dada del bloque de prueba.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from . import config


def asignar_bloques(xy, lado_m=None, n_bloques_lado=None):
    """
    Asigna cada punto a un bloque de una grilla cuadrada.

    Parámetros
    ----------
    xy : numpy.ndarray de forma (n, 2)
        Coordenadas de los puntos (m).
    lado_m : float, opcional
        Lado del lote. Por defecto, config.LOTE_LADO_M.
    n_bloques_lado : int, opcional
        Bloques por lado. Por defecto, config.N_BLOQUES_LADO (4 x 4 = 16).

    Retorna
    -------
    numpy.ndarray de forma (n,)
        Número de bloque de cada punto (0 a n_bloques_lado ** 2 - 1).
    """
    lado_m = config.LOTE_LADO_M if lado_m is None else lado_m
    n_bloques_lado = config.N_BLOQUES_LADO if n_bloques_lado is None else n_bloques_lado
    tamano = lado_m / n_bloques_lado
    columna = np.clip((xy[:, 0] // tamano).astype(int), 0, n_bloques_lado - 1)
    fila = np.clip((xy[:, 1] // tamano).astype(int), 0, n_bloques_lado - 1)
    return fila * n_bloques_lado + columna


def particiones_aleatorias(n_puntos, n_particiones, semilla):
    """
    Validación cruzada aleatoria con K particiones (ignora la ubicación).

    Retorna
    -------
    list de tuplas (índices de entrenamiento, índices de prueba)
    """
    kfold = KFold(n_splits=n_particiones, shuffle=True, random_state=semilla)
    return list(kfold.split(np.arange(n_puntos)))


def particiones_por_bloques(xy, distancia_exclusion_m=0.0, lado_m=None, n_bloques_lado=None):
    """
    Validación cruzada por bloques espaciales, con zona de exclusión opcional.

    Cada bloque de la grilla se usa una vez como conjunto de prueba. Si la
    distancia de exclusión es mayor que 0, se retiran del entrenamiento los
    puntos que estén a menos de esa distancia de algún punto de prueba.

    Parámetros
    ----------
    xy : numpy.ndarray de forma (n, 2)
        Coordenadas de los puntos (m).
    distancia_exclusion_m : float
        Ancho de la zona de exclusión (m). 0 = sin zona de exclusión.
    lado_m, n_bloques_lado : opcionales
        Tamaño del lote y de la grilla (por defecto, los de config).

    Retorna
    -------
    list de tuplas (índices de entrenamiento, índices de prueba)
        Una por cada bloque que contenga puntos.
    """
    bloques = asignar_bloques(xy, lado_m, n_bloques_lado)
    particiones = []
    for bloque in np.unique(bloques):
        prueba = np.where(bloques == bloque)[0]
        entrenamiento = np.where(bloques != bloque)[0]
        if distancia_exclusion_m > 0:
            distancia_minima = cdist(xy[entrenamiento], xy[prueba]).min(axis=1)
            entrenamiento = entrenamiento[distancia_minima >= distancia_exclusion_m]
        particiones.append((entrenamiento, prueba))
    return particiones