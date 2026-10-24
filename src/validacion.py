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

import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from . import config
from .evaluacion import calcular_metricas
from .modelos import construir_modelo
from .simulacion import generar_base


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


# ---------------------------------------------------------------------
# Ejecución del experimento de la extensión
# ---------------------------------------------------------------------
ESQUEMAS = ["aleatoria", "bloques", "bloques_exclusion"]


def construir_particiones(esquema, xy, semilla):
    """
    Devuelve las particiones de un esquema de validación.

    Parámetros
    ----------
    esquema : str
        "aleatoria", "bloques" o "bloques_exclusion".
    xy : numpy.ndarray de forma (n, 2)
        Coordenadas de los puntos (m).
    semilla : int
        Semilla de la partición aleatoria.

    Retorna
    -------
    list de tuplas (índices de entrenamiento, índices de prueba)
    """
    if esquema == "aleatoria":
        return particiones_aleatorias(len(xy), config.N_PARTICIONES_CV, semilla)
    if esquema == "bloques":
        return particiones_por_bloques(xy)
    if esquema == "bloques_exclusion":
        return particiones_por_bloques(xy, config.DISTANCIA_EXCLUSION_M)
    raise ValueError(f"Esquema desconocido: {esquema}")


def predecir_fuera_de_muestra(nombre_modelo, sensor, parcela, X, y, particiones, semilla):
    """
    Entrena un modelo en cada partición y predice su conjunto de prueba.

    Retorna
    -------
    numpy.ndarray
        Una predicción por punto, hecha por un modelo que no vio ese punto
        en el entrenamiento.
    """
    prediccion = np.full(len(y), np.nan)
    for entrenamiento, prueba in particiones:
        modelo = construir_modelo(nombre_modelo, sensor, parcela, semilla)
        modelo.fit(X[entrenamiento], y[entrenamiento])
        prediccion[prueba] = modelo.predict(X[prueba])
    return prediccion


def evaluar_replica_extension(replica, alcance_m, sensor, estrategia):
    """
    Evalúa todos los esquemas, respuestas y modelos en una réplica.

    Retorna
    -------
    list de dict
        Una fila por parcela, esquema, respuesta y modelo.
    """
    semilla = config.SEMILLA + replica
    base = generar_base(semilla=semilla, alcance_m=alcance_m, guardar=False)
    base = base[(base["sensor"] == sensor) & (base["estrategia"] == estrategia)
                & (base["tipo_punto"] == "muestreo")]
    filas = []
    for parcela, datos in base.groupby("parcela"):
        xy = datos[["x", "y"]].to_numpy()
        X = datos[config.PREDICTORES].to_numpy()
        y_real = datos[config.VARIABLE_RESPUESTA].to_numpy()
        respuestas = [config.VARIABLE_RESPUESTA]
        if datos["rendimiento_interpolado"].notna().all():
            respuestas.append("rendimiento_interpolado")
        for esquema in ESQUEMAS:
            particiones = construir_particiones(esquema, xy, semilla)
            for respuesta in respuestas:
                y = datos[respuesta].to_numpy()
                for modelo in config.MODELOS_EXTENSION:
                    prediccion = predecir_fuera_de_muestra(
                        modelo, sensor, parcela, X, y, particiones, semilla)
                    filas.append({
                        "replica": replica, "alcance_m": alcance_m, "parcela": parcela,
                        "esquema": esquema, "respuesta": respuesta, "modelo": modelo,
                        **calcular_metricas(y, prediccion),
                        "r2_vs_real": calcular_metricas(y_real, prediccion)["r2"],
                    })
    return filas


def ejecutar_extension(n_replicas=None, alcance_m=None, sensor="Sentinel-2",
                       estrategia="multivariada", archivo="03_extension.csv",
                       recalcular=False, n_procesos=-1):
    """
    Ejecuta el experimento de la extensión: compara los esquemas de
    validación para los cuatro modelos, con réplicas procesadas en paralelo.

    Parámetros
    ----------
    n_replicas : int, opcional
        Número de réplicas. Por defecto, config.N_REPLICAS_EXTENSION (10).
    alcance_m : float, opcional
        Alcance de la autocorrelación (m). Por defecto, el de config.
    sensor, estrategia : str
        Combinación evaluada (por defecto, Sentinel-2 y multivariada).
    archivo : str
        Nombre del archivo CSV en results/.
    recalcular : bool
        Si es True, recalcula aunque el archivo exista.
    n_procesos : int
        Procesos en paralelo (-1 = todos los núcleos disponibles).

    Retorna
    -------
    pandas.DataFrame
        Una fila por réplica, parcela, esquema, respuesta y modelo, con las
        métricas de las predicciones fuera de muestra. La columna r2_vs_real
        mide el desempeño frente al rendimiento real, aunque el modelo se
        haya entrenado con el interpolado.
    """
    ruta = config.DIR_RESULTADOS / archivo
    if ruta.exists() and not recalcular:
        return pd.read_csv(ruta)

    n_replicas = config.N_REPLICAS_EXTENSION if n_replicas is None else n_replicas
    alcance_m = config.AUTOCORRELACION["alcance_m"] if alcance_m is None else alcance_m
    inicio = time.time()
    por_replica = Parallel(n_jobs=n_procesos)(
        delayed(evaluar_replica_extension)(r, alcance_m, sensor, estrategia)
        for r in range(n_replicas)
    )
    tabla = pd.DataFrame([fila for filas in por_replica for fila in filas])
    print(f"{n_replicas} réplicas terminadas en {(time.time() - inicio) / 60:.1f} min")
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla
