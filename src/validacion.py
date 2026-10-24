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


def evaluar_replica_extension(replica, alcance_m, sensor, estrategia,
                              parcelas=None, modelos=None, esquemas=None):
    """
    Evalúa los esquemas, respuestas y modelos indicados en una réplica.

    Los parámetros parcelas, modelos y esquemas permiten acotar el cálculo
    (por ejemplo, en la sensibilidad al alcance). Si son None, se usan
    todas las parcelas, config.MODELOS_EXTENSION y ESQUEMAS.

    Retorna
    -------
    list de dict
        Una fila por parcela, esquema, respuesta y modelo.
    """
    parcelas = config.PARCELAS if parcelas is None else parcelas
    modelos = config.MODELOS_EXTENSION if modelos is None else modelos
    esquemas = ESQUEMAS if esquemas is None else esquemas
    semilla = config.SEMILLA + replica
    base = generar_base(semilla=semilla, alcance_m=alcance_m, guardar=False)
    base = base[(base["sensor"] == sensor) & (base["estrategia"] == estrategia)
                & (base["tipo_punto"] == "muestreo") & base["parcela"].isin(parcelas)]
    filas = []
    for parcela, datos in base.groupby("parcela"):
        xy = datos[["x", "y"]].to_numpy()
        X = datos[config.PREDICTORES].to_numpy()
        y_real = datos[config.VARIABLE_RESPUESTA].to_numpy()
        respuestas = [config.VARIABLE_RESPUESTA]
        if datos["rendimiento_interpolado"].notna().all():
            respuestas.append("rendimiento_interpolado")
        for esquema in esquemas:
            particiones = construir_particiones(esquema, xy, semilla)
            for respuesta in respuestas:
                y = datos[respuesta].to_numpy()
                for modelo in modelos:
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
                       recalcular=False, n_procesos=-1,
                       parcelas=None, modelos=None, esquemas=None):
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
    parcelas, modelos, esquemas : list, opcionales
        Permiten acotar el cálculo. Por defecto, todos.

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
        delayed(evaluar_replica_extension)(r, alcance_m, sensor, estrategia,
                                           parcelas, modelos, esquemas)
        for r in range(n_replicas)
    )
    tabla = pd.DataFrame([fila for filas in por_replica for fila in filas])
    print(f"{n_replicas} réplicas terminadas en {(time.time() - inicio) / 60:.1f} min")
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla


# ---------------------------------------------------------------------
# Validación por parcela: entrenar con tres parcelas y evaluar en la cuarta
# ---------------------------------------------------------------------
def evaluar_replica_por_parcela(replica, sensor, estrategia):
    """
    Evalúa la transferencia entre parcelas en una réplica: para cada
    parcela, entrena con las otras tres y predice la parcela excluida.

    Los modelos del artículo usan los hiperparámetros de la Tabla 3 de la
    parcela evaluada. Se usa el rendimiento real, porque la respuesta
    interpolada solo existe en B y D.

    Retorna
    -------
    list de dict
        Una fila por parcela evaluada y modelo, con las métricas y el sesgo
        (media predicha menos media observada, en t/ha).
    """
    semilla = config.SEMILLA + replica
    base = generar_base(semilla=semilla, guardar=False)
    base = base[(base["sensor"] == sensor) & (base["estrategia"] == estrategia)
                & (base["tipo_punto"] == "muestreo")]
    filas = []
    for parcela in config.PARCELAS:
        entrenamiento = base[base["parcela"] != parcela]
        prueba = base[base["parcela"] == parcela]
        X_ent = entrenamiento[config.PREDICTORES].to_numpy()
        y_ent = entrenamiento[config.VARIABLE_RESPUESTA].to_numpy()
        X_pru = prueba[config.PREDICTORES].to_numpy()
        y_pru = prueba[config.VARIABLE_RESPUESTA].to_numpy()
        for modelo in config.MODELOS_EXTENSION:
            ajustado = construir_modelo(modelo, sensor, parcela, semilla).fit(X_ent, y_ent)
            prediccion = ajustado.predict(X_pru)
            filas.append({
                "replica": replica, "parcela_evaluada": parcela, "modelo": modelo,
                **calcular_metricas(y_pru, prediccion),
                "sesgo": float(prediccion.mean() - y_pru.mean()),
            })
    return filas


def ejecutar_validacion_por_parcela(n_replicas=None, sensor="Sentinel-2",
                                    estrategia="multivariada",
                                    archivo="03_validacion_por_parcela.csv",
                                    recalcular=False, n_procesos=-1):
    """
    Ejecuta la validación por parcela en varias réplicas, en paralelo.

    Parámetros
    ----------
    n_replicas : int, opcional
        Número de réplicas. Por defecto, config.N_REPLICAS_EXTENSION (10).
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
        Una fila por réplica, parcela evaluada y modelo.
    """
    ruta = config.DIR_RESULTADOS / archivo
    if ruta.exists() and not recalcular:
        return pd.read_csv(ruta)

    n_replicas = config.N_REPLICAS_EXTENSION if n_replicas is None else n_replicas
    inicio = time.time()
    por_replica = Parallel(n_jobs=n_procesos)(
        delayed(evaluar_replica_por_parcela)(r, sensor, estrategia) for r in range(n_replicas)
    )
    tabla = pd.DataFrame([fila for filas in por_replica for fila in filas])
    print(f"{n_replicas} réplicas terminadas en {(time.time() - inicio) / 60:.1f} min")
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla


# ---------------------------------------------------------------------
# Selección de hiperparámetros: fija, con fuga y anidada (hallazgo 12)
# ---------------------------------------------------------------------
def _configuracion(nombre, combinacion):
    """Completa una combinación de la rejilla con los valores fijos del modelo."""
    if nombre == "Ridge":
        return combinacion
    return {**config.HIPERPARAMETROS_FIJOS[nombre], **combinacion}


def _r2_validacion_interna(nombre, sensor, parcela, X, y, combinacion, semilla):
    """R² medio de una combinación en una validación cruzada interna."""
    interna = KFold(n_splits=config.N_PARTICIONES_INTERNAS, shuffle=True, random_state=semilla)
    puntajes = []
    for ent, val in interna.split(X):
        modelo = construir_modelo(nombre, sensor, parcela, semilla, _configuracion(nombre, combinacion))
        modelo.fit(X[ent], y[ent])
        puntajes.append(calcular_metricas(y[val], modelo.predict(X[val]))["r2"])
    return float(np.mean(puntajes))


def evaluar_replica_seleccion(replica, sensor="Sentinel-2", estrategia="multivariada",
                              parcelas=("A", "C"), modelos=("Ridge", "RF")):
    """
    Compara tres formas de elegir hiperparámetros en una réplica, con una
    validación cruzada externa aleatoria de 5 particiones:

    - "fija": hiperparámetros del artículo (Tabla 3) o alpha = 1 en Ridge.
    - "con_fuga": en cada partición externa se elige la combinación de la
      rejilla con mejor desempeño en el propio conjunto de prueba, como
      hicieron los autores (hallazgo 12).
    - "anidada": la combinación se elige con una validación cruzada interna
      solo sobre el entrenamiento; el conjunto de prueba se usa una vez.

    Retorna
    -------
    list de dict
        Una fila por parcela, modelo y forma de selección.
    """
    semilla = config.SEMILLA + replica
    base = generar_base(semilla=semilla, guardar=False)
    base = base[(base["sensor"] == sensor) & (base["estrategia"] == estrategia)
                & (base["tipo_punto"] == "muestreo") & base["parcela"].isin(parcelas)]
    externa = KFold(n_splits=config.N_PARTICIONES_EXTERNAS, shuffle=True, random_state=semilla)
    filas = []
    for parcela, datos in base.groupby("parcela"):
        X = datos[config.PREDICTORES].to_numpy()
        y = datos[config.VARIABLE_RESPUESTA].to_numpy()
        for nombre in modelos:
            predicciones = {forma: np.full(len(y), np.nan) for forma in ("fija", "con_fuga", "anidada")}
            for ent, pru in externa.split(X):
                # 1. Fija
                modelo = construir_modelo(nombre, sensor, parcela, semilla).fit(X[ent], y[ent])
                predicciones["fija"][pru] = modelo.predict(X[pru])
                # 2. Con fuga: se elige mirando el conjunto de prueba
                mejor, mejor_r2 = None, -np.inf
                for combinacion in config.REJILLAS[nombre]:
                    modelo = construir_modelo(nombre, sensor, parcela, semilla,
                                              _configuracion(nombre, combinacion)).fit(X[ent], y[ent])
                    pred = modelo.predict(X[pru])
                    r2 = calcular_metricas(y[pru], pred)["r2"]
                    if r2 > mejor_r2:
                        mejor, mejor_r2 = pred, r2
                predicciones["con_fuga"][pru] = mejor
                # 3. Anidada: se elige con validación interna en el entrenamiento
                elegida = max(config.REJILLAS[nombre], key=lambda c: _r2_validacion_interna(
                    nombre, sensor, parcela, X[ent], y[ent], c, semilla))
                modelo = construir_modelo(nombre, sensor, parcela, semilla,
                                          _configuracion(nombre, elegida)).fit(X[ent], y[ent])
                predicciones["anidada"][pru] = modelo.predict(X[pru])
            for forma, pred in predicciones.items():
                filas.append({"replica": replica, "parcela": parcela, "modelo": nombre,
                              "seleccion": forma, **calcular_metricas(y, pred)})
    return filas


def ejecutar_seleccion_hiperparametros(n_replicas=5, archivo="03_seleccion_hiperparametros.csv",
                                       recalcular=False, n_procesos=-1, **opciones):
    """
    Ejecuta la comparación de formas de selección de hiperparámetros en
    varias réplicas, en paralelo, y guarda los resultados en results/.
    """
    ruta = config.DIR_RESULTADOS / archivo
    if ruta.exists() and not recalcular:
        return pd.read_csv(ruta)
    inicio = time.time()
    por_replica = Parallel(n_jobs=n_procesos)(
        delayed(evaluar_replica_seleccion)(r, **opciones) for r in range(n_replicas)
    )
    tabla = pd.DataFrame([fila for filas in por_replica for fila in filas])
    print(f"{n_replicas} réplicas terminadas en {(time.time() - inicio) / 60:.1f} min")
    config.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, index=False)
    return tabla
