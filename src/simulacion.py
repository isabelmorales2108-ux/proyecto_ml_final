"""
Generación de los datos simulados del proyecto.

Modelo generador (ver references/ficha_extraccion.md, sección 11):
coordenadas -> rendimiento con autocorrelación espacial -> cobertura del dosel
-> bandas por mezcla lineal follaje-suelo -> índices de vegetación.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm, skewnorm

from . import config


def generar_coordenadas(n_puntos, lado_m, rng):
    """
    Genera coordenadas aleatorias uniformes dentro de un lote cuadrado.

    Reproduce el muestreo del artículo: puntos aleatorios distribuidos
    dentro de los límites de la parcela.

    Parámetros
    ----------
    n_puntos : int
        Número de puntos a generar (300 en el artículo).
    lado_m : float
        Lado del lote cuadrado, en metros.
    rng : numpy.random.Generator
        Generador aleatorio, creado con np.random.default_rng(semilla).

    Retorna
    -------
    numpy.ndarray de forma (n_puntos, 2)
        Columna 0: coordenada x (m). Columna 1: coordenada y (m).
    """
    return rng.uniform(0.0, lado_m, size=(n_puntos, 2))


def generar_campo_espacial(xy, alcance_m, pepita, rng):
    """
    Genera un campo aleatorio estandarizado con autocorrelación espacial.

    Usa un modelo exponencial con efecto pepita:
        correlación(d) = (1 - pepita) * exp(-3 * d / alcance)
    y la descomposición de Cholesky de la matriz de covarianza para
    mezclar valores aleatorios independientes.

    Parámetros
    ----------
    xy : numpy.ndarray de forma (n, 2)
        Coordenadas de los puntos, en metros.
    alcance_m : float
        Distancia a la que la correlación cae a ~5 %.
    pepita : float
        Fracción de la variabilidad sin estructura espacial (entre 0 y 1).
    rng : numpy.random.Generator
        Generador aleatorio.

    Retorna
    -------
    numpy.ndarray de forma (n,)
        Valores con media teórica 0, varianza 1 y autocorrelación espacial.
    """
    # 1. Distancias entre todos los pares de puntos (matriz n x n)
    distancias = cdist(xy, xy)

    # 2. Matriz de covarianza: correlación según la distancia
    covarianza = (1.0 - pepita) * np.exp(-3.0 * distancias / alcance_m)

    # 3. En la diagonal (cada punto consigo mismo) la varianza total es 1:
    #    parte espacial (1 - pepita) + parte local (pepita)
    covarianza[np.diag_indices_from(covarianza)] = 1.0

    # 4. Descomposición de Cholesky: covarianza = L @ L.T
    L = np.linalg.cholesky(covarianza)

    # 5. Mezcla de valores aleatorios independientes
    w = rng.standard_normal(len(xy))
    return L @ w


def transformar_a_rendimiento(z, media, desv, minimo, maximo, asimetria):
    """
    Convierte un campo estandarizado en rendimiento (t/ha).

    Si la asimetría es 0, aplica una transformación lineal. Si es distinta
    de 0, aplica una transformación por cuantiles hacia una normal
    asimétrica, que conserva el orden de los puntos y, por tanto, la
    estructura espacial del campo.

    Parámetros
    ----------
    z : numpy.ndarray
        Campo estandarizado (salida de generar_campo_espacial).
    media, desv : float
        Media y desviación estándar del rendimiento de la parcela (t/ha).
    minimo, maximo : float
        Límites del rendimiento (t/ha); los valores fuera se recortan.
    asimetria : float
        Parámetro de forma de la normal asimétrica (0 = simétrica;
        positivo = cola hacia valores altos).

    Retorna
    -------
    numpy.ndarray
        Rendimiento en t/ha.
    """
    if asimetria == 0:
        estandar = z
    else:
        # 1. De z a percentil (entre 0 y 1)
        percentil = norm.cdf(z)
        # 2. Del percentil al valor de la normal asimétrica
        valor = skewnorm.ppf(percentil, asimetria)
        # 3. Estandarizar con la media y la varianza teóricas de esa distribución
        m, v = skewnorm.stats(asimetria, moments="mv")
        estandar = (valor - m) / np.sqrt(v)

    rendimiento = media + desv * estandar
    return np.clip(rendimiento, minimo, maximo)


def calcular_cobertura(rendimiento, minima, maxima, rendimiento_medio, pendiente):
    """
    Calcula la cobertura promedio del dosel en la temporada (entre 0 y 1)
    a partir del rendimiento, con una curva logística con saturación.

    Parámetros
    ----------
    rendimiento : numpy.ndarray
        Rendimiento en t/ha.
    minima, maxima : float
        Cobertura mínima y máxima de la curva.
    rendimiento_medio : float
        Rendimiento (t/ha) en el punto de inflexión de la curva.
    pendiente : float
        Inclinación de la curva; controla la rapidez de la saturación.

    Retorna
    -------
    numpy.ndarray
        Fracción del píxel cubierta por follaje, promedio de junio a septiembre.
    """
    return minima + (maxima - minima) / (
        1.0 + np.exp(-pendiente * (rendimiento - rendimiento_medio))
    )


def mezclar_bandas(cobertura, reflectancia_follaje, reflectancia_suelo):
    """
    Calcula la reflectancia de cada banda por mezcla lineal de follaje y suelo.

        reflectancia = cobertura * follaje + (1 - cobertura) * suelo

    Parámetros
    ----------
    cobertura : numpy.ndarray
        Fracción del píxel cubierta por follaje (entre 0 y 1).
    reflectancia_follaje : dict
        Reflectancia del follaje por banda, por ejemplo {"azul": 0.02, ...}.
    reflectancia_suelo : dict
        Reflectancia del suelo por banda. Los valores pueden ser números
        (suelo uniforme) o arreglos con un valor por punto (suelo variable,
        como en la estrategia multivariada).

    Retorna
    -------
    dict
        Un arreglo de reflectancias por banda: {"azul": ..., "verde": ..., ...}.
    """
    return {
        banda: cobertura * reflectancia_follaje[banda]
        + (1.0 - cobertura) * reflectancia_suelo[banda]
        for banda in reflectancia_follaje
    }


def aplicar_ruido_sensor(bandas, sigma_relativo, rng):
    """
    Aplica el error radiométrico del sensor (fuente 2): un error relativo
    independiente en cada banda y en cada punto.

        banda_observada = banda * (1 + e),  e ~ Normal(0, sigma_relativo)

    Parámetros
    ----------
    bandas : dict
        Reflectancias por banda (salida de mezclar_bandas).
    sigma_relativo : float
        Desviación estándar del error relativo (0.025 = 2,5 %).
    rng : numpy.random.Generator
        Generador aleatorio.

    Retorna
    -------
    dict
        Reflectancias con ruido, por banda.
    """
    return {
        banda: valores * (1.0 + rng.normal(0.0, sigma_relativo, size=len(valores)))
        for banda, valores in bandas.items()
    }


def aplicar_iluminacion(bandas, sigma_relativo, rng):
    """
    Aplica la variación de iluminación o calibración (fuente 4): un mismo
    factor por punto que escala las cuatro bandas en la misma proporción.

        banda_observada = banda * (1 + f),  f ~ Normal(0, sigma_relativo)

    Como el factor es común a todas las bandas, se cancela en los índices
    normalizados (NDVI, GNDVI), pero no en las bandas individuales.

    Parámetros
    ----------
    bandas : dict
        Reflectancias por banda.
    sigma_relativo : float
        Desviación estándar del factor relativo (0.03 = 3 %).
    rng : numpy.random.Generator
        Generador aleatorio.

    Retorna
    -------
    dict
        Reflectancias con el factor de iluminación aplicado.
    """
    n_puntos = len(next(iter(bandas.values())))
    factor = 1.0 + rng.normal(0.0, sigma_relativo, size=n_puntos)
    return {banda: valores * factor for banda, valores in bandas.items()}


def calcular_indices(bandas, savi_l=0.5):
    """
    Calcula los cuatro índices de vegetación a partir de las bandas.

        NDVI  = (NIR - Rojo) / (NIR + Rojo)
        GNDVI = (NIR - Verde) / (NIR + Verde)
        EVI   = 2,5 * (NIR - Rojo) / (NIR + 6 * Rojo - 7,5 * Azul + 1)
        SAVI  = (1 + L) * (NIR - Rojo) / (NIR + Rojo + L)

    El EVI usa la formulación estándar (7,5 * Azul); el artículo escribe
    7 * Azul (hallazgo 8 de la ficha). Las reflectancias deben estar en
    escala 0-1.

    Parámetros
    ----------
    bandas : dict
        Reflectancias por banda: "azul", "verde", "rojo" y "nir".
    savi_l : float
        Factor de ajuste del suelo del SAVI (0,5 en el artículo).

    Retorna
    -------
    dict
        Un arreglo por índice: {"ndvi": ..., "gndvi": ..., "evi": ..., "savi": ...}.
    """
    azul, verde, rojo, nir = bandas["azul"], bandas["verde"], bandas["rojo"], bandas["nir"]
    return {
        "ndvi": (nir - rojo) / (nir + rojo),
        "gndvi": (nir - verde) / (nir + verde),
        "evi": 2.5 * (nir - rojo) / (nir + 6.0 * rojo - 7.5 * azul + 1.0),
        "savi": (1.0 + savi_l) * (nir - rojo) / (nir + rojo + savi_l),
    }


def ubicar_mediciones_excavacion(lado_m, n_zonas=3, n_columnas=5, n_filas=2):
    """
    Ubica los puntos de medición manual de las parcelas de excavación.

    Reproduce el diseño del artículo: el lote se divide en 3 zonas y en
    cada zona se miden 10 puntos distribuidos uniformemente. Se asume [S]
    que las zonas son franjas horizontales y que los 10 puntos forman una
    grilla de 5 columnas x 2 filas dentro de cada franja.

    Parámetros
    ----------
    lado_m : float
        Lado del lote cuadrado, en metros.
    n_zonas, n_columnas, n_filas : int
        Número de franjas y tamaño de la grilla dentro de cada franja.

    Retorna
    -------
    numpy.ndarray de forma (n_zonas * n_columnas * n_filas, 2)
        Coordenadas x, y de los puntos de medición (m).
    """
    alto_zona = lado_m / n_zonas
    xs = (np.arange(n_columnas) + 0.5) * lado_m / n_columnas
    fracciones = (np.arange(n_filas) + 0.5) / n_filas
    puntos = [
        (x, zona * alto_zona + f * alto_zona)
        for zona in range(n_zonas)
        for f in fracciones
        for x in xs
    ]
    return np.array(puntos)


def interpolar_idw(xy_mediciones, valores, xy_destino, potencia=2.0):
    """
    Interpola por distancia inversa ponderada (IDW).

    Cada punto de destino recibe un promedio de las mediciones, ponderado
    por 1 / distancia ** potencia: las mediciones cercanas pesan más.

    Parámetros
    ----------
    xy_mediciones : numpy.ndarray de forma (m, 2)
        Coordenadas de los puntos medidos.
    valores : numpy.ndarray de forma (m,)
        Valor medido en cada punto.
    xy_destino : numpy.ndarray de forma (n, 2)
        Coordenadas donde se estima el valor.
    potencia : float
        Exponente de la distancia (2 es el valor habitual).

    Retorna
    -------
    numpy.ndarray de forma (n,)
        Valores interpolados.
    """
    distancias = cdist(xy_destino, xy_mediciones)
    distancias = np.maximum(distancias, 1e-9)   # evita dividir por cero
    pesos = 1.0 / distancias ** potencia
    return (pesos @ valores) / pesos.sum(axis=1)


def generar_parcela(parcela, rng, alcance_m=None, ruido=None):
    """
    Genera los datos simulados de una parcela para los dos sensores y las
    dos estrategias de simulación.

    Dentro de la parcela, los dos sensores y las dos estrategias comparten
    los mismos puntos, el mismo rendimiento y la misma cobertura del dosel.
    Las estrategias solo difieren en las fuentes de ruido compartidas entre
    bandas (brillo del suelo e iluminación); los sensores, solo en el ruido
    del sensor.

    En las parcelas de excavación manual (B y D) se agregan las 30 mediciones
    del artículo y se calcula el rendimiento interpolado por IDW en los 300
    puntos de muestreo, como hicieron los autores. El rendimiento real se
    conserva en todos los puntos para medir el efecto de la interpolación.

    Parámetros
    ----------
    parcela : str
        Identificador de la parcela: "A", "B", "C" o "D".
    rng : numpy.random.Generator
        Generador aleatorio.
    alcance_m : float, opcional
        Alcance de la autocorrelación (m). Por defecto, el de config.
    ruido : dict, opcional
        Parámetros de ruido con la estructura de config.RUIDO. Por defecto,
        los de config. Permite calibrar sin modificar config.

    Retorna
    -------
    pandas.DataFrame
        Una fila por punto, sensor y estrategia: 1.200 filas en A y C
        (300 puntos) y 1.320 en B y D (300 de muestreo + 30 de medición).
    """
    info = config.PARCELAS_INFO[parcela]
    alcance = config.AUTOCORRELACION["alcance_m"] if alcance_m is None else alcance_m
    pepita = config.AUTOCORRELACION["pepita"]
    ruido = config.RUIDO if ruido is None else ruido
    cob = config.COBERTURA
    es_excavacion = info["metodo"] == "excavacion"

    # 1. Puntos: 300 de muestreo y, en B y D, las 30 mediciones manuales
    xy_muestreo = generar_coordenadas(config.N_PUNTOS, config.LOTE_LADO_M, rng)
    if es_excavacion:
        xy_medicion = ubicar_mediciones_excavacion(config.LOTE_LADO_M)
        xy = np.vstack([xy_muestreo, xy_medicion])
        tipo_punto = ["muestreo"] * len(xy_muestreo) + ["medicion"] * len(xy_medicion)
    else:
        xy = xy_muestreo
        tipo_punto = ["muestreo"] * len(xy_muestreo)

    # 2. Rendimiento real con autocorrelación espacial (en todos los puntos)
    z = generar_campo_espacial(xy, alcance, pepita, rng)
    rendimiento = transformar_a_rendimiento(
        z, info["media"], info["desv"], info["minimo"], info["maximo"], info["asimetria"]
    )

    # 3. Rendimiento interpolado (solo B y D): IDW desde las 30 mediciones
    if es_excavacion:
        n = len(xy_muestreo)
        medido = rendimiento[n:]
        rendimiento_interpolado = np.concatenate([
            interpolar_idw(xy_medicion, medido, xy_muestreo, config.IDW_POTENCIA),
            medido,
        ])
    else:
        rendimiento_interpolado = np.full(len(xy), np.nan)

    # 4. Fuente 1: desacople dosel-rendimiento (campo espacial independiente)
    desacople = generar_campo_espacial(xy, alcance, pepita, rng)
    rendimiento_visible = rendimiento + ruido["desacople"] * info["desv"] * desacople
    cobertura = calcular_cobertura(
        rendimiento_visible, cob["minima"], cob["maxima"],
        cob["rendimiento_medio"], cob["pendiente"],
    )

    tablas = []
    for estrategia in config.ESTRATEGIAS:
        # 5. Suelo: fijo (univariada) o con brillo variable en manchas (multivariada)
        if estrategia == "multivariada":
            humedad = norm.cdf(generar_campo_espacial(xy, alcance, pepita, rng))
            suelo = {
                b: config.REFLECTANCIA_SUELO_HUMEDO[b]
                + humedad * (config.REFLECTANCIA_SUELO_SECO[b] - config.REFLECTANCIA_SUELO_HUMEDO[b])
                for b in config.BANDAS
            }
        else:
            suelo = config.REFLECTANCIA_SUELO_HUMEDO

        bandas_base = mezclar_bandas(cobertura, config.REFLECTANCIA_FOLLAJE, suelo)

        # 6. Fuente 4: iluminación compartida (solo multivariada)
        if estrategia == "multivariada":
            bandas_base = aplicar_iluminacion(bandas_base, ruido["iluminacion"], rng)

        for sensor in config.SENSORES:
            # 7. Fuente 2: ruido del sensor y cálculo de índices
            bandas = aplicar_ruido_sensor(bandas_base, ruido["sensor"][sensor], rng)
            indices = calcular_indices(bandas, config.SAVI_L)

            tabla = pd.DataFrame({
                "parcela": parcela,
                "anio": info["anio"],
                "metodo": info["metodo"],
                "sensor": sensor,
                "estrategia": estrategia,
                "tipo_punto": tipo_punto,
                "punto": np.arange(len(xy)),
                "x": xy[:, 0],
                "y": xy[:, 1],
                config.VARIABLE_RESPUESTA: rendimiento,
                "rendimiento_interpolado": rendimiento_interpolado,
                **bandas,
                **indices,
                "cobertura_real": cobertura,
            })
            tablas.append(tabla)

    return pd.concat(tablas, ignore_index=True)


def generar_base(semilla=None, alcance_m=None, ruido=None, guardar=True,
                 nombre_archivo="datos_simulados.csv"):
    """
    Genera la base simulada completa: las cuatro parcelas, los dos sensores
    y las dos estrategias de simulación.

    Parámetros
    ----------
    semilla : int, opcional
        Semilla del generador aleatorio. Por defecto, config.SEMILLA.
        Para las réplicas se usan semillas distintas.
    alcance_m : float, opcional
        Alcance de la autocorrelación (m). Por defecto, el de config.
    ruido : dict, opcional
        Parámetros de ruido. Por defecto, config.RUIDO.
    guardar : bool
        Si es True, guarda la base en la carpeta data/.
    nombre_archivo : str
        Nombre del archivo CSV.

    Retorna
    -------
    pandas.DataFrame
        Base completa: 5.040 filas (1.200 en A y C, 1.320 en B y D).
    """
    semilla = config.SEMILLA if semilla is None else semilla
    rng = np.random.default_rng(semilla)

    base = pd.concat(
        [generar_parcela(p, rng, alcance_m=alcance_m, ruido=ruido) for p in config.PARCELAS],
        ignore_index=True,
    )
    base.insert(0, "semilla", semilla)

    if guardar:
        config.DIR_DATOS.mkdir(parents=True, exist_ok=True)
        base.to_csv(config.DIR_DATOS / nombre_archivo, index=False)

    return base