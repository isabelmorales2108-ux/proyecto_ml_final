"""
Generación de los datos simulados del proyecto.

Modelo generador (ver references/ficha_extraccion.md, sección 11):
coordenadas -> rendimiento con autocorrelación espacial -> cobertura del dosel
-> bandas por mezcla lineal follaje-suelo -> índices de vegetación.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, skewnorm

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