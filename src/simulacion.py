"""
Generación de los datos simulados del proyecto.

Modelo generador (ver references/ficha_extraccion.md, sección 11):
coordenadas -> rendimiento con autocorrelación espacial -> cobertura del dosel
-> bandas por mezcla lineal follaje-suelo -> índices de vegetación.
"""

import numpy as np


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