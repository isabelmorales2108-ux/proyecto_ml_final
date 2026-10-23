"""
Calibración de las fuentes de ruido del generador.

Compara el R² de prueba de los modelos sobre los datos simulados con el
reportado en las Tablas 4 y 5 del artículo. Por decisión de diseño, se
calibra solo con las parcelas de cosechadora (A y C); el mismo ruido se
aplica después a B y D sin ajustarlo, para no ocultar el efecto de la
interpolación y de la saturación.
"""

import pandas as pd

from . import config
from .evaluacion import evaluar_como_articulo
from .simulacion import generar_base


def comparar_con_articulo(ruido=None, parcelas=("A", "C"), estrategia="multivariada",
                          semillas=range(5)):
    """
    Evalúa los tres modelos sobre bases simuladas con varias semillas y
    compara el R² de prueba promedio con el reportado en el artículo.

    Parámetros
    ----------
    ruido : dict, opcional
        Parámetros de ruido. Por defecto, config.RUIDO.
    parcelas : tuple
        Parcelas a comparar (por defecto, las de cosechadora).
    estrategia : str
        "univariada" o "multivariada".
    semillas : iterable de int
        Desplazamientos de semilla; cada base usa config.SEMILLA + desplazamiento.

    Retorna
    -------
    pandas.DataFrame
        Una fila por parcela, sensor y modelo con el R² simulado (media y
        desviación entre semillas), el R² del artículo y la diferencia.
    """
    filas = []
    for semilla in semillas:
        base = generar_base(semilla=config.SEMILLA + semilla, ruido=ruido, guardar=False)
        base = base[base["estrategia"] == estrategia]
        for (parcela, sensor), datos in base.groupby(["parcela", "sensor"]):
            if parcela not in parcelas:
                continue
            for modelo in config.MODELOS:
                filas.append(evaluar_como_articulo(datos, modelo, config.SEMILLA + semilla))

    resultados = pd.DataFrame(filas)
    tabla = (resultados.groupby(["parcela", "sensor", "modelo"])["pru_r2"]
             .agg(r2_sim="mean", r2_sim_desv="std").reset_index())
    tabla["r2_articulo"] = [
        config.METRICAS_ARTICULO[s][p][m][2]
        for p, s, m in zip(tabla["parcela"], tabla["sensor"], tabla["modelo"])
    ]
    tabla["diferencia"] = tabla["r2_sim"] - tabla["r2_articulo"]
    return tabla