'''
 *
 * Copyright (C) 2025 Juan Solsona
 *
'''

import sys
from typing import Dict
from dataclasses import dataclass


@dataclass
class DatosEjemplo:
    lote: int
    edad: int
    ingresos: float
    compra: bool
    es_entrenamiento: bool

    @classmethod
    def init_con_sample(cls) -> Dict[int, 'DatosEjemplo']:
        datos = [
            cls(lote=1, edad=42, ingresos=2.6, compra=True, es_entrenamiento=True),
            cls(lote=2, edad=25, ingresos=1.5, compra=False, es_entrenamiento=True),
            cls(lote=3, edad=55, ingresos=1.7, compra=False, es_entrenamiento=True),
            cls(lote=4, edad=38, ingresos=2.5, compra=True, es_entrenamiento=True),
            cls(lote=5, edad=60, ingresos=1.3, compra=False, es_entrenamiento=True),
            cls(lote=6, edad=32, ingresos=2.1, compra=True, es_entrenamiento=True),
            cls(lote=7, edad=48, ingresos=2.2, compra=True, es_entrenamiento=True),
            cls(lote=8, edad=28, ingresos=1.8, compra=False, es_entrenamiento=True),
            cls(lote=9, edad=50, ingresos=1.9, compra=False, es_entrenamiento=True),
            cls(lote=10, edad=35, ingresos=2.3, compra=True, es_entrenamiento=True),
            cls(lote=11, edad=22, ingresos=1.2, compra=False, es_entrenamiento=False),
            cls(lote=12, edad=40, ingresos=2.8, compra=True, es_entrenamiento=False),
            cls(lote=13, edad=58, ingresos=1.5, compra=False, es_entrenamiento=False),
            cls(lote=14, edad=45, ingresos=2.4, compra=True, es_entrenamiento=False),
        ]
        return {dato.lote: dato for dato in datos}

    @staticmethod
    def normalizar(datos: Dict[int, 'DatosEjemplo'],
                   solo_entrenamiento: bool = False) -> Dict[int, 'DatosEjemplo']:
        """
        Normaliza edad e ingresos de todas las instancias en el diccionario.

        Args:
            datos: Diccionario con instancias de DatosEjemplo
            solo_entrenamiento: Si True, usa solo datos de entrenamiento para calcular rangos

        Returns:
            Nuevo diccionario con instancias normalizadas
        """
        # Calcular rangos para edad
        max_edad = calcula_max(datos, 'edad', solo_entrenamiento)
        min_edad = calcula_min(datos, 'edad', solo_entrenamiento)
        rango_edad = max_edad - min_edad

        # Calcular rangos para ingresos
        max_ingresos = calcula_max(datos, 'ingresos', solo_entrenamiento)
        min_ingresos = calcula_min(datos, 'ingresos', solo_entrenamiento)
        rango_ingresos = max_ingresos - min_ingresos

        # Crear nuevo diccionario con instancias normalizadas
        datos_normalizados = {}
        for lote, instancia in datos.items():
            edad_normalizada = (instancia.edad - min_edad) / rango_edad if rango_edad > 0 else 0
            ingresos_normalizados = (instancia.ingresos - min_ingresos) / rango_ingresos if rango_ingresos > 0 else 0

            datos_normalizados[lote] = DatosEjemplo(
                lote=instancia.lote,
                edad=edad_normalizada,
                ingresos=ingresos_normalizados,
                compra=instancia.compra,
                es_entrenamiento=instancia.es_entrenamiento
            )

        return datos_normalizados
# Calcula el máximo de una columna
def calcula_max(valores: Dict[int, DatosEjemplo], columna: str,
                solo_entrenamiento: bool = False) -> float:
    max_actual: float = 0
    for dato in valores.values():
        if not solo_entrenamiento or dato.es_entrenamiento:
            if dato.__dict__[columna] > max_actual:
                max_actual = dato.__dict__[columna]
    return max_actual


# Calcula el mínimo de una columna
def calcula_min(valores: Dict[int, DatosEjemplo], columna: str,
                solo_entrenamiento: bool = False) -> float:
    min_actual: float = sys.float_info.max
    for dato in valores.values():
        if not solo_entrenamiento or dato.es_entrenamiento:
            if dato.__dict__[columna] < min_actual:
                min_actual = dato.__dict__[columna]
    return min_actual


# Normaliza una columna (valores entre 0 y 1)
def normaliza_columna(valores: Dict[int, DatosEjemplo], columna: str,
                      solo_entrenamiento: bool = False) -> Dict[
    int, float]:
    maximo = calcula_max(valores, columna, solo_entrenamiento)
    minimo = calcula_min(valores, columna, solo_entrenamiento)
    rango = maximo - minimo

    valores_normalizados = {}
    for lote, dato in valores.items():
        if not solo_entrenamiento or dato.es_entrenamiento:
            valor_normalizado = (dato.__dict__[columna] - minimo) / rango \
                if rango > 0 else 0
            valores_normalizados[lote] = valor_normalizado

    return valores_normalizados


# Programa principal
datos_mapa = DatosEjemplo.init_con_sample()

# Normalizar edades e ingresos
edades_normalizadas = normaliza_columna(datos_mapa, 'edad')
ingresos_normalizados = normaliza_columna(datos_mapa, 'ingresos')

# Crear diccionario final con lote, edad_normalizada e ingresos_normalizados
diccionario_final = []
for lote, dato in datos_mapa.items():
    diccionario_final.append({
        'lote': lote,
        'edad_normalizada': edades_normalizadas.get(lote, None),
        'ingresos_normalizados': ingresos_normalizados.get(lote, None)
    })

print("\n" + "=" * 70)
print("Diccionario con valores normalizados:")
print("=" * 70)
for registro in diccionario_final:
    print(
        f"Lote: {registro['lote']:<3} | Edad Normalizada: "
        f"{registro['edad_normalizada']:.4f} | Ingresos Normalizados: "
        f"{registro['ingresos_normalizados']:.4f}")