# Base de datos: Mushroom, describe hongos en términos de características físicas y
# clasifica hongos venenosos y comestibles

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# ======================================
# 3.1 Cargar los datos Iris
# ======================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
directorio_local = "EjerciciosMushroom/agaricus-lepiota.data"

mushroom = pd.read_csv(directorio_local, header=None)

print("=== 3.1 Información general del DataFrame ===")
print("(filas, columnas):", mushroom.shape)
print("\nTipos de datos:\n", mushroom.dtypes)
print("\nPrimeras 10 filas:\n", mushroom.head(10))

# ======================================
# 3.2 Imprimir llaves y número de filas y columnas
# ======================================
print("\n=== 3.2 Llaves y dimensiones ===")
print("Llaves:", mushroom.keys().tolist())
print("Número de filas y columnas:", mushroom.shape)

# ======================================
# 3.3 Número de muestras faltantes o NaN
# ======================================
print("\n=== 3.3 Valores faltantes (NaN) ===")
print("NaN por columna:\n", mushroom.isnull().sum())
print("Total de NaN en todo el DataFrame:", mushroom.isnull().sum().sum())

# ======================================
# 3.5 Estadísticas básicas: media y desviación estándar

# NOTA: Como todos los datos de este DataFrame son categóricos, no se pueden calcular
# estadísticas como la media o la desviación estándar. Sin embargo, se pueden obtener
# estadísticas descriptivas básicas, como la categoría más frecuente.
# ======================================
print("\n=== 3.5 Estadísticas básicas ===")
desc = mushroom.describe(include = 'object')
print("Moda:\n", desc.loc["top"])

# ======================================
# 3.6 Número de muestras por clase
# ======================================
print("\n=== 3.6 Número de muestras por clase (comestibles vs venenosos) ===")
print(mushroom["poisonous"].value_counts())

# ======================================
# 3.7 Añadir encabezados y repetir conteo por clase
# ======================================
columnas = ["poisonous","cap_shape", "cap_surface", "cap_color", "bruises", "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color", "stalk_shape", "stalk_root", "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring", "stalk_color_below_ring", "veil_type", "veil_color", "ring_number", "ring_type", "spore_print_color", "population", "habitat"]
mushroom.columns = columnas

print("\n=== 3.7 Con encabezados añadidos ===")
print("Nombres de columnas:", mushroom.columns.tolist())
print("\nNúmero de muestras por clase con encabezado:")
print(mushroom["poisonous"].value_counts())

# ======================================
# 3.8 Mostrar las primeras 10 filas y 2 primeras columnas
# ======================================
print("\n=== 3.8 Primeras 10 filas y 2 primeras columnas ===")
print(mushroom.iloc[:10, :2])

