# ---------------------------------------
# Análisis de la base de datos Wine Quality
# ---------------------------------------
import pandas as pd #se utiliza la lib 
import numpy as np
from scipy.sparse import csr_matrix

# ---------------------------------------
# 3.1 Cargar los datos en un DataFrame
# ---------------------------------------
url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(url_wine, sep=";")  # delimitador es ";"

print("=== WINE QUALITY ===")
print("Forma (registros, atributos):", wine.shape)
print("\nTipos de datos:\n", wine.dtypes)
print("\nPrimeras 10 filas:\n", wine.head(10))

# ---------------------------------------
# 3.2 Llaves y dimensiones
# ---------------------------------------
print("\nLlaves :", wine.keys().tolist())
print("Número de filas (regsistros):", wine.shape[0])
print("Número de columnas(atributos):", wine.shape[1])

# ---------------------------------------
# 3.3 Valores faltantes
# ---------------------------------------
print("\nValores faltantes (NaN) por columna:\n", wine.isna().sum())

# ---------------------------------------
# 3.4 Matriz 2D 5x5 y conversión a dispersa
# ---------------------------------------
arr = np.eye(5)  # matriz identidad
print("\nMatriz 5x5 con diagonal de unos:\n", arr)

sparse_matrix = csr_matrix(arr)
print("\nMatriz dispersa (CSR):\n", sparse_matrix)

# ---------------------------------------
# 3.5 Estadísticas básicas
# ---------------------------------------
desc_wine = wine.describe()
print("\nMedia de cada columna:\n", desc_wine.loc["mean"])
print("\nDesviación estándar de cada columna:\n", desc_wine.loc["std"])

# ---------------------------------------
# 3.6 Número de muestras por clase
# ---------------------------------------
print("\nNúmero de muestras por clase (columna 'quality'):\n", wine["quality"].value_counts())

# ---------------------------------------
# 3.7 Añadir encabezados (ya vienen incluidos en el CSV)
# Repetimos conteo de clases
print("\n[Con encabezados ya incluidos en el CSV]")
print("Número de muestras por clase (quality):\n", wine["quality"].value_counts())

# ---------------------------------------
# 3.8 Primeras 10 filas y 2 primeras columnas
# ---------------------------------------
print("\nPrimeras 10 filas y 2 primeras columnas:\n", wine.iloc[:10, :2])
