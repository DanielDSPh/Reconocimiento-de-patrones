
# Análisis de la base de datos Wine Quality

import pandas as pd #se utiliza para trabajar con datos en forma de tablas
import numpy as np #para cálculos numéricos y creación de arreglos
from scipy.sparse import csr_matrix #para convertir matriz dispersa en CSR


# 3.1 Cargar los datos en un DataFrame

url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv(url_wine, sep=";")  # delimitador es ";"

print(" WINE QUALITY ")
print("Forma (registros, atributos):", wine.shape) #devuelve el núm de (filas, columnas)
print("\nTipos de datos:\n", wine.dtypes) #muestra el tipo de dato de cada columna
print("\nPrimeras 10 filas:\n", wine.head(10)) ##imprime las primeras 10 filas del dataFrame


# 3.2 Llaves y dimensiones

print("\nLlaves :", wine.keys().tolist()) #para obtener los nombres de los atributos y colocarlos en una lista
print("Número de filas (regsistros):", wine.shape[0]) #devuelve el primer valor de la tupla (filas)
print("Número de columnas(atributos):", wine.shape[1]) #devuelve el segundo valor de la tupla (columnas)


# 3.3 Valores faltantes

print("\nValores faltantes (NaN) por columna:\n", wine.isna().sum()) #crea una tabla donde cada valor es true si está vacío o es NaN
#después suma todos los true 

# 3.4 Matriz 2D 5x5 y conversión a dispersa

arr = np.eye(5)  # matriz identidad
print("\nMatriz identidad 5x5 :\n", arr)

sparse_matrix = csr_matrix(arr)
print("\nMatriz dispersa (CSR):\n", sparse_matrix)


# 3.5 Estadísticas básicas(media, mínimo, máximo, percentil, desviación estándar)

desc_wine = wine.describe() #calcula las estadísticas básicas
print("\nMedia de cada columna:\n", desc_wine.loc["mean"]) #selecciona solo la fila de medias
print("\nDesviación estándar de cada columna:\n", desc_wine.loc["std"]) #selecciona solo la fila de desviaciones estándar 
#print("\nMínimo de cada columna:\n", desc_wine.loc["min"]) #selecciona solo la fila minimo
#print("\nMáximo de cada columna:\n", desc_wine.loc["max"]) #selecciona solo la fila de máximo
#print("\nPercentil 1 de cada columna:\n", desc_wine.loc["25%"]) #selecciona solo la fila de percentil 25%
#print("\nPercentil 2 de cada columna:\n", desc_wine.loc["50%"]) #selecciona solo la fila de percentil 50%
#print("\nPercentil 3 de cada columna:\n", desc_wine.loc["75%"]) #selecciona solo la fila de percentil 75%

# 3.6 Número de muestras por clase

print("\nNúmero de muestras por clase (columna 'quality'):\n", wine["quality"].value_counts())



# Repetimos conteo de clases

print("Número de muestras por clase (quality):\n", wine["quality"].value_counts()) #selecciona la columna quality y cuenta cuantos vinos hay en cada nivel


# 3.8 Primeras 10 filas y 2 primeras columnas

print("\nPrimeras 10 filas y 2 primeras columnas:\n", wine.iloc[:10, :2]) #usando indices del data frame
