# practica2.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# ======================================
# 3.1 Cargar los datos Iris
# ======================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
directorio_local = "EjerciciosIris/iris.data"

iris = pd.read_csv(directorio_local, header=None)

print("=== 3.1 Información general del DataFrame ===")
print("Forma (filas, columnas):", iris.shape)
print("\nTipos de datos:\n", iris.dtypes)
print("\nPrimeras 10 filas:\n", iris.head(10))


# ======================================
# 3.2 Imprimir llaves y número de filas y columnas
# ======================================
print("\n=== 3.2 Llaves y dimensiones ===")
print("Llaves:", iris.keys().tolist())
print("Número de filas y columnas:", iris.shape)


# ======================================
# 3.3 Número de muestras faltantes o NaN
# ======================================
print("\n=== 3.3 Valores faltantes (NaN) ===")
print("NaN por columna:\n", iris.isnull().sum())
print("Total de NaN en todo el DataFrame:", iris.isnull().sum().sum())


# ======================================
# 3.4 Crear matriz 5x5 identidad y convertir a dispersa CRS
# ======================================
print("\n=== 3.4 Matriz 5x5 e implementación dispersa ===")
A = np.eye(5)  # matriz identidad 5x5
print("Matriz 5x5:\n", A)

# Convertir a formato disperso CRS
A_sparse = csr_matrix(A)
print("\nMatriz dispersa en formato CRS:\n", A_sparse)

# Mostrar detalles del CRS
print("\nValores no cero:", A_sparse.data)
print("Índices de columna:", A_sparse.indices)
print("Índices de fila (indptr):", A_sparse.indptr)

# ======================================
# 3.5 Estadísticas básicas: media y desviación estándar
# ======================================
print("\n=== 3.5 Estadísticas básicas ===")
desc = iris.describe()
print("Media:\n", desc.loc["mean"])
print("\nDesviación estándar:\n", desc.loc["std"])

# ======================================
# 3.6 Número de muestras por clase
# ======================================
print("\n=== 3.6 Número de muestras por clase ===")
print(iris[4].value_counts())

# ======================================
# 3.7 Añadir encabezados y repetir conteo por clase
# ======================================
columnas = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris.columns = columnas

print("\n=== 3.7 Con encabezados añadidos ===")
print("Nombres de columnas:", iris.columns.tolist())
print("\nNúmero de muestras por clase con encabezado:")
print(iris["species"].value_counts())

# ======================================
# 3.8 Mostrar las primeras 10 filas y 2 primeras columnas
# ======================================
print("\n=== 3.8 Primeras 10 filas y 2 primeras columnas ===")
print(iris.iloc[:10, :2])

# ======================================
# 3.9 Gráfico de barras de mínimo, media y máximo
# ======================================
print("\n=== 3.9 Gráfico de barras de estadísticas ===")
# Estadísticas
media = desc.loc["mean"]
minimo = desc.loc["min"]
maximo = desc.loc["max"]


x = np.arange(len(columnas)-1)
width = 0.25

plt.figure(figsize=(9,5))
plt.bar(x - width, minimo, width, label="Mínimo", color="skyblue")
plt.bar(x, media, width, label="Media", color="orange")
plt.bar(x + width, maximo, width, label="Máximo", color="green")

# Etiquetas del eje X con nombres descriptivos
plt.xticks(x, columnas[:-1], rotation=45)
plt.ylabel("Valor")
plt.title("Estadísticas básicas del dataset Iris")
plt.legend(title="Medidas")
plt.tight_layout()
plt.show()

# ======================================
# 3.10 Gráfico de pastel de la distribución de especies
# ======================================
print("\n=== 3.10 Gráfico de pastel de distribución de especies ===")
# Contar especies
frecuencias = iris["species"].value_counts()

# Gráfico de pastel
plt.figure(figsize=(6,6))
plt.pie(frecuencias, labels=frecuencias.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribución de especies en el dataset Iris")
plt.show()

# ======================================
# 3.11 Relación entre longitud y ancho del sépalo por especie
# ======================================
print("\n=== 3.11 Gráfico de dispersión de sépalo (longitud vs ancho) ===")

plt.figure(figsize=(8,6))

# Graficar cada especie con un color distinto
for especie, datos in iris.groupby("species"):
    plt.scatter(datos["sepal_length"], datos["sepal_width"], label=especie, alpha=0.7)

plt.xlabel("Longitud del sépalo (cm)")
plt.ylabel("Ancho del sépalo (cm)")
plt.title("Relación entre longitud y ancho del sépalo por especie")
plt.legend(title="Especie")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
