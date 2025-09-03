# ---------------------------------------
# Análisis de la base de datos Wine Quality
# ---------------------------------------
from matplotlib import colors
import pandas as pd #se utiliza la lib 
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# ---------------------------------------
# 3.9 Cree una gráfica de barras que muestre la media, mínimo y máximo de todos los datos
# ---------------------------------------

media = desc_wine.loc["mean"]
minimo = desc_wine.loc["min"]
maximo = desc_wine.loc["max"]

x = np.arange(len(media))
width = 0.25

plt.figure(figsize=(12,6))
plt.bar(x - width, minimo, width, label="Mínimo", color="#8ecae6")
plt.bar(x, media, width, label="Media", color="#b388eb")
plt.bar(x + width, maximo, width, label="Máximo", color="#90be6d")

plt.xticks(x, desc_wine.columns, rotation=45)
plt.ylabel("Valor (escala logarítmica)")
plt.title("Estadísticas básicas del dataset Wine")
plt.yscale("log")   # Escala logarítmica
plt.legend(title="Medidas")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 3.10 Muestre la frecuencia de la calidad "Quality" como una gráfica pastel
# ---------------------------------------
frecuencia_calidad = wine["quality"].value_counts().sort_index()

# Colores
colores = [
    "#8ecae6", "#b388eb", "#90be6d", "#219ebc",
    "#ffafcc", "#a2d2ff", "#cdb4db", "#d0f4de"
]

# Mover solo la etiqueta de calidad 3 para evitar solapamientos
def func_autopct(pct, all_vals):
    total = sum(all_vals)
    value = int(round(pct*total/100.0))
    if value == frecuencia_calidad.loc[3]:
        return f"\n\n\n{pct:.1f}%"
    else:
        return f"{pct:.1f}%"

plt.figure(figsize=(8,8))
plt.pie(frecuencia_calidad, 
        labels=frecuencia_calidad.index, 
        autopct=lambda pct: func_autopct(pct, frecuencia_calidad),
        startangle=90, 
        colors=colores)

plt.title("Distribución de la calidad del vino")
plt.axis("equal")
plt.show()

# ---------------------------------------
# 3.11 Cree una gráfica que muestre la relación entre sulfatos y calidad
# ---------------------------------------

plt.figure(figsize=(10,6))

for quality, datos in wine.groupby("quality"):
    plt.scatter(datos["sulphates"], datos["quality"], label=f"Quality {quality}", color=colores[quality-1])

plt.title("Relación entre Sulfatos y Calidad del Vino")
plt.xlabel("Sulfatos")
plt.ylabel("Calidad")
plt.legend()
plt.show()

# ---------------------------------------
# 3.12 Obtenga los histogramas de las variables fixed acidity, volatile acidity, alcohol y density
# ---------------------------------------

plt.figure(figsize=(12,10))
for i, col in enumerate(["fixed acidity", "volatile acidity", "alcohol", "density"], 1):
    plt.subplot(2, 2, i)
    sns.histplot(wine[col], bins=30, kde=True, color=colores[i-1])
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

# ---------------------------------------
# 3.13 Cree gráficas de dispersión usando pairplot y muestre con colores la calidad de los vinos
# ---------------------------------------
colores2 = [
    "#8ecae6", "#b388eb", "#90be6d", "#219ebc",
    "#ffafcc", "#a2d2ff"
]

# Se seleccionan columnas clave (son demasiadas variables, se reducirán para mayor legibilidad)
columnas_clave = ["alcohol", "volatile acidity", "sulphates", "citric acid", "density", "quality"]

sns.pairplot(wine[columnas_clave], hue="quality", palette=colores2)
plt.suptitle("Gráficas de dispersión: variables más relevantes", y=1.02)
plt.show()

# ---------------------------------------
# 3.14 Cree una gráfica usando joinplot para mostrar la dispersión entre la calidad y sulfatos y las distribuciones de estas
# dos variables
# ---------------------------------------
sns.jointplot(data=wine, x="sulphates", y="quality", kind="scatter", color="#219ebc")
plt.suptitle("Dispersión entre Calidad y Sulfatos", y=1.02)
plt.show()

# ---------------------------------------
# 3.15 Repita el ejercicio anterior con kind = "hex"
# ---------------------------------------
sns.jointplot(data=wine, x="sulphates", y="quality", kind="hex", color="#219ebc")
plt.suptitle("Dispersión entre Calidad y Sulfatos (Hex)", y=1.02)
plt.show()