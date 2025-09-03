# practica1.py
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns # Necesitas instalar esta libreria para las nuevas visualizaciones
from PIL import Image # Necesitas instalar esta libreria para el analisis de imagenes
import cv2 # Necesitas esta libreria para procesar la imagen de manera mas eficaz

# ======================================
# 3.1 Cargar los datos Iris
# ======================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
directorio_local = "iris.data"

# Carga los datos desde un archivo local, si no existe, lo descarga
try:
    iris = pd.read_csv(directorio_local, header=None)
except FileNotFoundError:
    print(f"El archivo '{directorio_local}' no se encontro localmente. Intentando descargar desde la URL...")
    iris = pd.read_csv(url, header=None)
    iris.to_csv(directorio_local, index=False, header=False)
    print(f"Archivo '{directorio_local}' descargado y guardado localmente.")

print("=== 3.1 Informacion general del DataFrame ===")
print("Forma (filas, columnas):", iris.shape)
print("\nTipos de datos:\n", iris.dtypes)
print("\nPrimeras 10 filas:\n", iris.head(10))


# ======================================
# 3.2 Imprimir llaves y numero de filas y columnas
# ======================================
print("\n=== 3.2 Llaves y dimensiones ===")
print("Llaves:", iris.keys().tolist())
print("Numero de filas y columnas:", iris.shape)


# ======================================
# 3.3 Numero de muestras faltantes o NaN
# ======================================
print("\n=== 3.3 Valores faltantes (NaN) ===")
print("NaN por columna:\n", iris.isnull().sum())
print("Total de NaN en todo el DataFrame:", iris.isnull().sum().sum())


# ======================================
# 3.4 Crear matriz 5x5 identidad y convertir a dispersa CRS
# ======================================
print("\n=== 3.4 Matriz 5x5 e implementacion dispersa ===")
A = np.eye(5) # matriz identidad 5x5
print("Matriz 5x5:\n", A)

# Convertir a formato disperso CRS
A_sparse = csr_matrix(A)
print("\nMatriz dispersa en formato CRS:\n", A_sparse)

# Mostrar detalles del CRS
print("\nValores no cero:", A_sparse.data)
print("Indices de columna:", A_sparse.indices)
print("Indices de fila (indptr):", A_sparse.indptr)

# ======================================
# 3.5 Estadisticas basicas: media y desviacion estandar
# ======================================
print("\n=== 3.5 Estadisticas basicas ===")
desc = iris.describe()
print("Media:\n", desc.loc["mean"])
print("\nDesviacion estandar:\n", desc.loc["std"])

# ======================================
# 3.6 Numero de muestras por clase
# ======================================
print("\n=== 3.6 Numero de muestras por clase ===")
print(iris[4].value_counts())

# ======================================
# 3.7 Anadir encabezados y repetir conteo por clase
# ======================================
columnas = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris.columns = columnas

print("\n=== 3.7 Con encabezados anadidos ===")
print("Nombres de columnas:", iris.columns.tolist())
print("\nNumero de muestras por clase con encabezado:")
print(iris["species"].value_counts())

# ======================================
# 3.8 Mostrar las primeras 10 filas y 2 primeras columnas
# ======================================
print("\n=== 3.8 Primeras 10 filas y 2 primeras columnas ===")
print(iris.iloc[:10, :2])

# ======================================
# 3.9 Grafico de barras de minimo, media y maximo
# ======================================
print("\n=== 3.9 Grafico de barras de estadisticas ===")
desc = iris.describe()
media = desc.loc["mean"]
minimo = desc.loc["min"]
maximo = desc.loc["max"]

numeric_cols = columnas[:-1]
x = np.arange(len(numeric_cols))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, minimo[numeric_cols], width, label="Minimo", color="skyblue")
plt.bar(x, media[numeric_cols], width, label="Media", color="orange")
plt.bar(x + width, maximo[numeric_cols], width, label="Maximo", color="green")

plt.xticks(x, numeric_cols, rotation=45, ha="right")
plt.ylabel("Valor")
plt.title("Ejercicio 3.9: Estadisticas basicas de las caracteristicas de Iris")
plt.legend(title="Medidas")
plt.tight_layout()
plt.show()

# ======================================
# 3.10 Grafico de pastel de la distribucion de especies
# ======================================
print("\n=== 3.10 Grafico de pastel de distribucion de especies ===")
frecuencias = iris["species"].value_counts()

plt.figure(figsize=(7,7))
plt.pie(frecuencias, labels=frecuencias.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title("Ejercicio 3.10: Distribucion de especies en el dataset Iris")
plt.ylabel('')
plt.tight_layout()
plt.show()

# ======================================
# 3.11 Relacion entre longitud y ancho del sepalo por especie
# ======================================
print("\n=== 3.11 Grafico de dispersion de sepalo (longitud vs ancho) ===")
plt.figure(figsize=(9,7))
sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species", s=100, alpha=0.7)
plt.xlabel("Longitud del sepalo (cm)")
plt.ylabel("Ancho del sepalo (cm)")
plt.title("Ejercicio 3.11: Relacion entre longitud y ancho del sepalo por especie")
plt.legend(title="Especie")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ======================================
# 3.12 Histogramas de las variables
# ======================================
print("\n=== 3.12 Histogramas de las caracteristicas ===")
plt.figure(figsize=(12, 10))
for i, col in enumerate(columnas[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(data=iris, x=col, kde=True, bins=15, color='purple')
    plt.title(f'Ejercicio 3.12: Histograma de {col.replace("_", " ").title()}')
    plt.xlabel(col.replace("_", " ").title())
    plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()

# ======================================
# 3.13 Graficas de dispersion (pairplot)
# ======================================
print("\n=== 3.13 Graficas de dispersion (pairplot) por especie ===")
sns.pairplot(iris, hue="species", palette="viridis")
plt.suptitle("Ejercicio 3.13: Pairplot de las caracteristicas de Iris por especie", y=1.02)
plt.tight_layout()
plt.show()

# ======================================
# 3.14 Grafica conjunta (jointplot)
# ======================================
print("\n=== 3.14 Grafica conjunta (jointplot) de longitud vs ancho del sepalo ===")
sns.jointplot(data=iris, x="sepal_length", y="sepal_width", kind="scatter", palette="bright", height=7)
plt.suptitle("Ejercicio 3.14: Dispersion y distribucion marginal de longitud vs ancho del sepalo", y=1.02)
plt.tight_layout()
plt.show()

# ======================================
# 3.15 Grafica conjunta (jointplot) con kind="hex"
# ======================================
print("\n=== 3.15 Grafica conjunta (jointplot) hexagonal de longitud vs ancho del sepalo ===")
sns.jointplot(data=iris, x="sepal_length", y="sepal_width", kind="hex", color="#4CB391", height=7)
plt.suptitle("Ejercicio 3.15: Dispersion hexagonal y distribucion marginal de longitud vs ancho del sepalo", y=1.02)
plt.tight_layout()
plt.show()


# ======================================
# 3.16 Analisis de zona morada del petalo en una imagen (conceptual)
# ======================================
print("\n=== 3.16 Analisis de zona morada del petalo en una imagen (conceptual) ===")

def analizar_zona_morada(ruta_imagen):
    """
    Analiza una zona morada en una imagen para obtener su valor promedio
    y desviacion estandar.
    
    Argumentos:
    ruta_imagen (str): La ruta del archivo de imagen a procesar.
    
    Retorna:
    tuple: Un tuple con el valor promedio y la desviacion estandar, o None si falla.
    """
    try:
        # Paso 1: Cargar la imagen
        imagen_rgb = cv2.imread(ruta_imagen)
        if imagen_rgb is None:
            print(f"[ERROR]: No se pudo cargar la imagen de la ruta: {ruta_imagen}")
            return None
        
        # Paso 2: Convertir la imagen a un espacio de color adecuado (HSV)
        imagen_hsv = cv2.cvtColor(imagen_rgb, cv2.COLOR_BGR2HSV)
        
        # Paso 3: Definir un rango de color para el morado en HSV
        # Estos valores son un rango tipico para el morado, pero pueden requerir ajuste
        # segun la iluminacion de la imagen.
        rango_morado_bajo = np.array([125, 50, 50])
        rango_morado_alto = np.array([150, 255, 255])
        
        # Paso 4: Crear una mascara binaria para aislar la zona morada
        mascara = cv2.inRange(imagen_hsv, rango_morado_bajo, rango_morado_alto)
        
        # Paso 5: Aplicar la mascara a la imagen original
        # Esto extrae solo los pixeles que son morados.
        zona_morada = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=mascara)
        
        # Paso 6: Extraer los valores de los pixeles de la zona morada
        # Convertimos la zona morada a escala de grises para tener un solo valor por pixel
        zona_morada_gris = cv2.cvtColor(zona_morada, cv2.COLOR_BGR2GRAY)
        
        # Obtenemos los valores de los pixeles que no son negros (los de la mascara)
        valores_pixeles = zona_morada_gris[zona_morada_gris > 0]
        
        if valores_pixeles.size > 0:
            # Paso 7: Calcular el valor promedio
            promedio = np.mean(valores_pixeles)
            
            # Paso 8: Calcular la desviacion estandar
            desviacion_estandar = np.std(valores_pixeles)
            
            return promedio, desviacion_estandar
        else:
            print("[ADVERTENCIA]: No se encontraron pixeles morados en la imagen.")
            return None
            
    except Exception as e:
        print(f"[ERROR]: Ocurrio un error al procesar la imagen: {e}")
        return None

# --- EJECUTAR EL PROGRAMA ---
# El nombre de la imagen se ha actualizado a 'iris.png'
ruta_de_mi_imagen = 'iris.png'

print(f"=== Analizando la imagen en: {ruta_de_mi_imagen} ===")
resultados = analizar_zona_morada(ruta_de_mi_imagen)

if resultados:
    promedio, desviacion_estandar = resultados
    print(f"El valor promedio de la zona morada es: {promedio:.2f}")
    print(f"La desviacion estandar de la zona morada es: {desviacion_estandar:.2f}")
else:
    print("No se pudieron obtener estadisticas de la imagen.")