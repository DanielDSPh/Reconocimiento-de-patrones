#Ejercicio 4.1.2
#Imprimir el tipo de imagen, el tamaño y el tipo de dato

from PIL import Image
import cv2
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ruta de la imagen
ruta = r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\LenaOrig.jpg"

# ===== Matplotlib =====
imagen_matplotlib = mpimg.imread(ruta)
print("===== Matplotlib =====")
print("Tipo de imagen:", type(imagen_matplotlib))
print("Tamaño (alto, ancho, canales):", imagen_matplotlib.shape)
print("Tipo de dato:", imagen_matplotlib.dtype)
print()

# ===== PIL / Pillow =====
imagen_pil = Image.open(ruta)
print("===== PIL / Pillow =====")
print("Tipo de imagen:", type(imagen_pil))
print("Tamaño (ancho, alto):", imagen_pil.size)
print("Modo de color / tipo de dato:", imagen_pil.mode)
print()

# ===== OpenCV =====
imagen_cv = cv2.imread(ruta)
print("===== OpenCV =====")
print("Tipo de imagen:", type(imagen_cv))
print("Tamaño (alto, ancho, canales):", imagen_cv.shape)
print("Tipo de dato:", imagen_cv.dtype)
print()

# ===== Scikit-Image =====
imagen_sk = io.imread(ruta)
print("===== Scikit-Image =====")
print("Tipo de imagen:", type(imagen_sk))
print("Tamaño (alto, ancho, canales):", imagen_sk.shape)
print("Tipo de dato:", imagen_sk.dtype)
print()