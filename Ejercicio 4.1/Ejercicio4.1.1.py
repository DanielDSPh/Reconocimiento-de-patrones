#Ejercicio 4.1.1
#Desarrolla un script para leer y desplegar cada imagen con los paquetes de Matplotlib, OpenCV, Sikit-Image y PIL

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage import io
from PIL import Image

# Ruta de la imagen
ruta = r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\LenaOrig.jpg"

# 1. Matplotlib
img1 = mpimg.imread(ruta)
plt.imshow(img1, cmap="gray")
plt.title("Matplotlib")
plt.show()

# 2. OpenCV
img2 = cv2.imread(ruta)                       # OpenCV lee en BGR
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # Convertir a RGB
plt.imshow(img2)
plt.title("OpenCV")
plt.show()

# 3. Scikit-Image
img3 = io.imread(ruta)
plt.imshow(img3, cmap="gray")
plt.title("Scikit-Image")
plt.show()

# 4. PIL
img4 = Image.open(ruta)
plt.imshow(img4, cmap="gray")
plt.title("PIL")
plt.show()
