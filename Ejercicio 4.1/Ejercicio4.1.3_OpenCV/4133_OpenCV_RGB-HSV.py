#Ejercicio 4.1.3.2
#   De las imágen "lena_Orig.jpg" "peppers_color.tif". Desarrolla un script con OpenCV para cambiar el espacio de color y 
#   despliegue en grises cada uno de los componentes resultantes de la imágen: 
#   RGB a HSV

import cv2
import matplotlib.pyplot as plt

#Ruta de las imagenes
lena = r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\LenaOrig.jpg"
peppers = r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\peppers_color.tif"

#Cargamos imagenes
lena_img = cv2.cvtColor(cv2.imread(lena), cv2.COLOR_BGR2HSV)
peppers_img = cv2.cvtColor(cv2.imread(peppers), cv2.COLOR_BGR2HSV)

#Separamos los canales de las imagenes
lena_h, lena_s, lena_v = cv2.split(lena_img)
peppers_h, peppers_s, peppers_v = cv2.split(peppers_img)

plt.figure(figsize=(10, 10))

# Mostramos los canales de la imagen de Lena
plt.subplot(2, 3, 1)
plt.imshow(lena_h, cmap='gray')
plt.title('Lena - H')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(lena_s, cmap='gray')
plt.title('Lena - S')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(lena_v, cmap='gray')
plt.title('Lena - V')
plt.axis('off')

# Mostramos los canales de la imagen de Peppers
plt.subplot(2, 3, 4)
plt.imshow(peppers_h, cmap='gray')
plt.title('Peppers - H')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(peppers_s, cmap='gray')
plt.title('Peppers - S')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(peppers_v, cmap='gray')
plt.title('Peppers - V')
plt.axis('off')

plt.tight_layout()
plt.show()