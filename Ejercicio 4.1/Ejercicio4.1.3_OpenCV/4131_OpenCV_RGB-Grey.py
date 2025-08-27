#Ejercicio 4.1.3.1
#   De las imágen "lena_Orig.jpg" "peppers_color.tif". Desarrolla un script con OpenCV para cambiar el espacio de color y 
#   despliegue en grises cada uno de los componentes resultantes de la imágen: 
#   RGB a escala de grises

import cv2
import matplotlib.pyplot as plt

#Ruta de las imagenes
lena = r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\2.jpg"
peppers = r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\peppers_color.tif"

#Cargamos imagenes
lena_img = cv2.imread(lena)
peppers_img = cv2.imread(peppers)

#Separamos los canales de las imagenes
lena_b, lena_g, lena_r = cv2.split(lena_img)
peppers_b, peppers_g, peppers_r = cv2.split(peppers_img)

plt.figure(figsize=(10, 10))

# Mostramos los canales de la imagen de Lena
plt.subplot(2, 3, 1)
plt.imshow(lena_r, cmap='gray')
plt.title('Lena - Rojo')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(lena_g, cmap='gray')
plt.title('Lena - Verde')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(lena_b, cmap='gray')
plt.title('Lena - Azul')
plt.axis('off')

# Mostramos los canales de la imagen de Peppers
plt.subplot(2, 3, 4)
plt.imshow(peppers_r, cmap='gray')
plt.title('Peppers - Rojo')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(peppers_g, cmap='gray')
plt.title('Peppers - Verde')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(peppers_b, cmap='gray')
plt.title('Peppers - Azul')
plt.axis('off')

plt.tight_layout()
plt.show()