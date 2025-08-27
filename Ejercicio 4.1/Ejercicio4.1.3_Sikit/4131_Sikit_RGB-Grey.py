#Ejercicio 4.1.3.1
#   De las imágen "lena_Orig.jpg" "peppers_color.tif". Desarrolla un script con Sikit_Image para cambiar el espacio de color y 
#   despliegue en grises cada uno de los componentes resultantes de la imágen: 
#   RGB a escala de grises

from skimage import io
import matplotlib.pyplot as plt

#Cargamos las imagenes
lena = io.imread(r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\LenaOrig.jpg")
peppers = io.imread(r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\peppers_color2.jpg")

#Separamos canales
lena_r = lena[:, :, 0]
lena_g = lena[:, :, 1]
lena_b = lena[:, :, 2]

peppers_r = peppers[:, :, 0]
peppers_g = peppers[:, :, 1]
peppers_b = peppers[:, :, 2]

plt.figure(figsize=(10, 10))

#Mostramos canales de Lena
plt.subplot(2,3,1)
plt.imshow(lena_r, cmap='gray')
plt.title("Lena - R")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(lena_g, cmap='gray')
plt.title("Lena - G")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(lena_b, cmap='gray')
plt.title("Lena - B")
plt.axis('off')

# Mostramos los canales de la imagen de Peppers
plt.subplot(2, 3, 4)
plt.imshow(peppers_r, cmap='gray')
plt.title('Peppers - R')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(peppers_g, cmap='gray')
plt.title('Peppers - G')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(peppers_b, cmap='gray')
plt.title('Peppers - B')
plt.axis('off')

plt.tight_layout()
plt.show()