#Ejercicio 4.1.3.1
#   De las imágen "lena_Orig.jpg" "peppers_color.tif". Desarrolla un script con Sikit_Image para cambiar el espacio de color y 
#   despliegue en grises cada uno de los componentes resultantes de la imágen: 
#   RGB a HSV

from skimage import io, color   
import matplotlib.pyplot as plt

#Cargamos las imagenes
lena = io.imread(r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\LenaOrig.jpg")
peppers = io.imread(r"C:\Users\Rixde\Documents\ReconPatrones_ws\reconpatrones2026-1\Pracitcas\P0\Imagenes\peppers_color2.jpg")

#Convertimos a HSV
lena_hsv = color.rgb2hsv(lena)
peppers_hsv = color.rgb2hsv(peppers)

#Separamos canales
lena_h = lena_hsv[:, :, 0]
lena_s = lena_hsv[:, :, 1]
lena_v = lena_hsv[:, :, 2]

peppers_h = peppers_hsv[:, :, 0]
peppers_s = peppers_hsv[:, :, 1]
peppers_v = peppers_hsv[:, :, 2]

plt.figure(figsize=(10, 10))

#Mostramos canales de Lena
plt.subplot(2,3,1)
plt.imshow(lena_h, cmap='gray')
plt.title("Lena - H")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(lena_s, cmap='gray')
plt.title("Lena - S")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(lena_v, cmap='gray')
plt.title("Lena - V")
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