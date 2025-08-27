#Ejercicio 4.1.3.1
#   De las imágen "lena_Orig.jpg" "peppers_color.tif". Desarrolla un script con Sikit_Image para cambiar el espacio de color y 
#   despliegue en grises cada uno de los componentes resultantes de la imágen: 
#   RGB a HUV

from skimage import io, color   
import matplotlib.pyplot as plt

#Cargamos las imagenes
lena = io.imread(r"Imagenes\LenaOrig.jpg")
peppers = io.imread(r"Imagenes\peppers_color2.jpg")

#Convertimos a YUV
lena_yuv = color.rgb2yuv(lena)
peppers_yuv = color.rgb2yuv(peppers)

#Separamos canales
lena_y = lena_yuv[:, :, 0]
lena_u = lena_yuv[:, :, 1]
lena_v = lena_yuv[:, :, 2]

peppers_y = peppers_yuv[:, :, 0]
peppers_u = peppers_yuv[:, :, 1]
peppers_v = peppers_yuv[:, :, 2]

plt.figure(figsize=(10, 10))

#Mostramos canales de Lena
plt.subplot(2,3,1)
plt.imshow(lena_y, cmap='gray')
plt.title("Lena - Y")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(lena_u, cmap='gray')
plt.title("Lena - U")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(lena_v, cmap='gray')
plt.title("Lena - V")
plt.axis('off')

# Mostramos los canales de la imagen de Peppers
plt.subplot(2, 3, 4)
plt.imshow(peppers_y, cmap='gray')
plt.title('Peppers - Y')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(peppers_u, cmap='gray')
plt.title('Peppers - U')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(peppers_v, cmap='gray')
plt.title('Peppers - V')
plt.axis('off')

plt.tight_layout()
plt.show()