import cv2
import os
import matplotlib.pyplot as plt

# Función para mostrar imagen con Matplotlib
def mostrar_imagen_matplotlib(imagen, titulo="Imagen"):
    # Convertir de BGR (OpenCV) a RGB (Matplotlib)
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Rutas
imagen_nombre = r"Imagenes\peppers_color2.jpg"
carpeta_salida = r"Imagenes\Resultados"
os.makedirs(carpeta_salida, exist_ok=True)

# Leer imagen
img = cv2.imread(imagen_nombre)
if img is None:
    print("No se pudo leer la imagen. Revisa el nombre o la ruta.")
    exit()

h, w = img.shape[:2]

# 4.2.1 Reajustar a 5 veces
img_5x = cv2.resize(img, (w*5, h*5), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(carpeta_salida, "imagen_5x.png"), img_5x)
mostrar_imagen_matplotlib(img_5x, "Imagen 5x°")

# 4.2.2 Reajustar a 3 veces
img_3x = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(carpeta_salida, "imagen_3x.png"), img_3x)
mostrar_imagen_matplotlib(img_3x, "Imagen 3x°")

# 4.2.3 Rotaciones
# Rotación 45 grados
M45 = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0)
rot45 = cv2.warpAffine(img, M45, (w, h))
cv2.imwrite(os.path.join(carpeta_salida, "imagen_rot45.png"), rot45)
mostrar_imagen_matplotlib(rot45, "Rotación 45°")

# Rotación 90 grados
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(os.path.join(carpeta_salida, "imagen_rot90.png"), rot90)
mostrar_imagen_matplotlib(rot90, "Rotación 90°")

# Rotación 180 grados
rot180 = cv2.rotate(img, cv2.ROTATE_180)
cv2.imwrite(os.path.join(carpeta_salida, "imagen_rot180.png"), rot180)
mostrar_imagen_matplotlib(rot180, "Rotación 180°")

print("✅ Punto 4.2 terminado. Imágenes guardadas en la carpeta:", carpeta_salida)
