import cv2 #Librebría para procesar imágenes
import os #librería para crear carpetas en el sistema de archivos


# 4.2 Escalado y rotaciones


# Definir rutas para entrada y salida de imagen
imagen_nombre = r"D:\Universidad\Reconocimiento de patrones\Imagenes\peppers_color2.jpg"
# se usa 'r' para que python pueda interpretar los \
carpeta_salida = r"D:\Universidad\Reconocimiento de patrones\Imagenes\Resultados"
os.makedirs(carpeta_salida, exist_ok=True) #crea la carpeta si no existe

img = cv2.imread(imagen_nombre)# Leer imagen
if img is None:
    print("No se pudo leer la imagen. Revisa el nombre o la ruta.")
    exit() #termina la ejecución si no se encuentra la imagen 


h, w = img.shape[:2] #se guarda alto y ancho de la imagen original

# 4.2.1 Reajustar a 5 veces
img_5x = cv2.resize(img, (w*5, h*5), interpolation=cv2.INTER_CUBIC) #interpolación bicúbica para mantener calidad
cv2.imwrite(os.path.join(carpeta_salida, "imagen_5x.png"), img_5x) #guarda la imagen escalada
cv2.imshow("Imagen 5x°", img_5x)
cv2.waitKey(0)      # Espera hasta que se presione una tecla
cv2.destroyAllWindows()  # Cerrar la ventana


# 4.2.2 Reajustar a 3 veces
img_3x = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(carpeta_salida, "imagen_3x.png"), img_3x)
cv2.imshow("Imagen 3x°", img_3x)
cv2.waitKey(0)      
cv2.destroyAllWindows()  

# 4.2.3 Rotaciones
# Rotación 45 grados
M45 = cv2.getRotationMatrix2D((w//2, h//2), 45, 1.0) #matriz de transformación para rotar imagen 
#centro de la imagen, angulo de rotacion en grados, factor de escala
rot45 = cv2.warpAffine(img, M45, (w, h)) #aplica la rotacion a la imagen 
cv2.imwrite(os.path.join(carpeta_salida, "imagen_rot45.png"), rot45)
cv2.imshow("Rotación 45°", rot45) #abre una ventana mostrando la imagen 
cv2.waitKey(0)      # Espera hasta que presiones una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas

# Rotación 90 grados
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) #rotar la imagen 
cv2.imwrite(os.path.join(carpeta_salida, "imagen_rot90.png"), rot90)
cv2.imshow("Rotación 90°", rot90)
cv2.waitKey(0)      
cv2.destroyAllWindows() 

# Rotación 180 grados
rot180 = cv2.rotate(img, cv2.ROTATE_180)
cv2.imwrite(os.path.join(carpeta_salida, "imagen_rot180.png"), rot180)
cv2.imshow("Rotación 180°", rot180)
cv2.waitKey(0)     
cv2.destroyAllWindows()  


print("✅ Punto 4.2 terminado. Imágenes guardadas en la carpeta:", carpeta_salida)


