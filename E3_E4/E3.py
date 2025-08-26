import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def procesar_imagen():
    try:
        # Verificar si la carpeta Imagenes existe
        if not os.path.exists('Imagenes'):
            print("Error: La carpeta 'Imagenes' no existe")
            return
        
        ruta_imagen = r'Imagenes\peppers_color.tif'
        
        # Verificar si el archivo existe
        if not os.path.exists(ruta_imagen):
            print(f"Error: El archivo {ruta_imagen} no existe")
            print("Verifica la ruta y el nombre del archivo")
            return
        
        # Cargar imagen en color
        img_color = cv2.imread(ruta_imagen)
        
        # Verificar si la imagen se cargó correctamente
        if img_color is None:
            print("Error: No se pudo cargar la imagen. Verifica que el archivo no esté corrupto")
            return
        
        # Convertir BGR a RGB
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        
        # Convertir a escala de grises para el recorte
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        
        # Mostrar dimensiones originales
        print(f"Dimensiones originales color: {img_color.shape}")
        print(f"Dimensiones originales gris: {img_gray.shape}")
        
        # Verificar que las coordenadas de recorte estén dentro de los límites
        alto, ancho = img_gray.shape
        
        # Coordenadas del recorte
        y_inicio, y_fin = 175, 495
        x_inicio, x_fin = 172, 424
        
        # Validar coordenadas
        if (y_inicio < 0 or y_fin > alto or x_inicio < 0 or x_fin > ancho or 
            y_inicio >= y_fin or x_inicio >= x_fin):
            print(f"Error: Coordenadas de recorte fuera de rango")
            print(f"Límites de la imagen: Alto={alto}, Ancho={ancho}")
            print(f"Coordenadas solicitadas: Y[{y_inicio}:{y_fin}], X[{x_inicio}:{x_fin}]")
            return
        
        # Recortar el área en escala de grises
        recorte_gray = img_gray[y_inicio:y_fin, x_inicio:x_fin]
        
        # Verificar que el recorte no esté vacío
        if recorte_gray.size == 0:
            print("Error: El recorte resultó en una imagen vacía")
            return
        
        # Crear carpeta si no existe para guardar la imagen
        os.makedirs('Imagenes', exist_ok=True)
        
        # Guardar la imagen b/n y recortada
        ruta_guardar = r'Imagenes\peppers_bn_r.jpg'
        exito = cv2.imwrite(ruta_guardar, recorte_gray)
        
        if exito:
            print(f"Imagen nueva guardada como: {ruta_guardar}")
            print(f"Dimensiones del recorte: {recorte_gray.shape}")
        else:
            print("Error: No se pudo guardar la imagen. Verifica los permisos de escritura")
            return
        
        # Mostrar imágenes
        plt.figure(figsize=(12, 5))
        
        # Mostrar imagen original a color
        plt.subplot(1, 2, 1)
        plt.imshow(img_color)
        plt.axis("off")
        plt.title("Imagen Original (Color)")
        
        # Mostrar recorte en blanco y negro
        plt.subplot(1, 2, 2)
        plt.imshow(recorte_gray, cmap="gray")
        plt.axis("off")
        plt.title(f"Recorte B/N: {recorte_gray.shape}")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        print("Verifica la instalación de OpenCV y matplotlib")

# Ejecutar la función principal
if __name__ == "__main__":
    procesar_imagen()