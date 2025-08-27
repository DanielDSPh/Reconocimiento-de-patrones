import numpy as np
import cv2
import matplotlib.pyplot as plt

# Configuración de la imagen
filename = r"Imagenes\rosa800x600.raw"
height = 800    # filas
width = 600     # columnas
dtype = np.uint8  # integer8 (8 bits sin signo)

# Leer la imagen RAW
try:
    # Leer los datos binarios
    with open(filename, 'rb') as f:
        raw_data = f.read()
    
    # Convertir a array de numpy
    image_array = np.frombuffer(raw_data, dtype=dtype)
    
    image = image_array.reshape((height, width))
    
    print(f"Imagen cargada correctamente")
    print(f"Dimensiones: {image.shape}")
    print(f"Tipo de datos: {image.dtype}")
    print(f"Valores mínimo: {image.min()}, máximo: {image.max()}")
    
    # Mostrar la imagen
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')  # Usar cmap='gray' para imágenes en escala de grises
    plt.title('Imagen RAW - rosa800x600')
    plt.axis('off')
    plt.show()

except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{filename}'")
except Exception as e:
    print(f"Error al procesar la imagen: {e}")