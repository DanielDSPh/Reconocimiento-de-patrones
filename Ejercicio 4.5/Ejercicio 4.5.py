import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen en escala de grises
imagen = cv2.imread('IntestinoRGB.jpg', cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print("Error al cargar la imagen.")
else:
    print("Imagen cargada correctamente.")

# Verificar dimensiones
print(f"Dimensiones de la imagen (alto, ancho): {imagen.shape}")

# Calcular estadísticas
media = np.mean(imagen)
maximo = np.max(imagen)
minimo = np.min(imagen)

# Imprimir resultados
print(f"Media de intensidades: {media:.2f}")
print(f"Valor máximo: {maximo}")
print(f"Valor mínimo: {minimo}")

# Mostrar histograma
plt.figure(figsize=(8, 4))
plt.hist(imagen.ravel(), bins=256, range=[0, 256], color='gray')
plt.title('Histograma de Intensidades')
plt.xlabel('Nivel de gris (intensidad)')
plt.ylabel('Frecuencia (número de píxeles)')
plt.grid(True)
plt.tight_layout()
plt.show()