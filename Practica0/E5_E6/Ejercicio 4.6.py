import cv2
import matplotlib.pyplot as plt

# Cargar imagen original en color
lena_color = cv2.imread('Imagenes/LenaOrig.jpg')

if lena_color is None:
    print("Error al cargar la imagen.")
else:
    print("Imagen cargada correctamente.")

    # Conversión a escala de grises
    lena_gray = cv2.cvtColor(lena_color, cv2.COLOR_BGR2GRAY)

    # Aplicación del filtro Gaussiano
    lena_gauss = cv2.GaussianBlur(lena_gray, (5, 5), 1.0)

    # Aplicación del filtro Laplaciano
    lena_laplace = cv2.Laplacian(lena_gray, cv2.CV_64F)
    lena_laplace = cv2.convertScaleAbs(lena_laplace)

    # Visualización de resultados
    titles = ['Escala de grises', 'Filtro Gaussiano', 'Filtro Laplaciano']
    images = [lena_gray, lena_gauss, lena_laplace]

    plt.figure(figsize=(13, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()