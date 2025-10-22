import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. PARÁMETROS ENTRENADOS (los que obtuviste al combinar tus datos) ---
parametros = {
    'melanoma': {
        'media': np.array([76.21715705, 80.64092831, 110.72321576]),
        'cov': np.array([
            [1256.1255113, 1305.17839277, 1212.07415754],
            [1305.17839277, 1445.01930646, 1345.5722495],
            [1212.07415754, 1345.5722495, 1393.24298536]
        ]),
        'prior': 0.18825612,
        'color': (255, 255, 255)  # BGR
    },
    'fondo': {
        'media': np.array([179.60358171, 179.07305983, 179.10867199]),
        'cov': np.array([
            [337.278824, 341.9158843, 300.6349299],
            [341.9158843, 355.65817477, 310.17293123],
            [300.6349299, 310.17293123, 280.23933683]
        ]),
        'prior': 0.81174388,
        'color': (0, 0, 0)
    }
}

# --- 2. FUNCIONES AUXILIARES ---
def probabilidad_gaussiana(x, media, cov):
    k = len(media)
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    factor = 1 / np.sqrt((2 * np.pi) ** k * det_cov)
    diff = x - media
    exp_term = np.exp(-0.5 * np.dot(np.dot(diff, cov_inv), diff.T))
    return factor * exp_term

# --- 3. CARGAR IMAGEN DE PRUEBA ---
ruta_imagen = "Prueba1.jpg"  # Cambia por el nombre de tu imagen
img = cv2.imread(ruta_imagen)
if img is None:
    raise FileNotFoundError(f"No se encontró la imagen en la ruta: {ruta_imagen}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
alto, ancho, _ = img.shape
print(f"Imagen cargada: {ancho}x{alto}px")

# --- 4. CLASIFICACIÓN PIXEL A PIXEL ---
salida = np.zeros_like(img)
for y in range(alto):
    for x in range(ancho):
        pixel = img[y, x].astype(np.float64)

        probs = {}
        for clase, params in parametros.items():
            p_x_c = probabilidad_gaussiana(pixel, params['media'], params['cov']) * params['prior']
            probs[clase] = p_x_c

        clase_ganadora = max(probs, key=probs.get)
        salida[y, x] = parametros[clase_ganadora]['color']

# --- 5. MOSTRAR Y GUARDAR RESULTADOS ---
# Convertir la imagen clasificada a RGB para mostrar con matplotlib
salida_rgb = cv2.cvtColor(salida, cv2.COLOR_BGR2RGB)

# Mostrar resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Imagen Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(salida_rgb)
plt.title("Clasificación Bayesiana - Melanoma")
plt.axis("off")

plt.tight_layout()
plt.show()

# Guardar la imagen clasificada
cv2.imwrite("resultado_clasificacion.jpg", salida)
print("Imagen clasificada guardada como 'resultado_clasificacion.jpg'")
