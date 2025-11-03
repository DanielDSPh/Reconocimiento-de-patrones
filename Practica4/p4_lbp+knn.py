import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern      # Biblioteca para extraer descriptores LBP
from sklearn.neighbors import KNeighborsClassifier    # Clasificador k-NN de scikit-learn
import imageio.v2 as imageio
import os

# ===========================================================
# CONFIGURACIÓN INICIAL
# ===========================================================
# Rutas de las 7 texturas de entrenamiento (una por clase)
textures = [
    "training/D6.bmp",
    "training/D16.bmp",
    "training/D46.bmp",
    "training/D49.bmp",
    "training/D64.bmp",
    "training/D101.bmp",
    "training/Piedras.jpg",
]

# Imagen compuesta que contiene todas las texturas mezcladas
composite_path = "ComposicionJoseAntonio1181x1193.png"

# Carpeta donde se guardarán los resultados (máscaras y preview)
OUT_DIR = "lbp_hist_knn_7"
os.makedirs(OUT_DIR, exist_ok=True)

# ===========================================================
# PARÁMETROS DEL ALGORITMO LBP
# ===========================================================
# P  : número de vecinos considerados
# R  : radio de la vecindad
# METHOD: 'uniform' produce patrones más estables (P+2 posibles valores)
P, R, METHOD = 8, 1, "uniform"
N_BINS = P + 2      # número de bins del histograma LBP

# ===========================================================
# PARÁMETROS DE SEGMENTACIÓN Y CLASIFICACIÓN
# ===========================================================
PATCH = 16              # tamaño del parche analizado (en píxeles)
STRIDE = 4              # desplazamiento entre parches (solape)
SAMPLES_PER_TEXTURE = 500   # número de parches de entrenamiento por textura
NUM_CLASSES = 7             # cantidad de clases (texturas distintas)

# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================

def to_gray(path):
    """Abre una imagen, la convierte a escala de grises y la devuelve como arreglo numpy."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def lbp_hist(gray):
    """
    Aplica LBP a una imagen en escala de grises y devuelve su histograma normalizado.
    local_binary_pattern() es una función de scikit-image que genera una imagen
    donde cada píxel representa el patrón binario local de su vecindario.
    """
    lbp = local_binary_pattern(gray, P=P, R=R, method=METHOD)
    # Histograma de frecuencias normalizado (density=True)
    hist, _ = np.histogram(lbp, bins=N_BINS, range=(0, N_BINS), density=True)
    return hist.astype(np.float32)

def sample_patch_coords(h, w, patch):
    """Genera coordenadas aleatorias (y, x) para extraer un parche dentro de una imagen."""
    y = np.random.randint(0, h - patch + 1)
    x = np.random.randint(0, w - patch + 1)
    return y, x

# ===========================================================
# 1) ENTRENAMIENTO CON HISTOGRAMAS DE PARCHES
# ===========================================================
# Se toman parches aleatorios de cada textura y se calcula su histograma LBP.
# Estos histogramas servirán como características de entrada (X_train) al clasificador k-NN.

X_train, y_train = [], []
for cls, path in enumerate(textures, start=1):
    g = to_gray(path)
    h, w = g.shape
    for _ in range(SAMPLES_PER_TEXTURE):
        y, x = sample_patch_coords(h, w, PATCH)
        patch = g[y:y+PATCH, x:x+PATCH]
        X_train.append(lbp_hist(patch))
        y_train.append(cls)

# Se convierten las listas a arreglos numpy para alimentar al clasificador
X_train = np.vstack(X_train)
y_train = np.array(y_train, dtype=np.int32)

# ===========================================================
# 2) SEGMENTACIÓN DE LA IMAGEN COMPUESTA CON PARCHES SOLAPADOS
# ===========================================================
# Se recorre la imagen por bloques solapados y cada parche se clasifica
# usando el modelo k-NN. Cada píxel recibe “votos” de los parches que lo incluyen.

gc = to_gray(composite_path)
H, W = gc.shape
votes = np.zeros((H, W, NUM_CLASSES), dtype=np.float32)

# Se crea el clasificador k-NN con 3 vecinos
# weights='distance' hace que los vecinos más cercanos tengan mayor influencia
clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
clf.fit(X_train, y_train)  # Entrenamiento

# Recorremos la imagen por bloques con solape (stride < patch)
for y in range(0, H - PATCH + 1, STRIDE):
    for x in range(0, W - PATCH + 1, STRIDE):
        patch = gc[y:y+PATCH, x:x+PATCH]
        hvec = lbp_hist(patch).reshape(1, -1)  # histograma en forma de vector fila

        # predict_proba devuelve la probabilidad de pertenecer a cada clase
        # (vector de longitud NUM_CLASSES)
        proba = clf.predict_proba(hvec)[0]

        # Se suman las probabilidades en la región del parche (votación suave)
        votes[y:y+PATCH, x:x+PATCH, :] += proba[None, None, :]

# Se asigna a cada píxel la clase con mayor voto acumulado
labels_full = 1 + np.argmax(votes, axis=2)

# ===========================================================
# 3) GUARDAR RESULTADOS
# ===========================================================
# Se asignan colores distintos a cada clase para visualizar la segmentación
colors = np.array([
    [255, 0, 0],      # clase 1 - rojo
    [0, 255, 0],      # clase 2 - verde
    [0, 0, 255],      # clase 3 - azul
    [255, 255, 0],    # clase 4 - amarillo
    [255, 0, 255],    # clase 5 - magenta
    [0, 255, 255],    # clase 6 - cian
    [255, 128, 0],    # clase 7 - naranja
], dtype=np.uint8)

# Se crea la imagen a color asignando el color correspondiente a cada etiqueta
preview = colors[labels_full - 1]

# Guarda el mapa de segmentación completo
imageio.imwrite(os.path.join(OUT_DIR, "segmentation_preview.png"), preview)

# Guarda también las máscaras binarias individuales (una por textura)
for c in range(1, NUM_CLASSES + 1):
    mask = (labels_full == c).astype(np.uint8) * 255
    imageio.imwrite(os.path.join(OUT_DIR, f"mask_class_{c}.png"), mask)

print("--Segmentación completa. Resultados guardados en:", OUT_DIR)
