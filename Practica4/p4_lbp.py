import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# ============================
# CONFIGURACIÓN
# ============================
# Rutas a las 7 texturas (debes colocar las tuyas)
textures = [
    "training/D6.bmp",
    "training/D16.bmp",
    "training/D46.bmp",
    "training/D49.bmp",
    "training/D64.bmp",
    "training/D101.bmp",
    "training/Piedras.jpg",
]

# Imagen compuesta
composite_path = "ComposicionJoseAntonio1181x1193.png"

# Carpeta de salida
OUT_DIR = "result_knn_7textures"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# FUNCIÓN LBP 3x3 (manual simple)
# ============================
def lbp_manual(gray):
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    weights = [1, 2, 4, 8, 16, 32, 64, 128]
    neighbors = [(-1,-1), (-1,0), (-1,1),
                 ( 0, 1), ( 1,1), ( 1,0),
                 ( 1,-1), ( 0,-1)]
    for y in range(1, h-1):
        for x in range(1, w-1):
            c = gray[y, x]
            code = 0
            for wgt, (dy, dx) in zip(weights, neighbors):
                if gray[y+dy, x+dx] >= c:
                    code += wgt
            lbp[y, x] = code
    return lbp

# ============================
# 1. ENTRENAMIENTO
# ============================
X_train, y_train = [], []

for i, path in enumerate(textures, start=1):
    img = Image.open(path).convert("L")
    gray = np.array(img, dtype=np.uint8)
    lbp = lbp_manual(gray)
    X_train.extend(lbp.reshape(-1, 1))
    y_train.extend([i] * (gray.shape[0] * gray.shape[1]))

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Clases únicas en el entrenamiento:", np.unique(y_train))

print(f"✅ Entrenamiento listo con {len(y_train)} muestras (7 clases).")

# ============================
# 2. PRUEBA CON IMAGEN COMPUESTA
# ============================
comp_img = Image.open(composite_path).convert("L")
comp_gray = np.array(comp_img, dtype=np.uint8)
comp_lbp = lbp_manual(comp_gray)
h, w = comp_gray.shape
X_test = comp_lbp.reshape(-1, 1)

# ============================
# 3. CLASIFICACIÓN KNN
# ============================
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
labels_map = y_pred.reshape(h, w)

# ============================
# 4. GUARDAR RESULTADOS
# ============================
# Paleta de colores fija (7)
colors = np.array([
    [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255],
    [0, 255, 255], [255, 128, 0]
], dtype=np.uint8)

colored = colors[labels_map - 1]  # -1 porque clases van 1–7
imageio.imwrite(os.path.join(OUT_DIR, "segmentation_preview.png"), colored)

for i in range(1, 8):
    mask = (labels_map == i).astype(np.uint8) * 255
    imageio.imwrite(os.path.join(OUT_DIR, f"mask_class_{i}.png"), mask)

plt.figure(figsize=(6,6))
plt.imshow(colored)
plt.title("Segmentación k-NN (7 texturas)")
plt.axis("off")
plt.show()

print("✅ Segmentación completa. Máscaras guardadas en:", OUT_DIR)
