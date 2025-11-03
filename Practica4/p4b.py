import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# === Ruta de la imagen ===
image_path = "ComposicionJoseAntonio1181x1193.png"  # cámbiala si está en otra carpeta

# === Cargar y convertir a escala de grises ===
img = Image.open(image_path).convert("L")
gray = np.array(img, dtype=np.uint8)

# === Implementación manual del LBP ===
h, w = gray.shape
lbp = np.zeros_like(gray, dtype=np.uint8)

# Se definen pesos en sentido horario desde la esquina superior izquierda
weights = [1, 2, 4, 8, 16, 32, 64, 128]
neighbors = [(-1,-1), (-1,0), (-1,1),
             ( 0, 1), ( 1,1), ( 1,0),
             ( 1,-1), ( 0,-1)]

for y in range(1, h-1):
    for x in range(1, w-1):
        c = gray[y, x]
        code = 0
        for wgt, (dy, dx) in zip(weights, neighbors):
            nb = gray[y+dy, x+dx]
            if nb >= c:
                code += wgt
        lbp[y, x] = code

# === Mostrar resultados ===
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Imagen original (grises)")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("LBP manual 3x3")
plt.imshow(lbp, cmap="gray")
plt.axis("off")

plt.show()

# === Guardar resultado ===
Image.fromarray(lbp).save("lbp_resultado.png")
print("LBP guardado como lbp_resultado.png")
