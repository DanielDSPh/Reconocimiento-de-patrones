import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, feature, future, data
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from PIL import Image

# 1. CARGAMOS LAS IMÁGENES Y SUS SEGMENTACIONES

img1 = plt.imread("ISIC_0024313.jpg")
seg1 = np.asarray(Image.open("ISIC_0024313_segmentation.jpg").convert("L"))  # lo pasamos a blanco y negro

img2 = plt.imread("ISIC_0024351.jpg")
seg2 = np.asarray(Image.open("ISIC_0024351_segmentation.jpg").convert("L"))

img3 = plt.imread("ISIC_0024481.jpg")
seg3 = np.asarray(Image.open("ISIC_0024481_segmentation.jpg").convert("L"))

img4 = plt.imread("ISIC_0024496.jpg")
seg4 = np.asarray(Image.open("ISIC_0024496_segmentation.jpg").convert("L"))

# cortamos las imágenes para que todas tengan el mismo tamaño 
img1, seg1 = img1[:400, :500], seg1[:400, :500]
img2, seg2 = img2[:400, :500], seg2[:400, :500]
img3, seg3 = img3[:400, :500], seg3[:400, :500]
img4, seg4 = img4[:400, :500], seg3[:400, :500]
# listas para manejar todo más fácil
list_of_training_imgs = [img1, img2, img3]
list_of_ground_truths = [seg1, seg2, seg3]


images = [img1, img2, img3, img4]
image_names = ["img1", "img2", "img3", "img4"]

colors = [(0, 255, 0), (0, 0, 255)]  # rojo = clase1, verde = clase2

# Diccionario para guardar rectángulos de cada imagen
all_rects = {name: [] for name in image_names}

current_class = 1
start_x, start_y = None, None
current_img_idx = 0

def on_click(event):
    global start_x, start_y
    start_x, start_y = int(event.xdata), int(event.ydata)

def on_release(event):
    global current_img_idx
    end_x, end_y = int(event.xdata), int(event.ydata)
    x0, x1 = sorted([start_x, end_x])
    y0, y1 = sorted([start_y, end_y])
    
    # Guardar rectángulo en la lista correspondiente
    all_rects[image_names[current_img_idx]].append((y0, y1, x0, x1, current_class))
    
    # Dibujar sobre la máscara
    mask_viz = np.zeros_like(images[current_img_idx], dtype=np.uint8)
    for y0_r, y1_r, x0_r, x1_r, c in all_rects[image_names[current_img_idx]]:
        mask_viz[y0_r:y1_r, x0_r:x1_r] = colors[c-1]
    
    ax.imshow(images[current_img_idx])
    ax.imshow(mask_viz, alpha=0.4)
    fig.canvas.draw()

def toggle_class(event):
    global current_class
    if event.key == '1':
        current_class = 1
        print("Clase actual: 1 (melanoma)")
    elif event.key == '2':
        current_class = 2
        print("Clase actual: 2 (piel)")
    elif event.key == 'n':  # pasar a la siguiente imagen
        next_image()

def next_image():
    global current_img_idx, ax
    if current_img_idx < len(images)-1:
        current_img_idx += 1
        ax.clear()
        ax.imshow(images[current_img_idx])
        plt.title(f"Dibuja rectángulos para {image_names[current_img_idx]}")
        fig.canvas.draw()
        print(f"Imagen actual: {image_names[current_img_idx]}")
    else:
        print("Última imagen alcanzada.")

fig, ax = plt.subplots()
ax.imshow(images[current_img_idx])
plt.title(f"Dibuja rectángulos para {image_names[current_img_idx]} (1=verde, 2=azul, n=siguiente)")

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event', toggle_class)

plt.show()

print("Rectángulos definidos por imagen:")
for name, rects in all_rects.items():
    print(name, rects)
