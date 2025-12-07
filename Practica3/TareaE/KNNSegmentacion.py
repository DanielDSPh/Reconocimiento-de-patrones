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


# 2. CREAMOS MÁSCARAS DE ETIQUETAS PARA ENTRENAR


training_labels_1 = np.zeros(img1.shape[:2], dtype=np.uint8)  # empezamos con todo cero (fondo)
training_labels_1[220:293, 300:380] = 2  # zona de clase 1
training_labels_1[50:104, 118:239] = 1  # zona de clase 2

training_labels_2 = np.zeros(img2.shape[:2], dtype=np.uint8)
training_labels_2[174:293, 239:391] = 2
training_labels_2[32:85, 35:123] = 1

training_labels_3 = np.zeros(img3.shape[:2], dtype=np.uint8)
training_labels_3[157:257, 180:294] = 2
training_labels_3[26:109, 27:90] = 1

training_labels_4 = np.zeros(img4.shape[:2], dtype=np.uint8)
training_labels_4[100:150, 280:350] = 2
training_labels_4[320:370, 10:100] = 1

list_of_training_labels = [training_labels_1, training_labels_2, training_labels_3]


# 3. FUNCIÓN PARA CALCULAR IOU (qué tanto se parecen predicción y verdad)


def calculaIoU(gtMask, predMask):
        # Calcula verdaderos positivos, falsos positivos, and falsos negativos
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(gtMask)):
            for j in range(len(gtMask[0])):
                if gtMask[i][j] == 1 and predMask[i][j] == 2:
                    tp += 1
                elif gtMask[i][j] == 0 and predMask[i][j] == 2:
                    fp += 1
                elif gtMask[i][j] == 1 and predMask[i][j] == 1:
                    fn += 1

        # Calcula IoU
        iou = tp / (tp + fp + fn)

        return iou



# 4. SEGMENTACIÓN SOLO CON INTENSIDAD DE PÍXELES

all_training_features_int = []
all_training_labels_int = []

for img, labels in zip(list_of_training_imgs, list_of_training_labels):
    labels_flat = labels.ravel()                        # convertimos a 1D
    features_flat = img.reshape(-1, img.shape[-1])      # cada píxel es una fila
    mask = labels_flat > 0                               # solo tomamos los píxeles que etiquetamos
    all_training_features_int.append(features_flat[mask])
    all_training_labels_int.append(labels_flat[mask])

training_features_int_combined = np.concatenate(all_training_features_int, axis=0)
training_labels_int_combined = np.concatenate(all_training_labels_int)

# entrenamos un k-NN sencillo
clf_int = KNeighborsClassifier(n_neighbors=20)
clf_int.fit(training_features_int_combined, training_labels_int_combined)

# hacemos predicción en una imagen de prueba
test_img = img1
test_gt = seg1
test_labels_training = training_labels_1

result_int = future.predict_segmenter(test_img, clf_int)  # predice cada píxel

# mostramos resultados
print(f"--- Resultado con solo intensidad de píxeles ---")

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 5))
fig.suptitle('Resultado de Segmentación: Intensidad de pixeles', fontsize=16, fontweight='bold')
ax[0].imshow(test_img)
ax[0].set_title("Imagen Original")
ax[1].imshow(segmentation.mark_boundaries(test_img, result_int, mode='thick'))
ax[1].contour(test_labels_training)
ax[1].set_title('Segmentación')
ax[2].imshow(result_int, cmap='gray')
ax[2].set_title('Segmentación Esperada')
fig.tight_layout()


print("El porcentaje de segmentación correcta es: {}%".format(round(1-calculaIoU(seg4, result_int),4)*100))



# 5. SEGMENTACIÓN CON CARACTERÍSTICAS MULTIESCALA


sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=False, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)

all_training_features_ec = []
all_training_labels_ec = []

for img, labels in zip(list_of_training_imgs, list_of_training_labels):
    features = features_func(img)                       # sacamos características  de cada píxel
    labels_flat = labels.ravel()
    features_flat = features.reshape(-1, features.shape[-1])
    mask = labels_flat > 0
    all_training_features_ec.append(features_flat[mask])
    all_training_labels_ec.append(labels_flat[mask])

training_features_ec_combined = np.concatenate(all_training_features_ec, axis=0)
training_labels_ec_combined = np.concatenate(all_training_labels_ec)

# entrenamos otro k-NN con estas características
clf_ec = KNeighborsClassifier(n_neighbors=20)
clf_ec.fit(training_features_ec_combined, training_labels_ec_combined)

features_test = features_func(test_img)
result_ec = future.predict_segmenter(features_test, clf_ec)

# mostramos resultados
print(f"--- Resultado con características multiescala ---")
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(13, 5))
fig.suptitle('Resultado de Segmentación: Características Multiescala', fontsize=16, fontweight='bold')
ax[0].imshow(test_img)
ax[0].set_title("Imagen Original")
ax[1].imshow(segmentation.mark_boundaries(test_img, result_ec, mode='thick'))
ax[1].contour(test_labels_training)
ax[1].set_title('Segmentación ')
ax[2].imshow(result_ec, cmap='gray')
ax[2].set_title('Segmentación Esperada')
fig.tight_layout()
plt.show()

print("El porcentaje de segmentación correcta es: {}%".format(round(1-calculaIoU(seg4, result_ec),4)*100))
