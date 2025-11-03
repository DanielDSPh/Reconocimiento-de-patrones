import cv2
import numpy as np
import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm  # escala log para visualizar GLCM

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ============================================================
# CONFIGURACIÓN INICIAL
# ============================================================
IMAGE_DIR = 'texturas'      # Carpeta con las texturas (cada archivo = una clase)
IMG_SIZE = (640, 640)       # Tamaño esperado; si no coincide, la imagen se omite
WINDOW_SIZE = 64            # Tamaño del texel (ventana cuadrada)
N_TEST_WINDOWS = 4          # Nº de ventanas por imagen que se reservan a PRUEBA

# Parámetros de la GLCM (varias distancias y ángulos, como pide la consigna)
DISTANCES = [1, 3, 5]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# Estadísticos de 2º orden (Haralick) a extraer de la GLCM
PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# ----------------------------------------------------------------------
# PASOS 1-5: EXTRACCIÓN DE CARACTERÍSTICAS (GLCM por ventanas)
# ----------------------------------------------------------------------

def extract_windows(image, window_size):
    """Corta la imagen en ventanas no solapadas de tamaño fijo (texels)."""
    windows = []
    h, w = image.shape
    for r in range(0, h, window_size):
        for c in range(0, w, window_size):
            window = image[r:r + window_size, c:c + window_size]
            windows.append(window)
    return windows

def extract_features(window):
    """
    Calcula la GLCM de una ventana para todas las distancias y ángulos,
    y promedia cada propiedad de Haralick → vector de 6 características.
    levels=256 asume imágenes uint8 (0..255).
    """
    glcm = graycomatrix(window,
                        distances=DISTANCES,
                        angles=ANGLES,
                        levels=256,
                        symmetric=True,
                        normed=True)

    feature_vector = []
    for prop in PROPERTIES:
        # graycoprops devuelve matriz [len(distances) x len(angles)]
        # se promedia para obtener un único valor por propiedad
        prop_values = graycoprops(glcm, prop)
        feature_vector.append(np.mean(prop_values))
    return feature_vector

def process_images():
    """
    Recorre la carpeta: por cada imagen (clase) genera ventanas,
    separa TRAIN/TEST, extrae GLCM-features y devuelve X/y (NumPy).
    """
    train_features_list, train_labels_list = [], []
    test_features_list, test_labels_list = [], []

    if not os.path.exists(IMAGE_DIR):
        print(f"Error: El directorio '{IMAGE_DIR}' no existe.")
        return None, None, None, None

    print(f"Buscando imágenes en '{IMAGE_DIR}'...")
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.tif', '.jpg', '.bmp'))]

    if len(image_files) == 0:
        print("No se encontraron imágenes. Asegúrate de que están en la carpeta 'texturas'.")
        return None, None, None, None

    if len(image_files) < 8:
        print(f"Advertencia: Se encontraron {len(image_files)} imágenes. Se recomiendan 8-12.")

    # Mapeo nombre de archivo → etiqueta numérica (0..N-1)
    class_names = sorted(list(set(image_files)))
    class_map = {name: i for i, name in enumerate(class_names)}

    for file_name in image_files:
        class_label = class_map[file_name]
        file_path = os.path.join(IMAGE_DIR, file_name)

        # Leer en escala de grises (uint8)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"No se pudo leer la imagen: {file_name}")
            continue

        # Asegurar tamaño esperado (sino se omite para mantener ventanas completas)
        if img.shape != IMG_SIZE:
            print(f"Advertencia: La imagen {file_name} no es de {IMG_SIZE}. Omitiendo.")
            continue

        print(f"\nProcesando textura: {file_name} (Clase {class_label})")

        # 1) Ventaneo sin solape
        windows = extract_windows(img, WINDOW_SIZE)
        print(f"  > Extraídas {len(windows)} ventanas de {WINDOW_SIZE}x{WINDOW_SIZE}")

        # 2) Barajar y split simple: primeras N_TEST_WINDOWS a test, resto a train
        np.random.shuffle(windows)
        test_windows = windows[:N_TEST_WINDOWS]
        train_windows = windows[N_TEST_WINDOWS:]
        print(f"  > Dividido en {len(train_windows)} ventanas de entrenamiento y {len(test_windows)} de prueba.")

        # 3) Extraer features por ventana y etiquetar
        for window in train_windows:
            train_features_list.append(extract_features(window))
            train_labels_list.append(class_label)

        for window in test_windows:
            test_features_list.append(extract_features(window))
            test_labels_list.append(class_label)

    # Listas → arreglos NumPy
    X_train = np.array(train_features_list)
    y_train = np.array(train_labels_list)
    X_test  = np.array(test_features_list)
    y_test  = np.array(test_labels_list)

    print("\n--- Extracción de Características Completada ---")
    print(f"Forma de X_train (vectores): {X_train.shape}")
    print(f"Forma de y_train (etiquetas): {y_train.shape}")
    print(f"Forma de X_test (vectores):  {X_test.shape}")
    print(f"Forma de y_test (etiquetas):  {y_test.shape}")

    return X_train, y_train, X_test, y_test, class_names

# ----------------------------------------------------------------------
# PASO 6: CLASIFICADOR k-NN MANUAL (implementación didáctica)
# ----------------------------------------------------------------------

class kNN():
    def __init__(self, k=3, exp=2):
        self.k = k           # Nº de vecinos
        self.exp = exp       # p de Minkowski (2 = Euclidiana)

    def fit(self, X_train, Y_train):
        # Se guarda en Pandas para facilitar ordenamientos/índices
        self.X_train = X_train
        self.Y_train = Y_train

    def getDiscreteClassification(self, X_test):
        """Predice etiqueta por voto mayoritario de los k vecinos más cercanos."""
        Y_pred_test = []
        for i in range(len(X_test)):
            test_instance = X_test.iloc[i]
            distances = []
            for j in range(len(self.X_train)):
                train_instance = self.X_train.iloc[j]
                distances.append(self.Minkowski_distance(test_instance, train_instance))

            # Distancias en DataFrame para ordenar y tomar top-k
            df_dists = pd.DataFrame(data=distances, columns=['dist'], index=self.Y_train.index)
            df_knn = df_dists.sort_values(by=['dist'], axis=0)[:self.k]

            # Voto mayoritario entre las etiquetas de esos k índices
            y_pred_test = self.Y_train[df_knn.index].value_counts().index[0]
            Y_pred_test.append(y_pred_test)
        return Y_pred_test

    def Minkowski_distance(self, x1, x2):
        """Distancia de Minkowski vectorizada (p=self.exp)."""
        distance = (abs(x1 - x2)**self.exp).sum()
        return distance**(1/self.exp)

    # Normalización NO se usa aquí (se hace con StandardScaler para evitar leakage)
    def normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

# ----------------------------------------------------------------------
# VISUALIZACIÓN DE GLCM (opcional para el reporte)
# ----------------------------------------------------------------------

def plot_glcm_example(window, distance, angle):
    """Muestra una ventana y su GLCM para un (d, ángulo) específico."""
    glcm = graycomatrix(window, distances=[distance], angles=[angle],
                        levels=256, symmetric=True, normed=True)
    glcm_matrix = glcm[:, :, 0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(window, cmap='gray'); axes[0].set_title(f'Ventana {window.shape}'); axes[0].axis('off')
    if np.count_nonzero(glcm_matrix) > 0:
        axes[1].imshow(glcm_matrix, cmap='viridis', norm=LogNorm())
    else:
        axes[1].imshow(glcm_matrix, cmap='viridis')
    axes[1].set_title(f'GLCM (d={distance}, a={int(np.degrees(angle))}°)')
    axes[1].set_xlabel('Nivel de Gris (j)'); axes[1].set_ylabel('Nivel de Gris (i)')
    plt.tight_layout(); plt.show()

def save_glcm_visualizations(window, base_name, distances, angles):
    """Guarda figuras (Texel + GLCM) para todas las combinaciones d×ángulo."""
    output_dir = "glcm_imagenes"; os.makedirs(output_dir, exist_ok=True)
    print(f"  > Guardando GLCMs para '{base_name}' en '{output_dir}'...")
    for d in distances:
        for a_rad in angles:
            glcm = graycomatrix(window, distances=[d], angles=[a_rad],
                                levels=256, symmetric=True, normed=True)
            glcm_matrix = glcm[:, :, 0, 0]; a_deg = int(np.degrees(a_rad))

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(window, cmap='gray'); axes[0].set_title('Texel'); axes[0].axis('off')
            im = axes[1].imshow(glcm_matrix, cmap='viridis', norm=LogNorm() if np.count_nonzero(glcm_matrix) > 0 else None)
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            axes[1].set_title(f'GLCM (d={d}, a={a_deg}°)'); axes[1].set_xlabel('j'); axes[1].set_ylabel('i')
            fig.suptitle(f'Análisis GLCM: {base_name}', y=1.03); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_d{d}_a{a_deg}.png"), bbox_inches='tight')
            plt.close(fig)
    print(f"  > Imágenes GLCM compuestas para '{base_name}' guardadas.")

# ----------------------------------------------------------------------
# PASO 7: ENTRENAMIENTO, COMPARACIÓN Y RESULTADOS
# ----------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1) Crear datasets de características/etiquetas
    X_train_np, y_train_np, X_test_np, y_test_np, class_names = process_images()
    if X_train_np is None or X_train_np.shape[0] == 0:
        print("\nNo se procesaron datos. Finalizando programa.")
    else:
        # 2) Normalización (crucial para k-NN/SVM). Fit SOLO con train; luego transform a train y test.
        print("\n--- Normalizando Datos (StandardScaler) ---")
        scaler = StandardScaler()
        X_train_norm_np = scaler.fit_transform(X_train_np)
        X_test_norm_np  = scaler.transform(X_test_np)
        print("Datos normalizados listos.")

        # 3) Clasificador 1: k-NN MANUAL (k=5, distancia Euclidiana)
        print("\n--- Clasificador 1: k-NN Manual (k=5) ---")
        X_train_df = pd.DataFrame(X_train_norm_np); y_train_s = pd.Series(y_train_np)
        X_test_df  = pd.DataFrame(X_test_norm_np)
        knn_manual = kNN(k=5, exp=2)
        knn_manual.fit(X_train_df, y_train_s)
        y_pred_manual = knn_manual.getDiscreteClassification(X_test_df)
        acc_manual = accuracy_score(y_test_np, y_pred_manual)
        print(f"  > Precisión k-NN Manual: {acc_manual * 100:.2f}%")

        # 4) Clasificador 2: k-NN (scikit-learn)
        print("\n--- Clasificador 2: k-NN (sklearn, k=5) ---")
        knn_sklearn = KNeighborsClassifier(n_neighbors=5, p=2)  # p=2 = Euclidiana
        knn_sklearn.fit(X_train_norm_np, y_train_np)
        y_pred_knn_sklearn = knn_sklearn.predict(X_test_norm_np)
        acc_knn_sklearn = accuracy_score(y_test_np, y_pred_knn_sklearn)
        print(f"  > Precisión k-NN (sklearn): {acc_knn_sklearn * 100:.2f}%")

        # 5) Clasificador 3: SVM (segundo clasificador requerido)
        print("\n--- Clasificador 3: SVM (sklearn, kernel RBF) ---")
        svm_classifier = SVC(kernel='rbf')
        svm_classifier.fit(X_train_norm_np, y_train_np)
        y_pred_svm = svm_classifier.predict(X_test_norm_np)
        acc_svm = accuracy_score(y_test_np, y_pred_svm)
        print(f"  > Precisión SVM (RBF): {acc_svm * 100:.2f}%")

        # 6) Comparación general
        print("\n--- Comparación Final de Resultados ---")
        print(f"k-NN Manual (k=5):   \t{acc_manual * 100:.2f}%")
        print(f"k-NN sklearn (k=5):  \t{acc_knn_sklearn * 100:.2f}%")
        print(f"SVM sklearn (RBF):   \t{acc_svm * 100:.2f}%")

# 7) Matrices de confusión (se guardan como PNG)
        print("\nGenerando matrices de confusión...")
        labels_str = [class_names[i] for i in sorted(np.unique(y_train_np))]

        cm = confusion_matrix(y_test_np, y_pred_svm, labels=sorted(np.unique(y_train_np)))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_str, yticklabels=labels_str)
        plt.title(f'Matriz de Confusión - SVM ({acc_svm * 100:.2f}%)'); plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig('matriz_confusion_svm.png'); print("Guardado: matriz_confusion_svm.png")

        cm_manual = confusion_matrix(y_test_np, y_pred_manual, labels=sorted(np.unique(y_train_np)))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Greens', xticklabels=labels_str, yticklabels=labels_str)
        plt.title(f'Matriz de Confusión - k-NN Manual ({acc_manual * 100:.2f}%)'); plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig('matriz_confusion_knn_manual.png'); print("Guardado: matriz_confusion_knn_manual.png")

        cm_knn_sklearn = confusion_matrix(y_test_np, y_pred_knn_sklearn, labels=sorted(np.unique(y_train_np)))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_knn_sklearn, annot=True, fmt='d', cmap='Oranges', xticklabels=labels_str, yticklabels=labels_str)
        plt.title(f'Matriz de Confusión - k-NN sklearn ({acc_knn_sklearn * 100:.2f}%)'); plt.ylabel('Etiqueta Verdadera'); plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig('matriz_confusion_knn_sklearn.png'); print("Guardado: matriz_confusion_knn_sklearn.png")

