import cv2
import numpy as np
import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm # Para ver mejor los detalles

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Configuración Inicial ---
IMAGE_DIR = 'texturas'  # Carpeta donde están tus imágenes
IMG_SIZE = (640, 640)
WINDOW_SIZE = 64   # Tamaño de la ventana (texel)
N_TEST_WINDOWS = 4   # Número de ventanas para prueba por imagen

DISTANCES = [1, 3, 5]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# ----------------------------------------------------------------------
# PASOS 1-5: EXTRACCIÓN DE CARACTERÍSTICAS (TU SCRIPT ANTERIOR)
# ----------------------------------------------------------------------

def extract_windows(image, window_size):
    """Divide una imagen en ventanas (texels) no superpuestas."""
    windows = []
    h, w = image.shape
    for r in range(0, h, window_size):
        for c in range(0, w, window_size):
            window = image[r:r + window_size, c:c + window_size]
            windows.append(window)
    return windows

def extract_features(window):
    """Calcula la GLCM y extrae el vector de características de una ventana."""
    glcm = graycomatrix(window, 
                        distances=DISTANCES, 
                        angles=ANGLES, 
                        levels=256,
                        symmetric=True, 
                        normed=True)
    
    feature_vector = []
    for prop in PROPERTIES:
        prop_values = graycoprops(glcm, prop)
        feature_vector.append(np.mean(prop_values))
        
    return feature_vector

def process_images():
    """Procesa todas las imágenes en el directorio y crea los conjuntos de datos."""
    train_features_list = []
    train_labels_list = []
    test_features_list = []
    test_labels_list = []

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

    # Usaremos etiquetas numéricas para los clasificadores (0, 1, 2...)
    # pero guardamos los nombres para los reportes.
    class_names = sorted(list(set(image_files)))
    class_map = {name: i for i, name in enumerate(class_names)}

    for file_name in image_files:
        class_label = class_map[file_name] # Etiqueta numérica
        file_path = os.path.join(IMAGE_DIR, file_name)
        
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"No se pudo leer la imagen: {file_name}")
            continue
            
        if img.shape != IMG_SIZE:
            print(f"Advertencia: La imagen {file_name} no es de {IMG_SIZE}. Omitiendo.")
            continue
            
        print(f"\nProcesando textura: {file_name} (Clase {class_label})")

        windows = extract_windows(img, WINDOW_SIZE)
        print(f"  > Extraídas {len(windows)} ventanas de {WINDOW_SIZE}x{WINDOW_SIZE}")

        np.random.shuffle(windows) 
        
        test_windows = windows[:N_TEST_WINDOWS]
        train_windows = windows[N_TEST_WINDOWS:]
        print(f"  > Dividido en {len(train_windows)} ventanas de entrenamiento y {len(test_windows)} de prueba.")

        for window in train_windows:
            features = extract_features(window)
            train_features_list.append(features)
            train_labels_list.append(class_label)
            
        for window in test_windows:
            features = extract_features(window)
            test_features_list.append(features)
            test_labels_list.append(class_label)

    # Convertir listas a arreglos de NumPy
    X_train = np.array(train_features_list)
    y_train = np.array(train_labels_list)
    X_test = np.array(test_features_list)
    y_test = np.array(test_labels_list)
    
    print("\n--- Extracción de Características Completada ---")
    print(f"Forma de X_train (vectores): {X_train.shape}")
    print(f"Forma de y_train (etiquetas): {y_train.shape}")
    print(f"Forma de X_test (vectores):  {X_test.shape}")
    print(f"Forma de y_test (etiquetas):  {y_test.shape}")
    
    return X_train, y_train, X_test, y_test, class_names

# ----------------------------------------------------------------------
# PASO 6: CLASIFICADOR K-NN MANUAL (TU CLASE)
# ----------------------------------------------------------------------

class kNN():
    def __init__(self, k = 3, exp = 2):
        self.k = k
        self.exp = exp
      
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train 
           
    def getDiscreteClassification(self, X_test):
        Y_pred_test = [] 
    
        for i in range(len(X_test)):
            test_instance = X_test.iloc[i] 
            
            distances = [] 
            
            for j in range(len(self.X_train)):
                train_instance = self.X_train.iloc[j] 
                distance = self.Minkowski_distance(test_instance, train_instance) 
                distances.append(distance)
        
            # Convertir distancias a DataFrame de Pandas
            df_dists = pd.DataFrame(data=distances, columns=['dist'], index = self.Y_train.index)
        
            df_nn = df_dists.sort_values(by=['dist'], axis=0)
            df_knn =  df_nn[:self.k]
            
            # Votación de la mayoría
            predictions = self.Y_train[df_knn.index].value_counts()
            y_pred_test = predictions.index[0]

            Y_pred_test.append(y_pred_test)
        
        return Y_pred_test

    
    def Minkowski_distance(self, x1, x2):
        # (Tu implementación vectorizada)
        distance = (abs(x1 - x2)**self.exp).sum()
        distance = distance**(1/self.exp)
        return distance
    
    # Nota: No usaremos normalize() ya que es mejor usar StandardScaler de sklearn
    # para evitar "data leakage" (filtrado de datos).
    def normalize(self, data):
        normalized_data = (data-data.min())/(data.max()-data.min())
        return normalized_data
    


def plot_glcm_example(window, distance, angle):
    """Calcula y grafica una ventana y su GLCM específica."""
    
    # Calcular UNA sola GLCM
    glcm = graycomatrix(window, 
                        distances=[distance], 
                        angles=[angle], 
                        levels=256,
                        symmetric=True, 
                        normed=True)
    
    # Extraer la matriz 2D (quitando las dimensiones de distancia y ángulo)
    glcm_matrix = glcm[:, :, 0, 0]
    
    # --- Graficar ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. La ventana original
    axes[0].imshow(window, cmap='gray')
    axes[0].set_title(f'Ventana (Texel) {window.shape}')
    axes[0].axis('off')
    
    # 2. La GLCM (usando LogNorm para mejor visualización)
    # Si no se usa LogNorm, la diagonal será un punto blanco y el resto negro.
    if np.count_nonzero(glcm_matrix) > 0:
        axes[1].imshow(glcm_matrix, cmap='viridis', norm=LogNorm())
    else:
        axes[1].imshow(glcm_matrix, cmap='viridis') # Caso de matriz vacía
        
    axes[1].set_title(f'GLCM (d={distance}, a={int(np.degrees(angle))}°)')
    axes[1].set_xlabel('Nivel de Gris (j)')
    axes[1].set_ylabel('Nivel de Gris (i)')
    
    plt.tight_layout()
    plt.show()

def save_glcm_visualizations(window, base_name, distances, angles):
    """
    Calcula y guarda como imágenes una figura COMPUESTA (Texel + GLCM),
    probando todas las combinaciones de distancias y ángulos.
    """
    
    output_dir = "glcm_imagenes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"  > Guardando GLCMs para '{base_name}' en la carpeta '{output_dir}'...")
    
    for d in distances:
        for a_rad in angles:
            glcm = graycomatrix(window, 
                                distances=[d], 
                                angles=[a_rad], 
                                levels=256,
                                symmetric=True, 
                                normed=True)
            
            glcm_matrix = glcm[:, :, 0, 0]
            a_deg = int(np.degrees(a_rad))
            
            # --- INICIO DE LA MODIFICACIÓN ---
            
            # NUEVO: Crear una figura con 2 subplots (1 fila, 2 columnas)
            # 'axes' será un array: axes[0] (izquierda), axes[1] (derecha)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Figura más ancha
            
            # --- Subplot 1: El Texel Original ---
            axes[0].imshow(window, cmap='gray')
            axes[0].set_title(f'Texel Original (Ventana)')
            axes[0].axis('off') # No mostrar ejes para la imagen
            
            # --- Subplot 2: La GLCM ---
            try:
                # Usar imshow() en el segundo subplot (axes[1])
                if np.count_nonzero(glcm_matrix) > 0:
                    im = axes[1].imshow(glcm_matrix, cmap='viridis', norm=LogNorm())
                else:
                    im = axes[1].imshow(glcm_matrix, cmap='viridis') # Caso de matriz vacía
                
                # Añadir barra de color (asociada a la figura y al subplot 'im')
                fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                
                axes[1].set_title(f'GLCM (Distancia={d}, Ángulo={a_deg}°)')
                axes[1].set_xlabel('Nivel de Gris (j)')
                axes[1].set_ylabel('Nivel de Gris (i)')
                
                # Título general para toda la figura
                fig.suptitle(f'Análisis GLCM para: {base_name}', fontsize=14, y=1.03)
                
                # Ajustar el layout para que no se solapen
                plt.tight_layout()
                
                # Definir nombre de archivo y guardar
                filename = f"{base_name}_d{d}_a{a_deg}.png"
                filepath = os.path.join(output_dir, filename)
                
                plt.savefig(filepath, bbox_inches='tight') # bbox_inches extra
                
            except Exception as e:
                print(f"    Error guardando {base_name}_d{d}_a{a_deg}.png: {e}")
            
            # Cerrar la figura para liberar memoria
            plt.close(fig)
            
            # --- FIN DE LA MODIFICACIÓN ---

    print(f"  > Imágenes GLCM compuestas para '{base_name}' guardadas.")

# ----------------------------------------------------------------------
# PASO 7: EJECUCIÓN, COMPARACIÓN Y RESULTADOS
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Ignorar advertencias de scikit-image
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 1. Obtener los datos (NumPy arrays)
    X_train_np, y_train_np, X_test_np, y_test_np, class_names = process_images()

    if X_train_np is None or X_train_np.shape[0] == 0:
        print("\nNo se procesaron datos. Finalizando programa.")
    else:
        # Cargar una imagen de ejemplo para la demo
        
        ''' Crea las imagenes de las matrices GLCM (Punto 4)
        descomentar y correr genera muchas imagenes en la carpeta glcm_imagenes
        de preferencia correr una sola vez y comentar de nuevo para no tardar demasiado
        '''
        '''
        print("\n--- Generando Imágenes de GLCM (Punto 4) ---")
        
        try:
            # Iterar sobre cada textura (clase)
            for i, class_name in enumerate(class_names):
                
                # Cargar la imagen original de esta clase
                file_path = os.path.join(IMAGE_DIR, class_name)
                img_demo = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                if img_demo is not None:
                    # Extraer la primera ventana (64x64) de esa imagen
                    window_demo = img_demo[0:WINDOW_SIZE, 0:WINDOW_SIZE]
                    
                    # Generar un nombre base para los archivos
                    # ej: 'textura_madera.tif' -> 'textura_madera'
                    base_name = os.path.splitext(class_name)[0] 
                    
                    # Llamar a la nueva función
                    save_glcm_visualizations(window_demo, 
                                             base_name, 
                                             DISTANCES,  # [1, 3, 5]
                                             ANGLES)     # [0, 45, 90, 135]
                else:
                    print(f"No se pudo leer {file_path} para generar GLCMs.")
                    
        except Exception as e:
            print(f"Error fatal al generar imágenes GLCM: {e}")
        '''

        # 2. Normalizar los datos (¡MUY IMPORTANTE!)
        # Usamos StandardScaler: se ajusta (fit) SÓLO con datos de train
        # y luego transforma (transform) train y test.
        print("\n--- Normalizando Datos (StandardScaler) ---")
        scaler = StandardScaler()
        X_train_norm_np = scaler.fit_transform(X_train_np)
        X_test_norm_np = scaler.transform(X_test_np)
        print("Datos normalizados listos.")

        # ---
        # Clasificador 1: Tu k-NN Manual
        # ---
        print("\n--- Clasificador 1: k-NN Manual (k=5) ---")
        
        # Convertir datos normalizados a Pandas para tu clase
        X_train_df = pd.DataFrame(X_train_norm_np)
        y_train_s = pd.Series(y_train_np)
        X_test_df = pd.DataFrame(X_test_norm_np)
        
        knn_manual = kNN(k=5, exp=2) # k=5, exp=2 (Distancia Euclidiana)
        
        print("Entrenando k-NN Manual...")
        knn_manual.fit(X_train_df, y_train_s)
        
        print("Prediciendo con k-NN Manual...")
        y_pred_manual = knn_manual.getDiscreteClassification(X_test_df)
        
        acc_manual = accuracy_score(y_test_np, y_pred_manual)
        print(f"  > Precisión (Accuracy) k-NN Manual: {acc_manual * 100:.2f}%")

        # ---
        # Clasificador 2: k-NN (Scikit-learn)
        # ---
        print("\n--- Clasificador 2: k-NN (Scikit-learn, k=5) ---")
        
        # Usamos los arrays de NumPy normalizados
        knn_sklearn = KNeighborsClassifier(n_neighbors=5, p=2) # p=2 (Euclidiana)
        
        print("Entrenando k-NN (sklearn)...")
        knn_sklearn.fit(X_train_norm_np, y_train_np)
        
        print("Prediciendo con k-NN (sklearn)...")
        y_pred_knn_sklearn = knn_sklearn.predict(X_test_norm_np)
        
        acc_knn_sklearn = accuracy_score(y_test_np, y_pred_knn_sklearn)
        print(f"  > Precisión (Accuracy) k-NN (sklearn): {acc_knn_sklearn * 100:.2f}%")

        # ---
        # Clasificador 3: SVM (Scikit-learn) - El 2º clasificador requerido
        # ---
        print("\n--- Clasificador 3: SVM (Scikit-learn) ---")
        
        svm_classifier = SVC(kernel='rbf') # Kernel RBF (Base Radial) es un buen inicio
        
        print("Entrenando SVM (sklearn)...")
        svm_classifier.fit(X_train_norm_np, y_train_np)
        
        print("Prediciendo con SVM (sklearn)...")
        y_pred_svm = svm_classifier.predict(X_test_norm_np)
        
        acc_svm = accuracy_score(y_test_np, y_pred_svm)
        print(f"  > Precisión (Accuracy) SVM (sklearn): {acc_svm * 100:.2f}%")


        # ---
        # PASO 8: DESPLIEGUE DE RESULTADOS Y COMPARACIÓN
        # ---
        print("\n--- Comparación Final de Resultados ---")
        print(f"k-NN Manual (k=5):   \t{acc_manual * 100:.2f}%")
        print(f"k-NN sklearn (k=5):  \t{acc_knn_sklearn * 100:.2f}%")
        print(f"SVM sklearn (RBF):   \t{acc_svm * 100:.2f}%")

        # Generar Matriz de Confusión para el mejor clasificador (o el SVM)
        print("\nGenerando matriz de confusión para el clasificador SVM...")
        
        # Mapear etiquetas numéricas de vuelta a nombres de archivo
        labels_str = [class_names[i] for i in sorted(np.unique(y_train_np))]
        
        cm = confusion_matrix(y_test_np, y_pred_svm, labels=sorted(np.unique(y_train_np)))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels_str, yticklabels=labels_str)
        plt.title(f'Matriz de Confusión - SVM ({acc_svm * 100:.2f}%)')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar la imagen de resultados
        output_filename = 'matriz_confusion_svm.png'
        plt.savefig(output_filename)
        print(f"¡Resultados guardados en '{output_filename}'!")


        # ---
        # Matriz de Confusión para k-NN Manual
        # ---
        print("\nGenerando matriz de confusión para k-NN Manual...")
        
        cm_manual = confusion_matrix(y_test_np, y_pred_manual, labels=sorted(np.unique(y_train_np)))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=labels_str, yticklabels=labels_str)
        plt.title(f'Matriz de Confusión - k-NN Manual ({acc_manual * 100:.2f}%)')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_filename_manual = 'matriz_confusion_knn_manual.png'
        plt.savefig(output_filename_manual)
        print(f"¡Resultados guardados en '{output_filename_manual}'!")

        # ---
        # Matriz de Confusión para k-NN (Scikit-learn)
        # ---
        print("\nGenerando matriz de confusión para k-NN (sklearn)...")
        
        cm_knn_sklearn = confusion_matrix(y_test_np, y_pred_knn_sklearn, labels=sorted(np.unique(y_train_np)))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_knn_sklearn, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=labels_str, yticklabels=labels_str)
        plt.title(f'Matriz de Confusión - k-NN sklearn ({acc_knn_sklearn * 100:.2f}%)')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_filename_knn_sklearn = 'matriz_confusion_knn_sklearn.png'
        plt.savefig(output_filename_knn_sklearn)
        print(f"¡Resultados guardados en '{output_filename_knn_sklearn}'!")