import cv2
import numpy as np

def calcular_score(X, media, cov, prior):
    """
    Calcula el score de la función discriminante Gaussiana.
    """
    # Manejo de error para matriz no invertible (determinante = 0)
    # y para determinantes muy cercanos a cero, que pueden causar problemas numéricos
    det_cov = np.linalg.det(cov)
    if det_cov <= 1e-6:  # Usar un umbral pequeño en lugar de solo 0
        return -np.inf
        
    inv_cov = np.linalg.inv(cov)
    diff = X - media
    termino_maha = diff.T @ inv_cov @ diff
    
    # Se usa np.log(prior) directamente, asumiendo que prior no será 0
    score = -0.5 * termino_maha - 0.5 * np.log(det_cov) + np.log(prior)
    
    return score

# --- 1. CONFIGURACIÓN: Pega tus parámetros aquí ---
# Estos son los parámetros obtenidos al usar el clasificador manualya clasificando 4 imágenes de entrenamiento
parametros = {
    'chile': {
        'media': np.array([21.60892744, 79.72258958, 59.73048319]),
        'cov': np.array([[ 544.0491998 ,  563.51519463,  363.34864716],
 [ 563.51519463,  941.97259144,  165.2251903 ],
 [ 363.34864716,  165.2251903 , 1024.06409725]]),
        'prior': 0.04715486,
        'color': (255, 0, 0) # Define el color BGR para 'chile'
    },
    'platano': {
        'media': np.array([ 57.216258  , 188.10567969, 213.32548102]),
        'cov': np.array([[ 448.80377986,  480.86037429,  288.34813979],
 [ 480.86037429, 1161.43617212,  627.76910442],
 [ 288.34813979,  627.76910442,  423.38455331]]),
        'prior': 0.12758681,
        'color': (0, 255, 0) # Define el color BGR para 'platano'
    },
    'huevo': {
        'media': np.array([218.98420133, 219.09267887, 228.00140599]),
        'cov': np.array([[1936.42286759, 1745.21504661,  559.67223208],
 [1745.21504661, 1587.90871941,  514.00772564],
 [ 559.67223208,  514.00772564,  221.41496068]]),
        'prior': 0.08347222,
        'color': (0, 0, 255) # Define el color BGR para 'huevo'
    },
    'fondo': {
        'media': np.array([ 62.66210591,  92.54598323, 221.55021476]),
        'cov': np.array([[ 49.99447768,  65.42957457,  69.34246127],
 [ 65.42957457, 112.71187066,  85.10111673],
 [ 69.34246127,  85.10111673, 133.25592881]]),
        'prior': 0.74178611,
        'color': (255, 255, 255) # Define el color BGR para 'fondo'
    },
}

# Lista de clases para mantener el orden consistente
nombres_clases = list(parametros.keys())


# --- 2. CARGA Y PREPROCESAMIENTO DE LA IMAGEN DE PRUEBA ---
ruta_imagen_prueba = 'Prueba1.jpg'
imagen_prueba = cv2.imread(ruta_imagen_prueba)
if imagen_prueba is None:
    print(f"Error: No se pudo cargar la imagen de prueba: {ruta_imagen_prueba}")
    exit()

# Aplicamos el mismo filtro Gaussiano
imagen_suavizada = cv2.GaussianBlur(imagen_prueba, (9, 9), 0)
alto, ancho, _ = imagen_suavizada.shape


# --- 3. INICIALIZACIÓN DE IMÁGENES DE SALIDA ---
# Una imagen para la segmentación multi-clase coloreada
imagen_segmentada_color = np.zeros((alto, ancho, 3), dtype=np.uint8)

# Las máscaras individuales siguen siendo útiles para depuración
mascaras_individuales = {nombre: np.zeros((alto, ancho), dtype=np.uint8) for nombre in nombres_clases}


# --- 4. BUCLE PRINCIPAL DE CLASIFICACIÓN ---
print("Clasificando píxel por píxel... Esto puede tardar un momento.")

for y in range(alto):
    for x in range(ancho):
        pixel_bgr = imagen_suavizada[y, x]
        
        scores = []
        for nombre_clase in nombres_clases:
            params = parametros[nombre_clase]
            score = calcular_score(pixel_bgr, params['media'], params['cov'], params['prior'])
            scores.append(score)
            
        # Encontrar la clase con el score más alto
        clase_ganadora_idx = np.argmax(scores)
        clase_ganadora_nombre = nombres_clases[clase_ganadora_idx]
        
        # Pintar el píxel en la imagen de segmentación coloreada
        imagen_segmentada_color[y, x] = parametros[clase_ganadora_nombre]['color']
        
        # Pintar el píxel en la máscara individual correspondiente (para fines de depuración/visualización adicional)
        mascaras_individuales[clase_ganadora_nombre][y, x] = 255

print("¡Clasificación completada!")

# --- 5. VISUALIZACIÓN DE RESULTADOS ---
cv2.imshow('Imagen de Prueba Original', imagen_prueba)
cv2.imshow('Segmentacion Multi-Clase Coloreada', imagen_segmentada_color)

# Opcional: Mostrar las máscaras individuales

#for nombre_clase, mascara in mascaras_individuales.items():
#    cv2.imshow(f'Mascara para la clase: {nombre_clase}', mascara)

print("\nPresiona cualquier tecla en una ventana de imagen para salir.")
cv2.waitKey(0)
cv2.destroyAllWindows()

