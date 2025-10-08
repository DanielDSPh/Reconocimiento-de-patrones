import cv2
import numpy as np

# --- CARGA DE IMAGEN Y MÁSCARA ---
ruta_imagen = "Entrenamiento2.jpg"
ruta_mascara_melanoma = "Entrenamiento3Mascara.png"  # blanco = melanoma, negro = fondo

imagen = cv2.imread(ruta_imagen)
if imagen is None:
    exit(f"Error: No se pudo cargar la imagen '{ruta_imagen}'")

mascara_melanoma = cv2.imread(ruta_mascara_melanoma, cv2.IMREAD_GRAYSCALE)
if mascara_melanoma is None:
    exit(f"Error: No se pudo cargar la máscara '{ruta_mascara_melanoma}'")

# Asegurar máscara binaria: 255 = melanoma, 0 = fondo
_, mascara_melanoma = cv2.threshold(mascara_melanoma, 127, 255, cv2.THRESH_BINARY)

# Crear máscara del fondo automáticamente
mascara_fondo = cv2.bitwise_not(mascara_melanoma)

# --- DEFINICIÓN DE CLASES ---
mascaras = {
    "melanoma": mascara_melanoma,
    "fondo": mascara_fondo
}

colores_clase = {
    "melanoma": (255, 255, 255),  # para referencia visual, blanco
    "fondo": (0, 0, 0)            # negro
}

# --- CÁLCULO DE PARÁMETROS ---
alto, ancho, _ = imagen.shape
pixeles_totales = alto * ancho
parametros_finales = {}

print("\nCalculando parámetros para cada clase...\n")

for nombre, mascara in mascaras.items():
    pixeles_clase = imagen[mascara > 0]  # Extrae los píxeles de la clase

    if len(pixeles_clase) == 0:
        print(f"Advertencia: no se encontraron píxeles para la clase '{nombre}'")
        continue

    prior = len(pixeles_clase) / pixeles_totales
    media = np.mean(pixeles_clase, axis=0)
    covarianza = np.cov(pixeles_clase, rowvar=False)

    parametros_finales[nombre] = {
        "media": media,
        "cov": covarianza,
        "prior": prior
    }

    print(f" -> Clase '{nombre}' calculada: {len(pixeles_clase)} píxeles")

# --- RESULTADOS FINALES ---
print("\n" + "="*60)
print("--- PARÁMETROS LISTOS PARA USAR ---")
print("="*60 + "\n")
print("parametros = {")
for nombre, params in parametros_finales.items():
    color = colores_clase[nombre]
    print(f"    '{nombre}': {{")
    print(f"        'media': np.array({np.array2string(params['media'], separator=', ')}),")
    print(f"        'cov': np.array({np.array2string(params['cov'], separator=', ')}),")
    print(f"        'prior': {params['prior']:.8f},")
    print(f"        'color': {color}  # Color BGR de referencia")
    print("    },")
print("}")

'''
--------------------------------
Entrenamiento 1
--------------------------------
parametros = {
    'melanoma': {
        'media': np.array([ 71.2770982 ,  78.85515266, 113.41832921]),
        'cov': np.array([[911.64669231, 907.30486449, 598.14679099],
 [907.30486449, 967.96863064, 655.51354551],
 [598.14679099, 655.51354551, 460.84628232]]),
        'prior': 0.15260000,
        'color': (255, 255, 255)  # Color BGR de referencia
    },
    'fondo': {
        'media': np.array([136.76110805, 137.73850733, 138.49843967]),
        'cov': np.array([[970.00738927, 925.87843518, 845.89594582],
 [925.87843518, 903.07057416, 823.7991064 ],
 [845.89594582, 823.7991064 , 762.14036435]]),
        'prior': 0.84740000,
        'color': (0, 0, 0)  # Color BGR de referencia
    },
}

--------------------------------
Entrenamiento 2
--------------------------------

parametros = {
    'melanoma': {
        'media': np.array([ 74.35822341,  85.0124168 , 110.07404091]),
        'cov': np.array([[1687.3061503 , 1890.53198023, 1965.82752081],
 [1890.53198023, 2175.3978465 , 2327.57723815],
 [1965.82752081, 2327.57723815, 2642.35800234]]),
        'prior': 0.15301852,
        'color': (255, 255, 255)  # Color BGR de referencia
    },
    'fondo': {
        'media': np.array([223.06871461, 222.97588823, 222.85592846]),
        'cov': np.array([[91.60505422, 80.73166418, 65.41360541],
 [80.73166418, 76.5118375 , 62.6908642 ],
 [65.41360541, 62.6908642 , 54.09706965]]),
        'prior': 0.84698148,
        'color': (0, 0, 0)  # Color BGR de referencia
    },
}

-------------------------------------
Entrenamiento 3
-------------------------------------
parametros = {
    'melanoma': {
        'media': np.array([ 83.01614955,  77.05387946, 108.52691829]),
        'cov': np.array([[1149.42369231, 1122.32469796,  866.63265088],
 [1122.32469796, 1194.86944225, 1053.29802084],
 [ 866.63265088, 1053.29802084, 1178.90015792]]),
        'prior': 0.25915185,
        'color': (255, 255, 255)  # Color BGR de referencia
    },
    'fondo': {
        'media': np.array([178.12092247, 177.79780432, 177.49764784]),
        'cov': np.array([[158.22270446, 171.96869293, 123.8855205 ],
 [171.96869293, 197.39427078, 144.26816929],
 [123.8855205 , 144.26816929, 117.90902345]]),
        'prior': 0.74084815,
        'color': (0, 0, 0)  # Color BGR de referencia
    },
}

'''