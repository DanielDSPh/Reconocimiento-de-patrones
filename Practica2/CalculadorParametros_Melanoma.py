import numpy as np

# --- 1. PEGA TUS DATOS AQUÍ ---
# Usa los nombres de tus clases: 'melanoma' (blanco) y 'fondo' (negro)
datos_por_imagen = [
    {  # --- DATOS DE ENTRENAMIENTO1.JPG ---
        'total_pixeles_img': 270000,  # <-- Pega aquí el número total de píxeles de la imagen
        'melanoma': {
            'media': np.array([71.2770982, 78.85515266, 113.41832921]),
            'cov': np.array([
                [911.64669231, 907.30486449, 598.14679099],
                [907.30486449, 967.96863064, 655.51354551],
                [598.14679099, 655.51354551, 460.84628232]
            ]),
            'prior': 0.15260000,
            'color': (255, 255, 255)  # Color BGR de referencia
        },
        'fondo': {
            'media': np.array([136.76110805, 137.73850733, 138.49843967]),
            'cov': np.array([
                [970.00738927, 925.87843518, 845.89594582],
                [925.87843518, 903.07057416, 823.7991064],
                [845.89594582, 823.7991064, 762.14036435]
            ]),
            'prior': 0.84740000,
            'color': (0, 0, 0)  # Color BGR de referencia
        }
    },

    {  # --- DATOS DE ENTRENAMIENTO2.JPG ---
        'total_pixeles_img': 270000,
        'melanoma': {
            'media': np.array([74.35822341, 85.0124168, 110.07404091]),
            'cov': np.array([
                [1687.3061503, 1890.53198023, 1965.82752081],
                [1890.53198023, 2175.3978465, 2327.57723815],
                [1965.82752081, 2327.57723815, 2642.35800234]
            ]),
            'prior': 0.15301852,
            'color': (255, 255, 255)
        },
        'fondo': {
            'media': np.array([223.06871461, 222.97588823, 222.85592846]),
            'cov': np.array([
                [91.60505422, 80.73166418, 65.41360541],
                [80.73166418, 76.5118375, 62.6908642],
                [65.41360541, 62.6908642, 54.09706965]
            ]),
            'prior': 0.84698148,
            'color': (0, 0, 0)
        }
    },

    {  # --- DATOS DE ENTRENAMIENTO3.JPG ---
        'total_pixeles_img': 270000,
        'melanoma': {
            'media': np.array([83.01614955, 77.05387946, 108.52691829]),
            'cov': np.array([
                [1149.42369231, 1122.32469796, 866.63265088],
                [1122.32469796, 1194.86944225, 1053.29802084],
                [866.63265088, 1053.29802084, 1178.90015792]
            ]),
            'prior': 0.25915185,
            'color': (255, 255, 255)
        },
        'fondo': {
            'media': np.array([178.12092247, 177.79780432, 177.49764784]),
            'cov': np.array([
                [158.22270446, 171.96869293, 123.8855205],
                [171.96869293, 197.39427078, 144.26816929],
                [123.8855205, 144.26816929, 117.90902345]
            ]),
            'prior': 0.74084815,
            'color': (0, 0, 0)
        }
    }
]

# --- 2. PROCESO DE COMBINACIÓN ---
print("Combinando parámetros de múltiples fuentes...")

acumuladores = {
    'melanoma': {'N': 0, 'suma_medias_ponderadas': 0, 'suma_productos_externos': 0},
    'fondo': {'N': 0, 'suma_medias_ponderadas': 0, 'suma_productos_externos': 0}
}
total_pixeles_general = 0

for datos_img in datos_por_imagen:
    total_pixeles_general += datos_img['total_pixeles_img']
    for nombre_clase, params in datos_img.items():
        if nombre_clase == 'total_pixeles_img':
            continue

        if params['media'] is None or params['cov'] is None or params['prior'] is None:
            continue  # Saltar si los valores no están definidos aún

        media, cov, prior = params['media'], params['cov'], params['prior']
        N = int(round(prior * datos_img['total_pixeles_img']))
        if N == 0:
            continue

        # Calcular la suma de productos externos
        suma_xxT = N * (cov + np.outer(media, media))

        acumuladores[nombre_clase]['N'] += N
        acumuladores[nombre_clase]['suma_medias_ponderadas'] += N * media
        acumuladores[nombre_clase]['suma_productos_externos'] += suma_xxT

# --- 3. CÁLCULO DE PARÁMETROS FINALES ---
parametros_finales = {}

for nombre_clase, acum in acumuladores.items():
    N_final = acum['N']
    if N_final == 0:
        print(f"ADVERTENCIA: No se encontraron datos para la clase '{nombre_clase}'.")
        continue

    media_final = acum['suma_medias_ponderadas'] / N_final
    cov_final = (acum['suma_productos_externos'] / N_final) - np.outer(media_final, media_final)
    prior_final = N_final / total_pixeles_general

    parametros_finales[nombre_clase] = {
        'media': media_final,
        'cov': cov_final,
        'prior': prior_final
    }
    print(f"-> Parámetros finales para '{nombre_clase}' calculados.")

# --- 4. IMPRESIÓN DEL RESULTADO FINAL ---
print("\n" + "=" * 60)
print("--- CÓDIGO FINAL LISTO PARA USAR EN EL CLASIFICADOR ---")
print("=" * 60 + "\n")
print("parametros = {")
for nombre_clase, params in parametros_finales.items():
    print(f"    '{nombre_clase}': {{")
    print(f"        'media': np.array({np.array2string(params['media'], separator=', ')}),")
    print(f"        'cov': np.array({np.array2string(params['cov'], separator=', ')}),")
    print(f"        'prior': {params['prior']:.8f},")
    color_ref = (255, 255, 255) if nombre_clase == 'melanoma' else (0, 0, 0)
    print(f"        'color': {color_ref}  # Color BGR para '{nombre_clase}'")
    print("    },")
print("}")
