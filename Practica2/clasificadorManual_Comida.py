import cv2
import numpy as np

# Variables globales para el estado del dibujo de la clase actual
puntos_contorno_actual = []
contornos_clase_actual = []
mouse_pos = (0, 0)

def definir_contornos(event, x, y, flags, param):
    global puntos_contorno_actual, contornos_clase_actual, mouse_pos
    copia_imagen = param['copia_imagen']
    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        puntos_contorno_actual.append((x, y))
        copia_imagen[y, x] = (0, 255, 255) # Vértice amarillo

    elif event == cv2.EVENT_RBUTTONDOWN and len(puntos_contorno_actual) > 2:
        contornos_clase_actual.append(np.array(puntos_contorno_actual, dtype=np.int32))
        pts = contornos_clase_actual[-1]
        
        # Rellena con el color de la clase actual
        color_clase = param['color_clase']
        overlay = copia_imagen.copy()
        cv2.fillPoly(overlay, [pts], color_clase)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, copia_imagen, 1 - alpha, 0, copia_imagen)
        cv2.polylines(copia_imagen, [pts], isClosed=True, color=color_clase, thickness=2)
        
        puntos_contorno_actual = []

# --- SCRIPT PRINCIPAL ---

ruta_imagen = 'Entrenamiento1.jpg'
imagen_bgr = cv2.imread(ruta_imagen)
if imagen_bgr is None: exit(f"Error: No se pudo cargar la imagen: {ruta_imagen}")

print("Aplicando filtro Gaussiano...")
imagen_suavizada = cv2.GaussianBlur(imagen_bgr, (5, 5), 0)

# --- CONFIGURACIÓN DE CLASES Y PROCESO ---
clases_a_definir = {
    'chile': (255, 0, 0),    # Azul
    'platano': (0, 255, 0),  # Verde
    'huevo': (0, 0, 255)     # Rojo
}
mascaras_finales = {}
mascara_acumulada = np.zeros(imagen_suavizada.shape[:2], dtype=np.uint8)

copia_para_dibujar = imagen_suavizada.copy()

# Bucle principal para definir cada clase
for nombre_clase, color_clase in clases_a_definir.items():
    puntos_contorno_actual = []
    contornos_clase_actual = []
    
    nombre_ventana = f"Define '{nombre_clase}' (ENTER para guardar, 'r' para reiniciar esta clase)"
    cv2.namedWindow(nombre_ventana)
    parametros_callback = {
        'copia_imagen': copia_para_dibujar,
        'color_clase': color_clase
    }
    cv2.setMouseCallback(nombre_ventana, definir_contornos, parametros_callback)

    print("\n" + "="*50)
    print(f"Ahora, define los contornos para la clase: {nombre_clase.upper()}")
    print("Clic Izq=Punto, Clic Der=Cerrar, 'r'=Reset, ENTER=Guardar y continuar")
    
    imagen_limpia_clase_actual = copia_para_dibujar.copy()

    while True:
        imagen_temporal = copia_para_dibujar.copy()
        if puntos_contorno_actual:
            cv2.line(imagen_temporal, puntos_contorno_actual[-1], mouse_pos, (255, 255, 0), 1)
        cv2.imshow(nombre_ventana, imagen_temporal)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 13: # ENTER
            break
        elif k == ord('r'):
            puntos_contorno_actual = []
            contornos_clase_actual = []
            copia_para_dibujar[:] = imagen_limpia_clase_actual[:]
            print(f"Contornos para '{nombre_clase}' reiniciados.")
            
    cv2.destroyAllWindows()
    
    # Crear y guardar la máscara para la clase recién definida
    if contornos_clase_actual:
        mascara_temporal = np.zeros(imagen_suavizada.shape[:2], dtype=np.uint8)
        for contorno in contornos_clase_actual:
            cv2.fillPoly(mascara_temporal, [contorno], 255)
        
        # Asegurarse de no solapar con clases ya definidas
        mascara_no_solapada = cv2.bitwise_and(mascara_temporal, cv2.bitwise_not(mascara_acumulada))
        mascaras_finales[nombre_clase] = mascara_no_solapada
        
        # Actualizar la máscara acumulada con los nuevos píxeles
        mascara_acumulada = cv2.bitwise_or(mascara_acumulada, mascara_no_solapada)

# --- CÁLCULO DE PARÁMETROS FINALES ---

# Añadir la clase 'fondo'
mascaras_finales['fondo'] = cv2.bitwise_not(mascara_acumulada)

parametros_finales = {}
alto, ancho, _ = imagen_suavizada.shape
pixeles_totales = alto * ancho

print("\nCalculando parámetros para todas las clases...")
for nombre, mascara in mascaras_finales.items():
    pixeles_clase = imagen_suavizada[mascara > 0]
    
    if len(pixeles_clase) == 0: continue
    
    prior = len(pixeles_clase) / pixeles_totales
    media = np.mean(pixeles_clase, axis=0)
    covarianza = np.cov(pixeles_clase, rowvar=False)
    
    parametros_finales[nombre] = {'media': media, 'cov': covarianza, 'prior': prior}
    print(f" -> Parámetros para '{nombre}' calculados.")

# --- IMPRESIÓN DEL CÓDIGO FINAL ---
print("\n" + "="*60)
print("--- CÓDIGO LISTO PARA COPIAR Y PEGAR ---")
print("="*60 + "\n")
print("parametros = {")
for nombre_clase, params in parametros_finales.items():
    color_a_usar = clases_a_definir.get(nombre_clase, (255, 255, 255)) # Fondo blanco por defecto
    print(f"    '{nombre_clase}': {{")
    print(f"        'media': np.array({np.array2string(params['media'], separator=', ')}),")
    print(f"        'cov': np.array({np.array2string(params['cov'], separator=', ')}),")
    print(f"        'prior': {params['prior']:.8f},")
    print(f"        'color': {color_a_usar} # Color BGR para '{nombre_clase}'")
    print("    },")
print("}")

'''
parametros = {
    'chile': {
        'media': np.array([23.12446352, 81.24326678, 57.81936929]),
        'cov': np.array([[ 683.14916326,  692.03428858,  421.49281177],
 [ 692.03428858, 1030.58544312,  292.77870408],
 [ 421.49281177,  292.77870408,  718.59974175]]),
        'prior': 0.04465833,
        'color': (255, 0, 0) # Color BGR para 'chile'
    },
    'platano': {
        'media': np.array([ 52.94634232, 184.83491793, 210.90542697]),
        'cov': np.array([[ 413.07202462,  433.85844823,  302.76770028],
 [ 433.85844823, 1046.52159764,  674.45046533],
 [ 302.76770028,  674.45046533,  515.88507131]]),
        'prior': 0.12709167,
        'color': (0, 255, 0) # Color BGR para 'platano'
    },
    'huevo': {
        'media': np.array([220.73185185, 217.71505051, 227.3469697 ]),
        'cov': np.array([[1793.83219866, 1626.4825638 ,  506.59164506],
 [1626.4825638 , 1480.54619591,  464.22648589],
 [ 506.59164506,  464.22648589,  192.21642066]]),
        'prior': 0.08250000,
        'color': (0, 0, 255) # Color BGR para 'huevo'
    },
    'fondo': {
        'media': np.array([ 60.42973144,  86.16498305, 220.35783514]),
        'cov': np.array([[ 46.97298341,  55.98383763,  66.68466214],
 [ 55.98383763,  83.50171307,  80.48737623],
 [ 66.68466214,  80.48737623, 132.80341303]]),
        'prior': 0.74575000,
        'color': (255, 255, 255) # Color BGR para 'fondo'
    },
}
'''

'''
parametros = {
    'chile': {
        'media': np.array([23.77678995, 82.04113324, 60.92259365]),
        'cov': np.array([[ 590.93079195,  635.40583888,  351.56140642],
 [ 635.40583888, 1001.25253118,  198.22160631],
 [ 351.56140642,  198.22160631, 1012.93028909]]),
        'prior': 0.04686667,
        'color': (255, 0, 0) # Color BGR para 'chile'
    },
    'platano': {
        'media': np.array([ 58.4637342 , 188.36830133, 212.89450545]),
        'cov': np.array([[ 444.44949081,  486.93181188,  257.5592797 ],
 [ 486.93181188, 1188.33577318,  563.41521354],
 [ 257.5592797 ,  563.41521354,  337.75313778]]),
        'prior': 0.12699444,
        'color': (0, 255, 0) # Color BGR para 'platano'
    },
    'huevo': {
        'media': np.array([219.207272  , 218.13561749, 226.26810508]),
        'cov': np.array([[2096.32627637, 1870.14816799,  621.20095384],
 [1870.14816799, 1674.83746104,  558.52154395],
 [ 621.20095384,  558.52154395,  240.21554402]]),
        'prior': 0.08342500,
        'color': (0, 0, 255) # Color BGR para 'huevo'
    },
    'fondo': {
        'media': np.array([ 62.61113708,  94.31186302, 219.71527469]),
        'cov': np.array([[ 49.21254621,  63.73240057,  66.5560157 ],
 [ 63.73240057, 111.6811393 ,  81.36978196],
 [ 66.5560157 ,  81.36978196, 123.53335059]]),
        'prior': 0.74271389,
        'color': (255, 255, 255) # Color BGR para 'fondo'
    },
}
'''

'''
parametros = {
    'chile': {
        'media': np.array([19.34057312, 80.01961565, 64.32465317]),
        'cov': np.array([[ 367.05270751,  377.00361075,  284.75256311],
 [ 377.00361075,  750.54837281,  -74.56786293],
 [ 284.75256311,  -74.56786293, 1256.25042534]]),
        'prior': 0.04885556,
        'color': (255, 0, 0) # Color BGR para 'chile'
    },
    'platano': {
        'media': np.array([ 60.71072233, 190.96024597, 215.08535424]),
        'cov': np.array([[ 438.59063401,  440.99748579,  248.63324887],
 [ 440.99748579, 1033.13693375,  522.0494275 ],
 [ 248.63324887,  522.0494275 ,  334.56171175]]),
        'prior': 0.12828889,
        'color': (0, 255, 0) # Color BGR para 'platano'
    },
    'huevo': {
        'media': np.array([218.29364791, 220.94865534, 227.55819172]),
        'cov': np.array([[1811.80734646, 1618.0260664 ,  609.01200013],
 [1618.0260664 , 1462.79336417,  557.87082716],
 [ 609.01200013,  557.87082716,  272.98018839]]),
        'prior': 0.08418056,
        'color': (0, 0, 255) # Color BGR para 'huevo'
    },
    'fondo': {
        'media': np.array([ 63.40681325,  95.11102838, 221.81034736]),
        'cov': np.array([[ 42.48350846,  54.53483668,  58.23862942],
 [ 54.53483668,  96.38250648,  66.74258051],
 [ 58.23862942,  66.74258051, 114.52427497]]),
        'prior': 0.73867500,
        'color': (255, 255, 255) # Color BGR para 'fondo'
    },
}
'''

'''
parametros = {
    'chile': {
        'media': np.array([20.39704019, 75.7613728 , 55.68864448]),
        'cov': np.array([[ 535.61298059,  552.45069591,  406.39975792],
 [ 552.45069591,  973.07060921,  240.552591  ],
 [ 406.39975792,  240.552591  , 1040.04316157]]),
        'prior': 0.04823889,
        'color': (255, 0, 0) # Color BGR para 'chile'
    },
    'platano': {
        'media': np.array([ 56.71573692, 188.23169091, 214.39233775]),
        'cov': np.array([[ 466.70605593,  537.343685  ,  329.03473171],
 [ 537.343685  , 1358.60537306,  738.33324333],
 [ 329.03473171,  738.33324333,  495.29660933]]),
        'prior': 0.12797222,
        'color': (0, 255, 0) # Color BGR para 'platano'
    },
    'huevo': {
        'media': np.array([217.73503083, 219.53739805, 230.81702142]),
        'cov': np.array([[2037.71965009, 1869.94863848,  505.82137977],
 [1869.94863848, 1726.338726  ,  471.66675176],
 [ 505.82137977,  471.66675176,  168.09811151]]),
        'prior': 0.08378333,
        'color': (0, 0, 255) # Color BGR para 'huevo'
    },
    'fondo': {
        'media': np.array([ 64.2195967 ,  94.64374141, 224.33384134]),
        'cov': np.array([[ 53.31752649,  68.08813212,  78.59618751],
 [ 68.08813212, 104.35208655, 100.90026769],
 [ 78.59618751, 100.90026769, 149.53983464]]),
        'prior': 0.74000556,
        'color': (255, 255, 255) # Color BGR para 'fondo'
    },
}
'''


'''
'''