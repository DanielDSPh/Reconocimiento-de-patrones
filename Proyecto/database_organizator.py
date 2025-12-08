import os
import shutil

# --- CONFIGURACIÓN ---
RUTA_BASE = "Equipo"
RUTA_SALIDA = "Equipo_Fusionado"

# Mapeo de nombres a códigos de sujeto
MAPEO_SUJETOS = {
    "Daniel": "S0",
    "Bryan": "S1",
    "Ricardo": "S2",
    "Scarlett": "S3"
}

# CAS posibles
EMOCIONES = ["confused", "distracted", "fatigued", "joyful", "neutral"]


# --- CREAR CARPETAS DE SALIDA ---
os.makedirs(RUTA_SALIDA, exist_ok=True)
for emocion in EMOCIONES:
    os.makedirs(os.path.join(RUTA_SALIDA, emocion), exist_ok=True)


# --- PROCESAMIENTO PRINCIPAL ---
for sujeto in MAPEO_SUJETOS.keys():
    ruta_sujeto = os.path.join(RUTA_BASE, sujeto)
    
    if not os.path.isdir(ruta_sujeto):
        print(f"Advertencia: No existe carpeta para {sujeto}")
        continue

    codigo_sujeto = MAPEO_SUJETOS[sujeto]

    print(f"\nProcesando sujeto: {sujeto} → {codigo_sujeto}")

    for emocion in EMOCIONES:
        ruta_emocion = os.path.join(ruta_sujeto, emocion)

        if not os.path.isdir(ruta_emocion):
            print(f"  - No existe carpeta {emocion} para {sujeto}, se omite.")
            continue

        # Recorrer todos los archivos dentro de la emoción
        for archivo in os.listdir(ruta_emocion):
            if not archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            ruta_archivo = os.path.join(ruta_emocion, archivo)

            # Renombrar al nuevo formato
            partes = archivo.split("_")

            # Esperado: [Nombre, Clip, Frame, Emocion]
            if len(partes) >= 4:
                clip = partes[1]
                frame = partes[2]
                emocion_tag = partes[3].split(".")[0]
                
                nuevo_nombre = f"{codigo_sujeto}_{clip}_{frame}_{emocion_tag}.png"
            else:
                # Si no cumple formato, se deja como está pero con prefijo del sujeto
                nuevo_nombre = f"{codigo_sujeto}_{archivo}"

            ruta_salida_emocion = os.path.join(RUTA_SALIDA, emocion, nuevo_nombre)

            # Copiar archivo renombrado a la carpeta correspondiente
            shutil.copy2(ruta_archivo, ruta_salida_emocion)

print("\nPROCESO FINALIZADO.")
print(f"Frames reorganizados y renombrados en: {RUTA_SALIDA}")
