import cv2
import os
import numpy as np

print("Iniciando script para generar máscaras v2 (con limpieza avanzada)...")

# === CONFIGURACIÓN DE RUTAS ===
CARPETA_ENTRADA = "P/UP"  # Carpeta con imágenes originales
CARPETA_SALIDA = "P/mascaras_limpias2"  # Nueva carpeta para máscaras mejoradas

# === PARÁMETROS DE PROCESAMIENTO (Ajusta estos valores) ===

# 1. Desenfoque
BLUR_KERNEL_SIZE = (5, 5)

# 2. Umbralización Adaptativa
ADAPTIVE_BLOCK_SIZE = 21  # Debe ser impar
ADAPTIVE_C = 8           # Constante a restar (valor más alto = menos detección)

# 3. Limpieza Morfológica
KERNEL_OPEN = np.ones((3, 3), np.uint8) # Kernel para quitar ruido
KERNEL_CLOSE = np.ones((7, 7), np.uint8) # Kernel para rellenar grietas

# 4. Filtro de Contornos
MIN_AREA_GRIETA = 5000  # <-- PARÁMETRO CLAVE
                     # Elimina cualquier "objeto" detectado con menos píxeles que este valor.
                     # Sube este valor para eliminar más ruido.

# --- Fin de la Configuración ---

if not os.path.exists(CARPETA_SALIDA):
    os.makedirs(CARPETA_SALIDA)
    print(f"Carpeta de salida creada en: {CARPETA_SALIDA}")

try:
    lista_imagenes = [f for f in os.listdir(CARPETA_ENTRADA) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not lista_imagenes:
        print(f"❌ Error: No se encontraron imágenes en '{CARPETA_ENTRADA}'.")
        exit()
    print(f"Se encontraron {len(lista_imagenes)} imágenes para procesar.")
except FileNotFoundError:
    print(f"❌ Error: La carpeta de entrada '{CARPETA_ENTRADA}' no existe.")
    exit()

total_procesadas = 0

for nombre_archivo in lista_imagenes:
    ruta_completa = os.path.join(CARPETA_ENTRADA, nombre_archivo)
    
    img = cv2.imread(ruta_completa)
    if img is None:
        continue

    # 1. Pre-procesamiento
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gris, BLUR_KERNEL_SIZE, 0)
    
    # 2. Detección (Genera la máscara "sucia")
    mascara_sucia = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )
    
    # 3. LIMPIEZA MORFOLÓGICA
    # Paso 3a: Quitar ruido (puntos blancos pequeños)
    mascara_limpia = cv2.morphologyEx(mascara_sucia, cv2.MORPH_OPEN, KERNEL_OPEN, iterations=1)
    # Paso 3b: Rellenar huecos en las grietas (puntos negros pequeños)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=1)
    
    # 4. FILTRADO POR CONTORNOS (TAMAÑO)
    # Crear una máscara final, completamente negra, para dibujar solo las grietas "buenas"
    mascara_final = np.zeros_like(mascara_limpia)
    
    # Encontrar todos los "objetos" blancos (contornos) en la máscara limpia
    contours, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Calcular el área de cada objeto
        area = cv2.contourArea(cnt)
        
        # Si el área es más grande que nuestro mínimo, la consideramos una grieta real
        if area > MIN_AREA_GRIETA:
            # Dibujamos ese objeto en nuestra máscara final
            cv2.drawContours(mascara_final, [cnt], -1, (255), thickness=cv2.FILLED)
            
    # 5. Guardar la máscara final y filtrada
    ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
    cv2.imwrite(ruta_salida, mascara_final)
    
    total_procesadas += 1

print(f"\n✅ Proceso completado.")
print(f"Se generaron {total_procesadas} máscaras limpias en la carpeta '{CARPETA_SALIDA}'.")