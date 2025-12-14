import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.preprocessing import image
import os
import random  # Para severidad aleatoria

# === FUNCIÓN PARA CARGAR Y PREPARAR LA IMAGEN ===
def preparar_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=(256, 256))
    img_array = image.img_to_array(img)
    return np.expand_dims(img_array, axis=0)

# === FUNCIÓN PARA SEVERIDAD ===
def severidad_aleatoria():
    return random.choice(["Fisura leve", "Fisura estándar", "Fisura grave"])

# === FUNCIÓN PARA INTERPRETAR PREDICCIÓN ===
def interpretar_prediccion(prediccion):
    confianza = prediccion[0][0]

    if confianza > 0.5:
        clase_predicha = 1  # UD (sin grieta)
        confianza_final = confianza
    else:
        clase_predicha = 0  # CD (con grieta)
        confianza_final = 1 - confianza

    resultado = nombres_clases[clase_predicha]
    porcentaje_confianza = confianza_final * 100

    # Si detecta grieta → agregar severidad aleatoria
    if clase_predicha == 0:
        severidad = severidad_aleatoria()
        resultado_texto = f"{resultado} - {severidad}"
    else:
        resultado_texto = resultado

    print(f"Predicción: {resultado_texto}")
    print(f"Confianza: {porcentaje_confianza:.2f}%\n")

    return resultado_texto, porcentaje_confianza


# === CARGAR SOLO EL MODELO AVANZADO ===
modelo_avanzado = None

if os.path.exists('modelo_fisuras_avanzado.keras'):
    modelo_avanzado = tf.keras.models.load_model('modelo_fisuras_avanzado.keras')
    print("✅ Modelo Avanzado cargado correctamente.")
else:
    print("❌ No se encontró 'modelo_fisuras_avanzado.keras'. Saliendo.")
    exit()

# === SELECCIONAR LA IMAGEN ===
root = tk.Tk()
root.withdraw()

print("\n🖼 Selecciona una imagen para analizar...")
ruta_archivo = filedialog.askopenfilename(
    title="Selecciona una imagen",
    filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")]
)

if not ruta_archivo:
    print("❌ No se seleccionó imagen. Saliendo.")
    exit()

print(f"\nImagen seleccionada: {ruta_archivo}")
imagen_preparada = preparar_imagen(ruta_archivo)

# Nombres de clases
nombres_clases = {0: 'CD (con grieta)', 1: 'UD (sin grieta)'}

print("\n========== RESULTADOS ==========")

prediccion = modelo_avanzado.predict(imagen_preparada)
resultado_final, confianza_final = interpretar_prediccion(prediccion)

print("================================")

# === MOSTRAR IMAGEN CON RESULTADO ===
try:
    import matplotlib.pyplot as plt
    img_mostrar = plt.imread(ruta_archivo)
    plt.imshow(img_mostrar)
    plt.title(f"{resultado_final} ({confianza_final:.0f}%)")
    plt.axis('off')
    plt.show()
except ImportError:
    print("\n(Instala matplotlib para mostrar imágenes)")