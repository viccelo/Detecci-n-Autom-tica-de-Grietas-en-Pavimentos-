import tensorflow as tf
import matplotlib.pyplot as plt
import os

# === CONFIGURACIÓN GENERAL (Debe ser la misma que tu script principal) ===
BASE_DIR = "D"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
IMG_SIZE = (180, 180)
BATCH_SIZE = 32 # Usamos 32 para que un solo lote contenga al menos 25 imágenes

# === FUNCIÓN DE NORMALIZACIÓN (Opcional para visualizar, pero buena práctica) ===
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32) / 255.0
    return imagenes, etiquetas

# === CARGAR LOS DATOS DE ENTRENAMIENTO ===
# Solo necesitamos cargar los datos de entrenamiento para esta visualización
datos_entrenamiento = (
    tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    .map(normalizar)
)

print("✅ Datos de entrenamiento cargados.")

# === DEFINIR LOS NOMBRES DE LAS CLASES ===
# Creamos un diccionario para que sea más fácil leer las etiquetas
# 'UD' (Sin Grieta) será la clase 0 y 'CD' (Con Grieta) será la clase 1
# Nota: image_dataset_from_directory ordena las clases alfabéticamente.
# 'CD' es la clase 0 y 'UD' es la clase 1.
nombres_clases = {0: 'CD (con grieta)', 1: 'UD (sin grieta)'}


# === VISUALIZAR 25 IMÁGENES DEL DATASET 🖼️ ===
plt.figure(figsize=(10, 10))

# Tomamos solo UN lote de datos. Como el batch_size es 32, tendremos suficientes imágenes.
for imagenes, etiquetas in datos_entrenamiento.take(1):
    # Ahora recorremos las primeras 25 imágenes de ESE lote
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # Mostramos la imagen. No es necesario convertirla a escala de grises.
        # Matplotlib puede mostrar imágenes a color (3 canales) directamente.
        plt.imshow(imagenes[i])
        
        # Obtenemos la etiqueta de la imagen y la mostramos como título
        # tf.keras.utils ordena las carpetas alfabéticamente: CD (0), UD (1)
        # Revisa tus carpetas. Si 'CD' es 0 y 'UD' es 1, esta lógica es correcta.
        # Si es al revés, puedes cambiar el diccionario de nombres_clases.
        nombre_etiqueta = nombres_clases[int(etiquetas[i])]
        plt.xlabel(nombre_etiqueta)

plt.suptitle("Muestra de 25 Imágenes de Entrenamiento", fontsize=16)
plt.show()