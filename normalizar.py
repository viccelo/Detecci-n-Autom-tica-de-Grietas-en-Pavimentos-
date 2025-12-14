import tensorflow as tf
import matplotlib.pyplot as plt
import os

# === CONFIGURACIÓN GENERAL ===
BASE_DIR = "D"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

IMG_SIZE = (180, 180)
BATCH_SIZE = 32

# === FUNCIÓN DE NORMALIZACIÓN ===
def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32) / 255.0
    return imagenes, etiquetas

# === CARGAR Y PREPARAR LOS DATOS ===
datos_entrenamiento = (
    tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    .map(normalizar)
    .cache()
    .shuffle(1000)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

datos_pruebas = (
    tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )
    .map(normalizar)
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

print("✅ Datos cargados, normalizados y preparados en caché.")

# === OBTENER UNA IMAGEN DE PRUEBA ===
for imagenes, etiquetas in datos_pruebas.take(1):
    imagen = imagenes[0]
    etiqueta = etiquetas[0].numpy()
    break

# === CONVERTIR A ESCALA DE GRISES ===
imagen_gris = tf.image.rgb_to_grayscale(imagen)

# === MOSTRAR / GUARDAR LA IMAGEN ===
plt.figure(figsize=(4, 4))
plt.imshow(imagen_gris.numpy().squeeze(), cmap='gray')
plt.title(f"Etiqueta: {'CD (con grieta)' if etiqueta == 1 else 'UD (sin grieta)'}")
plt.axis("off")

# Intentar mostrar, si no funciona guardar
try:
    plt.show()
except Exception as e:
    plt.savefig("imagen_prueba.png")
    print("⚠️ No se pudo abrir la ventana de imagen, se guardó como imagen_prueba.png")