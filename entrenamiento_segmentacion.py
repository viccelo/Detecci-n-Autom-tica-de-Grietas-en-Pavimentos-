import tensorflow as tf
import os, glob
from tensorflow.keras import backend as K
from modelo_segmentacion import crear_modelo_unet

# --- 1. CONFIGURACIÓN ---
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCAS = 25

IMG_CON_GRIETA_DIR = os.path.join("P", "CP")
IMG_SIN_GRIETA_DIR = os.path.join("P", "UP")
MASCARAS_CON_GRIETA_DIR = os.path.join("P", "mascaras_limpias")
MASCARAS_SIN_GRIETA_DIR = os.path.join("P", "mascaras_limpias2")

# --- 2. FUNCIONES DE PÉRDIDA Y MÉTRICA ---
def dice_coeff(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

# --- 3. CARGA DE ARCHIVOS ---
def buscar_archivos(*carpetas):
    archivos = []
    for carpeta in carpetas:
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            archivos.extend(glob.glob(os.path.join(carpeta, ext)))
    return sorted(archivos)

def load_and_preprocess(ruta_imagen, ruta_mascara):
    img = tf.io.read_file(ruta_imagen)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(ruta_mascara)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0
    return img, mask

# --- 4. DATA AUGMENTATION ---
@tf.function
def augment(img, mask):
    """Aumenta los datos de forma aleatoria y sincronizada entre imagen y máscara."""
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_brightness(img, max_delta=0.1)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, mask

print("📂 Preparando datasets...")

lista_imagenes = buscar_archivos(IMG_CON_GRIETA_DIR, IMG_SIN_GRIETA_DIR)
lista_mascaras = buscar_archivos(MASCARAS_CON_GRIETA_DIR, MASCARAS_SIN_GRIETA_DIR)

if len(lista_imagenes) != len(lista_mascaras):
    raise ValueError(f"Número de imágenes ({len(lista_imagenes)}) ≠ máscaras ({len(lista_mascaras)})")

total_datos = len(lista_imagenes)
tamano_val = int(total_datos * 0.2)

full_dataset = tf.data.Dataset.from_tensor_slices((lista_imagenes, lista_mascaras))
full_dataset = full_dataset.shuffle(total_datos, reshuffle_each_iteration=False)

train_dataset = full_dataset.skip(tamano_val)
val_dataset = full_dataset.take(tamano_val)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = (
    train_dataset
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .map(augment, num_parallel_calls=AUTOTUNE)
    # .cache()  # Actívalo si tienes suficiente RAM
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print(f"✅ Datasets listos: {total_datos - tamano_val} entrenamiento, {tamano_val} validación.")

# --- 5. CREAR Y COMPILAR EL MODELO ---
modelo = crear_modelo_unet(
    input_size=(IMG_SIZE[0], IMG_SIZE[1], 3),
    base_filters=32,
    use_bn=True,
    dropout_rate=0.3
)

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_loss,
    metrics=[dice_coeff]
)

# --- 6. ENTRENAMIENTO ---
print("🚀 Iniciando entrenamiento con aumento de datos y dropout...")
historial = modelo.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCAS
)

# --- 7. GUARDAR ---
modelo.save('modelo_segmentacion_fisuras_final.keras')
print("🎉 ¡Entrenamiento finalizado y modelo guardado como 'modelo_segmentacion_fisuras_final.keras'!")
