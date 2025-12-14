import tensorflow as tf
import matplotlib.pyplot as plt
import os

# === 1. CONFIGURACIÓN ===
# Mantenemos la misma configuración base.
BASE_DIR = "D"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# === 2. CAPA DE AUMENTO DE DATOS (DATA AUGMENTATION) ===
# Creamos un pequeño "modelo" secuencial que solo se encargará de transformar
# las imágenes de entrenamiento de forma aleatoria.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# === 3. CARGA DE DATOS ===
# Cargamos los datos como antes.
datos_entrenamiento = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

datos_pruebas = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# --- Aplicar Aumento de Datos SOLO al set de entrenamiento ---
# Usamos .map para aplicar nuestra capa de aumento a cada imagen de entrenamiento.
datos_entrenamiento = datos_entrenamiento.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Optimizamos el set de pruebas también.
datos_pruebas = datos_pruebas.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("✅ Datos cargados y pipeline de aumento de datos preparada.")

# === 4. CREACIÓN DEL MODELO AVANZADO CON DROPOUT 🧠 ===
modelo_avanzado = tf.keras.Sequential([
    # Capa de entrada: normaliza y define el tamaño de las imágenes.
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Bloques Convolucionales
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    # Capa de Aplanado
    tf.keras.layers.Flatten(),
    
    # --- Capa Densa con Dropout ---
    # La capa densa aprende las combinaciones de alto nivel.
    tf.keras.layers.Dense(256, activation='relu'),
    # AÑADIMOS DROPOUT: "Apagamos" el 50% de las neuronas en cada paso de entrenamiento.
    tf.keras.layers.Dropout(0.5),
    
    # Capa de Salida
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# === 5. COMPILAR EL MODELO ===
modelo_avanzado.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Modelo avanzado creado y compilado.")
modelo_avanzado.summary()

# === 6. ENTRENAR EL MODELO ===
# Entrenamos por más épocas, ya que el aumento de datos crea un problema más "difícil"
# pero que resultará en un mejor modelo.
EPOCAS = 40

historial = modelo_avanzado.fit(
    datos_entrenamiento,
    epochs=EPOCAS,
    validation_data=datos_pruebas
)

# === 7. GUARDAR EL NUEVO MODELO 💾 ===
# Guardamos este modelo con un nuevo nombre para no sobrescribir el anterior.
modelo_avanzado.save('modelo_fisuras_avanzado.keras')
print("\n🎉 ¡Entrenamiento finalizado!")
print("💾 Modelo avanzado guardado como 'modelo_fisuras_avanzado.keras'")

# === 8. VISUALIZAR RESULTADOS ===
# Copiamos el mismo código de visualización para analizar el rendimiento.
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']
loss = historial.history['loss']
val_loss = historial.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCAS), acc, label='Precisión de Entrenamiento')
plt.plot(range(EPOCAS), val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCAS), loss, label='Pérdida de Entrenamiento')
plt.plot(range(EPOCAS), val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()