import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

# === 1. IMPORTAR EL MODELO ===
# Importamos la variable 'modelo' que creamos en nuestro archivo modelo.py
from modelo import modelo

# === 2. CONFIGURACIÓN Y CARGA DE DATOS ===
# Asegúrate de que estos parámetros sean los mismos que en tus otros archivos.
BASE_DIR = "D"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

# El tamaño de imagen debe coincidir con el input_shape del modelo (256, 256)
IMG_SIZE = (256, 256) 
BATCH_SIZE = 32

# Cargar los datasets de entrenamiento y pruebas
datos_entrenamiento = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

datos_pruebas = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Nombres de las clases (TensorFlow las ordena alfabéticamente: CD=0, UD=1)
nombres_clases = {0: 'CD (con grieta)', 1: 'UD (sin grieta)'}

print("✅ Datos y modelo listos para el entrenamiento.")

# === 3. ENTRENAMIENTO DEL MODELO ===
# Definimos cuántas "vueltas" completas dará a los datos de entrenamiento.
EPOCAS = 10 # Puedes empezar con 10 y luego ajustar si es necesario

historial = modelo.fit(
    datos_entrenamiento,
    epochs=EPOCAS,
    validation_data=datos_pruebas # Usamos los datos de prueba para validar en cada época
)

print("🎉 ¡Entrenamiento finalizado!")
modelo.save('modelo_fisuras.keras')
print("💾 Modelo guardado exitosamente como modelo_fisuras.keras")

# === 4. VISUALIZAR LA PÉRDIDA Y PRECISIÓN ===
# Graficar cómo cambiaron la precisión y la pérdida durante el entrenamiento.
acc = historial.history['accuracy']
val_acc = historial.history['val_accuracy']
loss = historial.history['loss']
val_loss = historial.history['val_loss']

plt.figure(figsize=(10, 5))
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


# === 5. VISUALIZAR PREDICCIONES ===
# Tomamos un lote de imágenes del set de pruebas para ver qué tan bien predice el modelo.
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    predicciones = modelo.predict(imagenes_prueba)
    # El resultado de 'sigmoid' es una probabilidad. Si > 0.5, es clase 1, si no, clase 0.
    predicciones_clase = (predicciones > 0.5).astype("int32")

# Graficar una cuadrícula con las imágenes, su predicción y la etiqueta real.
plt.figure(figsize=(15, 15))
for i in range(25): # Mostraremos las primeras 25 imágenes del lote
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    # Mostramos la imagen (convertimos de tensor a array de numpy)
    plt.imshow(imagenes_prueba[i].numpy().astype("uint8"))

    etiqueta_real = int(etiquetas_prueba[i][0])
    etiqueta_predicha = predicciones_clase[i][0]
    confianza = float(predicciones[i][0])

    # Ponemos el color azul si la predicción fue correcta, y rojo si fue incorrecta.
    if etiqueta_predicha == etiqueta_real:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(f"Pred: {nombres_clases[etiqueta_predicha]} ({confianza*100:.0f}%)\nReal: {nombres_clases[etiqueta_real]}", color=color)

plt.tight_layout()
plt.show()