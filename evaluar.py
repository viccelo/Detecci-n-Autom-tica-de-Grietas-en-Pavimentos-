import tensorflow as tf
import os
import numpy as np

# === 1. CONFIGURACIÓN ===
# Debe ser idéntica a la configuración de entrenamiento
BASE_DIR = "D"
TEST_DIR = os.path.join(BASE_DIR, "test")
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

# === 2. CARGAR EL MODELO ENTRENADO ===
# Cargamos el modelo que guardamos en el paso anterior
try:
    modelo_cargado = tf.keras.models.load_model('modelo_fisuras.keras')
    print("✅ Modelo 'modelo_fisuras.keras' cargado exitosamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    print("Asegúrate de haber ejecutado 'entrenamiento.py' para guardar el modelo primero.")
    exit()

# === 3. CARGAR LOS DATOS DE PRUEBA ===
# Cargamos los datos de prueba SIN mezclarlos (shuffle=False) para una evaluación ordenada
datos_pruebas = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False  # Importante: no mezclar para una evaluación consistente
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("✅ Datos de prueba cargados.")

# === 4. EVALUACIÓN AUTOMÁTICA (La forma más fácil) ===
# TensorFlow tiene una función para evaluar el modelo con todos los datos de prueba.
print("\n--- Evaluación Automática de TensorFlow ---")
loss, accuracy = modelo_cargado.evaluate(datos_pruebas)
print(f"Pérdida en los datos de prueba: {loss:.4f}")
print(f"Precisión (Accuracy) en los datos de prueba: {accuracy*100:.2f}%")


# === 5. CONTEO MANUAL DE PREDICCIONES (Lo que pediste) ===
# Hacemos predicciones para todo el dataset y contamos una por una.
print("\n--- Conteo Manual de Predicciones ---")
predicciones_correctas = 0
predicciones_incorrectas = 0
total_imagenes = 0

# Iteramos sobre cada lote (batch) de imágenes en el set de pruebas
for imagenes_lote, etiquetas_lote in datos_pruebas:
    # Hacemos la predicción sobre el lote de imágenes
    predicciones_lote = modelo_cargado.predict(imagenes_lote, verbose=0)
    # Convertimos las probabilidades (ej. 0.98) a clases (1 o 0)
    predicciones_clase = (predicciones_lote > 0.5).astype("int32")
    
    # Comparamos las predicciones con las etiquetas reales
    etiquetas_reales = etiquetas_lote.numpy().reshape(-1, 1)
    
    # Contamos cuántas coincidieron en este lote
    coincidencias = np.sum(predicciones_clase == etiquetas_reales)
    
    predicciones_correctas += coincidencias
    predicciones_incorrectas += len(etiquetas_reales) - coincidencias
    total_imagenes += len(etiquetas_reales)

# === 6. MOSTRAR LOS RESULTADOS FINALES ===
print("\n📊 Resultados Finales de la Evaluación:")
print(f"Total de imágenes de prueba: {total_imagenes}")
print(f"Predicciones Correctas:   ✅ {predicciones_correctas}")
print(f"Predicciones Incorrectas: ❌ {predicciones_incorrectas}")

# Verificamos que el cálculo manual coincida con el automático
precision_calculada = (predicciones_correctas / total_imagenes) * 100
print(f"Precisión Calculada Manualmente: {precision_calculada:.2f}%")