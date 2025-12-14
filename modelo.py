import tensorflow as tf

# === CONFIGURACIÓN DE PARÁMETROS ===
IMG_HEIGHT = 256
IMG_WIDTH = 256

# === CREACIÓN DEL MODELO CONVOLUCIONAL (CNN) ===
# Usamos un modelo Secuencial, donde las capas se apilan una tras otra.
modelo = tf.keras.Sequential([
    
    # --- Capa de Entrada ---
    # Normaliza los valores de los píxeles (de 0-255 a 0-1) y define el tamaño de entrada.
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    
    # --- Bloque Convolucional 1 ---
    # Busca 16 patrones diferentes en la imagen.
    tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    # Reduce el tamaño de la imagen a la mitad para acelerar el proceso.
    tf.keras.layers.MaxPooling2D(),
    
    # --- Bloque Convolucional 2 ---
    # Busca 32 patrones más complejos.
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    # --- Bloque Convolucional 3 ---
    # Busca 64 patrones aún más complejos.
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    
    # --- Capa para Aplanar y Clasificar ---
    # Aplana todos los mapas de características para convertirlos en un solo vector.
    tf.keras.layers.Flatten(),
    
    # --- Capa Densa (Cerebro de la red) ---
    # Una capa densa con 128 neuronas para aprender a combinar los patrones encontrados.
    tf.keras.layers.Dense(128, activation='relu'),
    
    # --- Capa de Salida ---
    # La capa final tiene 1 neurona. Usamos 'sigmoid' porque es una clasificación binaria.
    # El resultado será un número entre 0 (sin fisura) y 1 (con fisura).
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# === COMPILAR EL MODELO ===
# Aquí le decimos al modelo cómo debe aprender.
modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy', # La función de pérdida ideal para clasificación binaria (sí/no).
    metrics=['accuracy']
)

# === MOSTRAR RESUMEN DEL MODELO ===
# Esto es útil para ver la arquitectura y el número de parámetros.
print("✅ Modelo creado y compilado exitosamente.")
modelo.summary()