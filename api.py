from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import random
from PIL import Image

# ===============================
# CONFIGURACIÓN FLASK
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# CARGAR MODELO
# ===============================
MODEL_DIR = "models"
MODEL_NAME = "modelo_fisuras_avanzado.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MODEL_URL = "https://drive.google.com/file/d/1vwxUA0RD_nWp3m-D83mOb4-67aj6FY6n/view?usp=drive_link"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Descargando modelo entrenado...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modelo descargado")

modelo = load_model(MODEL_PATH)
print("🚀 Modelo cargado correctamente")

# ===============================
# CLASES
# ===============================
nombres_clases = {
    0: "CD (con grieta)",
    1: "UD (sin grieta)"
}

def severidad_aleatoria():
    return random.choice([
        "Fisura leve",
        "Fisura estándar",
        "Fisura grave"
    ])

# ===============================
# PREPROCESAMIENTO
# ===============================
def preparar_imagen(img: Image.Image):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# ENDPOINT DE PREDICCIÓN
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["file"]

    try:
        img = Image.open(file).convert("RGB")
        img_preparada = preparar_imagen(img)

        prediccion = modelo.predict(img_preparada)
        confianza = float(prediccion[0][0])

        if confianza > 0.5:
            clase = 1
            confianza_final = confianza
        else:
            clase = 0
            confianza_final = 1 - confianza

        resultado = nombres_clases[clase]

        # Severidad SOLO si hay grieta
        severidad = None
        if clase == 0:
            severidad = severidad_aleatoria()

        return jsonify({
            "resultado": resultado,
            "severidad": severidad,
            "confianza": round(confianza_final * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# EJECUCIÓN
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
