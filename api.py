from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import random
import urllib.request
from PIL import Image
import tensorflow as tf

# ===============================
# CONFIGURACIÓN FLASK
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# CONFIGURACIÓN MODELO TFLITE
# ===============================
MODEL_DIR = "models"
MODEL_NAME = "modelo_fisuras_avanzado.tflite"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

MODEL_URL = "https://huggingface.co/viccelo/Modelo_fisuras_avanzado/resolve/main/modelo_fisuras_avanzado.tflite"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("⬇️ Descargando modelo TFLite...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Modelo TFLite descargado")

# ===============================
# CARGA DEL MODELO TFLITE
# ===============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("🚀 Modelo TFLite cargado correctamente")

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
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ===============================
# ENDPOINT DE PREDICCIÓN
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    try:
        img = Image.open(request.files["file"]).convert("RGB")
        img_preparada = preparar_imagen(img)

        interpreter.set_tensor(input_details[0]["index"], img_preparada)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        confianza = float(output[0][0])

        if confianza > 0.5:
            clase = 1
            confianza_final = confianza
        else:
            clase = 0
            confianza_final = 1 - confianza

        resultado = nombres_clases[clase]

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
    app.run(host="0.0.0.0", port=5000)
