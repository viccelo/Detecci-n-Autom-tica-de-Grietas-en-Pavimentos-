from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import io

# Cargar tu modelo de Deep Learning
modelo = load_model('modelo_fisuras.keras')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Recibir la imagen
    img_file = request.files['file']
    img = image.load_img(io.BytesIO(img_file.read()), target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predecir la clase (con o sin grieta)
    prediccion = modelo.predict(img_array)
    resultado = "UD (sin grieta)" if prediccion[0][0] > 0.5 else "CD (con grieta)"
    confianza = max(prediccion[0][0], 1 - prediccion[0][0])

    return jsonify({'resultado': resultado, 'confianza': confianza * 100})

if __name__ == '__main__':
    app.run(debug=True)
