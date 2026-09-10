from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "model/efficientnet_best.tflite"

# Load TFLite model once at startup — much lighter than full Keras model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_labels = [
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(img):
    img = img.resize((224, 224)).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    # EfficientNet preprocessing: scale to [-1, 1]
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = Image.open(file.stream)
    input_data = preprocess_image(img)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    return jsonify({
        'disease': predicted_class,
        'confidence': f"{confidence:.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)
