from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

MODEL_PATH = "model/efficientnet_best.tflite"

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

recommendations = {
    'Pepper,_bell___Bacterial_spot': "Remove and destroy infected leaves. Avoid overhead watering to reduce leaf wetness. Apply copper-based bactericides. Rotate crops and use certified disease-free seeds next season.",
    'Pepper,_bell___healthy': "No disease detected. Continue regular monitoring, proper spacing for airflow, and balanced fertilization.",
    'Potato___Early_blight': "Remove infected leaves promptly. Apply fungicides containing chlorothalonil or mancozeb. Avoid overhead irrigation and ensure proper plant spacing for airflow.",
    'Potato___Late_blight': "Destroy infected plants immediately to prevent spread. Apply fungicides such as metalaxyl or copper-based products. Avoid working in fields when leaves are wet.",
    'Potato___healthy': "No disease detected. Maintain crop rotation and monitor regularly for early signs of disease.",
    'Tomato___Bacterial_spot': "Remove infected plant debris. Apply copper-based bactericides. Avoid overhead watering and working with wet plants. Use disease-free seeds and resistant varieties where possible.",
    'Tomato___Early_blight': "Prune lower infected leaves. Apply fungicides such as chlorothalonil or mancozeb. Mulch around plants to prevent soil splash onto leaves.",
    'Tomato___Late_blight': "Remove and destroy infected plants immediately, this disease spreads fast. Apply fungicides containing metalaxyl or copper compounds. Avoid overhead irrigation.",
    'Tomato___Leaf_Mold': "Improve air circulation by pruning and spacing plants properly. Reduce humidity in greenhouse settings. Apply fungicides if the infection is severe.",
    'Tomato___Septoria_leaf_spot': "Remove infected lower leaves. Apply fungicides such as chlorothalonil. Avoid overhead watering and practice crop rotation.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Spray plants with water to dislodge mites. Apply miticides or insecticidal soap if infestation is severe. Encourage natural predators like ladybugs.",
    'Tomato___Target_Spot': "Remove infected leaves and improve air circulation. Apply fungicides containing chlorothalonil or copper. Avoid overhead watering.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Remove and destroy infected plants. Control whitefly populations, the primary carrier, using insecticides or sticky traps. Use resistant varieties in future planting.",
    'Tomato___Tomato_mosaic_virus': "Remove and destroy infected plants immediately. Disinfect tools between uses. Avoid handling healthy plants after touching infected ones. Control aphid populations.",
    'Tomato___healthy': "No disease detected. Continue regular monitoring, proper watering, and balanced fertilization."
}

def preprocess_image(img):
    img = img.resize((224, 224)).convert('RGB')
    arr = np.array(img, dtype=np.float32)
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
    advice = recommendations.get(predicted_class, "No recommendation available.")

    return jsonify({
        'disease': predicted_class,
        'confidence': f"{confidence:.2f}%",
        'recommendation': advice
    })

if __name__ == '__main__':
    app.run(debug=True)
