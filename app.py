from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import google.genai as genai

app = Flask(__name__)

MODEL_PATH = "model/efficientnet_best.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class_labels = [
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

symptoms = {
    'Pepper,_bell___Bacterial_spot': "Small, dark, water-soaked spots on leaves that later turn brown with a yellow halo. Spots may also appear on fruit.",
    'Pepper,_bell___healthy': "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration.",
    'Potato___Early_blight': "Dark brown spots with concentric rings (target-like pattern) on older, lower leaves first.",
    'Potato___Late_blight': "Large, irregular, water-soaked dark green to brown patches on leaves, often with white fungal growth on the underside.",
    'Potato___healthy': "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration.",
    'Tomato___Bacterial_spot': "Small, dark, greasy-looking spots on leaves and fruit, often with a yellow halo.",
    'Tomato___Early_blight': "Dark brown spots with concentric rings, usually starting on older lower leaves, which may yellow and drop.",
    'Tomato___Late_blight': "Large, irregular, water-soaked grey-green patches, spreading quickly, often with white mold on leaf undersides.",
    'Tomato___Leaf_Mold': "Pale green or yellow spots on the upper leaf surface, with olive-green to grey mold visible underneath.",
    'Tomato___Septoria_leaf_spot': "Small, circular spots with dark borders and grey centers, mainly on lower leaves.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Fine yellow speckling on leaves, sometimes with visible webbing on the underside in heavy infestations.",
    'Tomato___Target_Spot': "Brown spots with concentric rings similar to early blight, can appear on leaves, stems, and fruit.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Upward curling and yellowing of leaves, stunted plant growth, and reduced fruit production.",
    'Tomato___Tomato_mosaic_virus': "Mottled light and dark green patches on leaves, with possible leaf curling and stunted growth.",
    'Tomato___healthy': "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration."
}

recommendations = {
    'Pepper,_bell___Bacterial_spot': ["Remove and destroy infected leaves immediately.", "Avoid overhead watering to reduce leaf wetness.", "Apply a copper-based bactericide.", "Rotate crops and use certified disease-free seeds next season."],
    'Pepper,_bell___healthy': ["No disease detected — plant appears healthy.", "Continue regular monitoring for early signs of disease.", "Maintain proper spacing for airflow.", "Apply balanced fertilization as needed."],
    'Potato___Early_blight': ["Remove infected leaves promptly.", "Apply a fungicide containing chlorothalonil or mancozeb.", "Avoid overhead irrigation.", "Ensure proper plant spacing for airflow."],
    'Potato___Late_blight': ["Destroy infected plants immediately — this disease spreads fast.", "Apply a fungicide containing metalaxyl or copper compounds.", "Avoid working in the field when leaves are wet.", "Monitor nearby healthy plants closely for new symptoms."],
    'Potato___healthy': ["No disease detected — plant appears healthy.", "Maintain crop rotation practices.", "Monitor regularly for early signs of disease."],
    'Tomato___Bacterial_spot': ["Remove and dispose of infected plant debris.", "Apply a copper-based bactericide.", "Avoid overhead watering and working with wet plants.", "Use disease-free seeds and resistant varieties where possible."],
    'Tomato___Early_blight': ["Prune and remove lower infected leaves.", "Apply a fungicide containing chlorothalonil or mancozeb.", "Add mulch around the base of plants to prevent soil splash onto leaves.", "Water at the base of the plant, not on the leaves."],
    'Tomato___Late_blight': ["Remove and destroy infected plants immediately — this disease spreads fast.", "Apply a fungicide containing metalaxyl or copper compounds.", "Avoid overhead irrigation.", "Do not compost infected plant material."],
    'Tomato___Leaf_Mold': ["Prune plants and improve spacing for better air circulation.", "Reduce humidity, especially in greenhouse conditions.", "Apply a fungicide if the infection is severe.", "Avoid wetting the leaves when watering."],
    'Tomato___Septoria_leaf_spot': ["Remove infected lower leaves.", "Apply a fungicide containing chlorothalonil.", "Avoid overhead watering.", "Practice crop rotation next season."],
    'Tomato___Spider_mites Two-spotted_spider_mite': ["Spray plants firmly with water to dislodge mites.", "Apply a miticide or insecticidal soap if infestation is severe.", "Encourage natural predators such as ladybugs.", "Monitor plants weekly, mites spread quickly in dry weather."],
    'Tomato___Target_Spot': ["Remove infected leaves.", "Improve air circulation around plants.", "Apply a fungicide containing chlorothalonil or copper.", "Avoid overhead watering."],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ["Remove and destroy infected plants — there is no cure once infected.", "Control whitefly populations, the primary carrier, using insecticides or sticky traps.", "Use resistant tomato varieties in future planting.", "Remove nearby weeds that may host whiteflies."],
    'Tomato___Tomato_mosaic_virus': ["Remove and destroy infected plants immediately.", "Disinfect all tools between uses.", "Avoid handling healthy plants after touching infected ones.", "Control aphid populations, which can spread the virus."],
    'Tomato___healthy': ["No disease detected — plant appears healthy.", "Continue regular monitoring.", "Maintain proper watering and balanced fertilization."]
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
    steps = recommendations.get(predicted_class, ["No recommendation available."])
    symptom_text = symptoms.get(predicted_class, "No symptom description available.")

    return jsonify({
        'disease': predicted_class,
        'confidence': f"{confidence:.2f}%",
        'symptoms': symptom_text,
        'recommendation_steps': steps
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    disease = data.get('disease', 'Unknown')
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    try:
        prompt = (
            f"You are an agricultural assistant helping a farmer whose crop leaf was diagnosed with: {disease}. "
            f"Answer their question clearly and practically in plain text, no markdown. "
            f"Farmer's question: {question}"
        )
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return jsonify({'answer': response.text})
    except Exception as e:
        return jsonify({'error': 'Could not get a response right now. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
