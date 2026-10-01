from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ai_edge_litert.interpreter import Interpreter
import os
import io
import base64
import time
from google import genai

app = Flask(__name__)

MODEL_PATH = "model/efficientnet_best.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
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

# Short, label-friendly names for the field-scan overlay (full names are too long to fit on a tile)
short_labels = {
    'Pepper,_bell___Bacterial_spot': 'Bacterial Spot',
    'Pepper,_bell___healthy': 'Healthy',
    'Potato___Early_blight': 'Early Blight',
    'Potato___Late_blight': 'Late Blight',
    'Potato___healthy': 'Healthy',
    'Tomato___Bacterial_spot': 'Bacterial Spot',
    'Tomato___Early_blight': 'Early Blight',
    'Tomato___Late_blight': 'Late Blight',
    'Tomato___Leaf_Mold': 'Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Septoria Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider Mites',
    'Tomato___Target_Spot': 'Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow Leaf Curl',
    'Tomato___Tomato_mosaic_virus': 'Mosaic Virus',
    'Tomato___healthy': 'Healthy'
}

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

def classify_tile(tile_img):
    """Runs the existing EfficientNet model on a single tile and returns (class_name, confidence)."""
    input_data = preprocess_image(tile_img)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    return predicted_class, confidence

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
    predicted_class, confidence = classify_tile(img)
    steps = recommendations.get(predicted_class, ["No recommendation available."])
    symptom_text = symptoms.get(predicted_class, "No symptom description available.")

    return jsonify({
        'disease': predicted_class,
        'confidence': f"{confidence:.2f}%",
        'symptoms': symptom_text,
        'recommendation_steps': steps
    })

@app.route('/field_scan', methods=['POST'])
def field_scan():
    """
    Precision Agriculture Field Screening Module.

    Accepts one wider image containing multiple plants/leaves, divides it into a
    grid of tiles, and runs the EXISTING leaf-level EfficientNet classifier on each
    tile independently. This is a field-level screening mechanism built on top of
    the leaf-level classifier — it does not claim to be a model trained on aerial
    or drone imagery. It flags which regions of a wider field photo look healthy
    versus which look diseased (and which disease), using the same model and the
    same confidence logic as the single-leaf /predict route above.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        rows = int(request.form.get('rows', 4))
        cols = int(request.form.get('cols', 4))
        rows = max(2, min(rows, 8))
        cols = max(2, min(cols, 8))
    except (TypeError, ValueError):
        rows, cols = 4, 4

    img = Image.open(file.stream).convert('RGB')
    W, H = img.size
    tile_w = W // cols
    tile_h = H // rows

    if tile_w < 20 or tile_h < 20:
        return jsonify({'error': 'Image is too small to divide into that many tiles. Try a larger image or fewer tiles.'}), 400

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, tile_h // 9))
    except Exception:
        font = ImageFont.load_default()

    tiles_info = []
    healthy_count = 0
    diseased_count = 0
    disease_tally = {}

    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            top = r * tile_h
            right = W if c == cols - 1 else left + tile_w
            bottom = H if r == rows - 1 else top + tile_h

            tile_img = img.crop((left, top, right, bottom))
            predicted_class, confidence = classify_tile(tile_img)
            is_healthy = predicted_class.endswith('healthy')
            label = short_labels.get(predicted_class, predicted_class)

            if is_healthy:
                healthy_count += 1
                fill = (34, 139, 34, 70)
                border = (27, 94, 32, 255)
            else:
                diseased_count += 1
                disease_tally[label] = disease_tally.get(label, 0) + 1
                fill = (200, 30, 30, 90)
                border = (183, 28, 28, 255)

            draw.rectangle([left, top, right - 1, bottom - 1], fill=fill, outline=border, width=3)

            tag = f"{label} {confidence:.0f}%"
            text_bbox = draw.textbbox((0, 0), tag, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            tx = left + 4
            ty = top + 4
            draw.rectangle([tx - 2, ty - 2, tx + tw + 2, ty + th + 4], fill=(0, 0, 0, 150))
            draw.text((tx, ty), tag, fill=(255, 255, 255, 255), font=font)

            tiles_info.append({
                'row': r, 'col': c, 'disease': predicted_class,
                'label': label, 'confidence': round(confidence, 1), 'healthy': is_healthy
            })

    annotated = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    buf = io.BytesIO()
    annotated.save(buf, format='JPEG', quality=88)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    total = rows * cols
    diseased_pct = round((diseased_count / total) * 100, 1)
    healthy_pct = round((healthy_count / total) * 100, 1)
    most_common = max(disease_tally, key=disease_tally.get) if disease_tally else None

    if diseased_pct == 0:
        field_status = "Field looks healthy — no disease signals detected in the scanned regions."
    elif diseased_pct < 20:
        field_status = "Mostly healthy — a few regions show possible disease and are worth a closer look."
    elif diseased_pct < 50:
        field_status = "Moderate concern — a notable portion of the field shows possible disease signals."
    else:
        field_status = "Requires attention — a large portion of the scanned field shows possible disease signals."

    return jsonify({
        'image': img_b64,
        'rows': rows,
        'cols': cols,
        'total_tiles': total,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'healthy_pct': healthy_pct,
        'diseased_pct': diseased_pct,
        'most_common_disease': most_common,
        'field_status': field_status,
        'tiles': tiles_info
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    disease = data.get('disease', 'Unknown')
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    prompt = (
        f"You are an agricultural assistant helping a farmer whose crop leaf was diagnosed with: {disease}. "
        f"Answer their question clearly and practically. "
        f"Format your answer using Markdown: use short '##' subheadings to break the answer into "
        f"sections where it makes sense (e.g. What it is, Immediate steps, Prevention), use blank lines "
        f"between paragraphs, and use numbered or bulleted lists for any steps. Keep each paragraph short. "
        f"Farmer's question: {question}"
    )

    models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash"]

    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return jsonify({'answer': response.text})
            except Exception as e:
                print(f"Gemini chat error [{model_name}] (attempt {attempt + 1}): {e}")
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2)
                    continue
                else:
                    break

    return jsonify({'error': 'Could not get a response right now. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
