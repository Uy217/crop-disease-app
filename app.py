from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ai_edge_litert.interpreter import Interpreter
import os
import io
import base64
import time
import threading
from google import genai

app = Flask(__name__)

# ============================================================
# MODEL SETUP
# ============================================================

MODEL_PATH = "model/efficientnet_best.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prevent simultaneous access to the same TFLite interpreter
model_lock = threading.Lock()

# Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# ============================================================
# CLASS LABELS
# ============================================================

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


# ============================================================
# SHORT LABELS FOR FIELD SCREENING
# ============================================================

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


# ============================================================
# SYMPTOMS
# ============================================================

symptoms = {
    'Pepper,_bell___Bacterial_spot':
        "Small, dark, water-soaked spots on leaves that later turn brown with a yellow halo. Spots may also appear on fruit.",

    'Pepper,_bell___healthy':
        "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration.",

    'Potato___Early_blight':
        "Dark brown spots with concentric rings (target-like pattern) on older, lower leaves first.",

    'Potato___Late_blight':
        "Large, irregular, water-soaked dark green to brown patches on leaves, often with white fungal growth on the underside.",

    'Potato___healthy':
        "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration.",

    'Tomato___Bacterial_spot':
        "Small, dark, greasy-looking spots on leaves and fruit, often with a yellow halo.",

    'Tomato___Early_blight':
        "Dark brown spots with concentric rings, usually starting on older lower leaves, which may yellow and drop.",

    'Tomato___Late_blight':
        "Large, irregular, water-soaked grey-green patches, spreading quickly, often with white mold on leaf undersides.",

    'Tomato___Leaf_Mold':
        "Pale green or yellow spots on the upper leaf surface, with olive-green to grey mold visible underneath.",

    'Tomato___Septoria_leaf_spot':
        "Small, circular spots with dark borders and grey centers, mainly on lower leaves.",

    'Tomato___Spider_mites Two-spotted_spider_mite':
        "Fine yellow speckling on leaves, sometimes with visible webbing on the underside in heavy infestations.",

    'Tomato___Target_Spot':
        "Brown spots with concentric rings similar to early blight, can appear on leaves, stems, and fruit.",

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        "Upward curling and yellowing of leaves, stunted plant growth, and reduced fruit production.",

    'Tomato___Tomato_mosaic_virus':
        "Mottled light and dark green patches on leaves, with possible leaf curling and stunted growth.",

    'Tomato___healthy':
        "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration."
}


# ============================================================
# PESTICIDE + MANAGEMENT RECOMMENDATIONS
# ============================================================

recommendations = {

    'Pepper,_bell___Bacterial_spot': [
        "Remove and destroy infected leaves immediately.",
        "Avoid overhead watering to reduce leaf wetness.",
        "Apply a copper-based bactericide according to the product label.",
        "Rotate crops and use certified disease-free seeds next season."
    ],

    'Pepper,_bell___healthy': [
        "No disease detected — plant appears healthy.",
        "Continue regular monitoring for early signs of disease.",
        "Maintain proper spacing for airflow.",
        "Apply balanced fertilization as needed."
    ],

    'Potato___Early_blight': [
        "Remove infected leaves promptly.",
        "Apply an appropriate fungicide according to the product label and local agricultural guidance.",
        "Avoid overhead irrigation where possible.",
        "Ensure proper plant spacing for airflow."
    ],

    'Potato___Late_blight': [
        "Remove and destroy severely infected plant material promptly.",
        "Apply an appropriate late-blight fungicide according to the product label and local agricultural guidance.",
        "Avoid working in the field when leaves are wet.",
        "Monitor nearby plants closely for new symptoms."
    ],

    'Potato___healthy': [
        "No disease detected — plant appears healthy.",
        "Maintain crop rotation practices.",
        "Monitor regularly for early signs of disease."
    ],

    'Tomato___Bacterial_spot': [
        "Remove and dispose of infected plant debris.",
        "Apply a copper-based bactericide according to the product label.",
        "Avoid overhead watering and working with wet plants.",
        "Use disease-free seeds and resistant varieties where possible."
    ],

    'Tomato___Early_blight': [
        "Prune and remove lower infected leaves.",
        "Apply an appropriate fungicide according to the product label and local agricultural guidance.",
        "Add mulch around the base of plants to reduce soil splash onto leaves.",
        "Water at the base of the plant rather than directly on the leaves."
    ],

    'Tomato___Late_blight': [
        "Remove and destroy severely infected plant material promptly.",
        "Apply an appropriate late-blight fungicide according to the product label and local agricultural guidance.",
        "Avoid overhead irrigation where possible.",
        "Do not compost heavily infected plant material."
    ],

    'Tomato___Leaf_Mold': [
        "Prune affected leaves and improve spacing for better air circulation.",
        "Reduce excessive humidity where possible.",
        "Apply an appropriate fungicide according to the product label if infection is severe.",
        "Avoid unnecessary wetting of leaves."
    ],

    'Tomato___Septoria_leaf_spot': [
        "Remove infected lower leaves.",
        "Apply an appropriate fungicide according to the product label.",
        "Avoid overhead watering where possible.",
        "Practice crop rotation next season."
    ],

    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "Spray plants firmly with water to dislodge mites where appropriate.",
        "Use an appropriate miticide or insecticidal soap according to the product label if infestation is severe.",
        "Encourage natural predators where practical.",
        "Monitor plants regularly because infestations can spread quickly in dry conditions."
    ],

    'Tomato___Target_Spot': [
        "Remove infected leaves.",
        "Improve air circulation around plants.",
        "Apply an appropriate fungicide according to the product label.",
        "Avoid overhead watering where possible."
    ],

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Remove and destroy infected plants where appropriate.",
        "Control whitefly populations using approved control methods.",
        "Use resistant tomato varieties in future planting.",
        "Remove nearby weeds that may host whiteflies."
    ],

    'Tomato___Tomato_mosaic_virus': [
        "Remove and destroy infected plants.",
        "Disinfect tools between uses.",
        "Avoid handling healthy plants after touching infected plants.",
        "Control insect vectors where applicable and follow local agricultural guidance."
    ],

    'Tomato___healthy': [
        "No disease detected — plant appears healthy.",
        "Continue regular monitoring.",
        "Maintain proper watering and balanced fertilization."
    ]
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(img):
    """
    Converts an image into the 224 x 224 RGB format
    expected by the trained EfficientNet model.
    """

    img = img.resize((224, 224)).convert('RGB')

    arr = np.array(img, dtype=np.float32)

    arr = np.expand_dims(arr, axis=0)

    return arr


# ============================================================
# MODEL CLASSIFICATION
# ============================================================

def classify_image(img):
    """
    Runs the existing EfficientNet model on one image.
    Returns:
        predicted_class
        confidence
    """

    input_data = preprocess_image(img)

    with model_lock:

        interpreter.set_tensor(
            input_details[0]['index'],
            input_data
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(
            output_details[0]['index']
        )

    predicted_class = class_labels[np.argmax(predictions)]

    confidence = float(np.max(predictions)) * 100

    return predicted_class, confidence


# ============================================================
# HOME
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400

    try:

        img = Image.open(file.stream).convert('RGB')

        predicted_class, confidence = classify_image(img)

        steps = recommendations.get(
            predicted_class,
            ["No recommendation available."]
        )

        symptom_text = symptoms.get(
            predicted_class,
            "No symptom description available."
        )

        short_name = short_labels.get(
            predicted_class,
            predicted_class
        )

        is_healthy = predicted_class.endswith('healthy')

        return jsonify({

            'disease': predicted_class,

            'label': short_name,

            'confidence': f"{confidence:.2f}%",

            'symptoms': symptom_text,

            'recommendation_steps': steps,

            'healthy': is_healthy

        })

    except Exception as e:

        print(f"Prediction error: {e}")

        return jsonify({
            'error': 'Unable to process the image.'
        }), 500


# ============================================================
# FIELD SCREENING
# ============================================================

@app.route('/field_scan', methods=['POST'])
def field_scan():

    if 'file' not in request.files:
        return jsonify({
            'error': 'No field image uploaded'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'No field image selected'
        }), 400

    # --------------------------------------------------------
    # GRID SETTINGS
    # --------------------------------------------------------

    try:

        rows = int(request.form.get('rows', 4))
        cols = int(request.form.get('cols', 4))

        # Prevent excessive processing
        rows = max(2, min(rows, 6))
        cols = max(2, min(cols, 6))

    except (TypeError, ValueError):

        rows = 4
        cols = 4

    try:

        img = Image.open(file.stream).convert('RGB')

        W, H = img.size

        tile_w = W // cols
        tile_h = H // rows

        if tile_w < 20 or tile_h < 20:

            return jsonify({
                'error':
                'Image is too small for the selected grid. '
                'Try a larger image or fewer regions.'
            }), 400

        # ----------------------------------------------------
        # CREATE TRANSPARENT OVERLAY
        # ----------------------------------------------------

        overlay = Image.new(
            'RGBA',
            img.size,
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(overlay)

        # ----------------------------------------------------
        # FONT
        # ----------------------------------------------------

        try:

            font_path = (
                "/usr/share/fonts/truetype/dejavu/"
                "DejaVuSans-Bold.ttf"
            )

            font_size = max(
                14,
                min(28, tile_h // 9)
            )

            font = ImageFont.truetype(
                font_path,
                font_size
            )

        except Exception:

            font = ImageFont.load_default()

        # ----------------------------------------------------
        # COUNTERS
        # ----------------------------------------------------

        tiles_info = []

        healthy_count = 0

        diseased_count = 0

        disease_tally = {}

        confidence_values = []

        # ----------------------------------------------------
        # PROCESS EVERY REGION
        # ----------------------------------------------------

        for r in range(rows):

            for c in range(cols):

                left = c * tile_w

                top = r * tile_h

                right = (
                    W
                    if c == cols - 1
                    else left + tile_w
                )

                bottom = (
                    H
                    if r == rows - 1
                    else top + tile_h
                )

                tile_img = img.crop(
                    (left, top, right, bottom)
                )

                # --------------------------------------------
                # RUN EXISTING EFFICIENTNET MODEL
                # --------------------------------------------

                predicted_class, confidence = classify_image(
                    tile_img
                )

                confidence_values.append(confidence)

                is_healthy = predicted_class.endswith(
                    'healthy'
                )

                label = short_labels.get(
                    predicted_class,
                    predicted_class
                )

                # --------------------------------------------
                # COUNT RESULTS
                # --------------------------------------------

                if is_healthy:

                    healthy_count += 1

                    fill = (
                        34,
                        139,
                        34,
                        65
                    )

                    border = (
                        27,
                        94,
                        32,
                        255
                    )

                else:

                    diseased_count += 1

                    disease_tally[label] = (
                        disease_tally.get(label, 0) + 1
                    )

                    fill = (
                        200,
                        30,
                        30,
                        85
                    )

                    border = (
                        183,
                        28,
                        28,
                        255
                    )

                # --------------------------------------------
                # DRAW REGION
                # --------------------------------------------

                draw.rectangle(
                    [
                        left,
                        top,
                        right - 1,
                        bottom - 1
                    ],
                    fill=fill,
                    outline=border,
                    width=3
                )

                # --------------------------------------------
                # DRAW LABEL
                # --------------------------------------------

                tag = (
                    f"{label} "
                    f"{confidence:.0f}%"
                )

                text_bbox = draw.textbbox(
                    (0, 0),
                    tag,
                    font=font
                )

                tw = (
                    text_bbox[2]
                    - text_bbox[0]
                )

                th = (
                    text_bbox[3]
                    - text_bbox[1]
                )

                tx = left + 6

                ty = top + 6

                draw.rectangle(
                    [
                        tx - 3,
                        ty - 3,
                        tx + tw + 5,
                        ty + th + 6
                    ],
                    fill=(0, 0, 0, 155)
                )

                draw.text(
                    (tx, ty),
                    tag,
                    fill=(255, 255, 255, 255),
                    font=font
                )

                # --------------------------------------------
                # SAVE TILE INFORMATION
                # --------------------------------------------

                tiles_info.append({

                    'row': r,

                    'col': c,

                    'disease': predicted_class,

                    'label': label,

                    'confidence': round(
                        confidence,
                        1
                    ),

                    'healthy': is_healthy,

                    'symptoms': symptoms.get(
                        predicted_class,
                        "No symptom description available."
                    ),

                    'recommendations':
                        recommendations.get(
                            predicted_class,
                            []
                        )

                })

        # ----------------------------------------------------
        # CREATE ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated = Image.alpha_composite(
            img.convert('RGBA'),
            overlay
        ).convert('RGB')

        # ----------------------------------------------------
        # CONVERT IMAGE TO BASE64
        # ----------------------------------------------------

        buf = io.BytesIO()

        annotated.save(
            buf,
            format='JPEG',
            quality=88
        )

        img_b64 = base64.b64encode(
            buf.getvalue()
        ).decode('utf-8')

        # ----------------------------------------------------
        # FIELD STATISTICS
        # ----------------------------------------------------

        total = rows * cols

        healthy_pct = round(
            (healthy_count / total) * 100,
            1
        )

        diseased_pct = round(
            (diseased_count / total) * 100,
            1
        )

        average_confidence = round(
            sum(confidence_values)
            / len(confidence_values),
            1
        )

        most_common = (
            max(
                disease_tally,
                key=disease_tally.get
            )
            if disease_tally
            else None
        )

        # ----------------------------------------------------
        # FIELD STATUS
        # ----------------------------------------------------

        if diseased_count == 0:

            field_status = (
                "No disease signals were detected "
                "in the scanned regions."
            )

        elif diseased_pct < 20:

            field_status = (
                "A small number of scanned regions "
                "show possible disease signals and "
                "may require closer inspection."
            )

        elif diseased_pct < 50:

            field_status = (
                "Several scanned regions show possible "
                "disease signals and should be inspected."
            )

        else:

            field_status = (
                "A large proportion of the scanned regions "
                "show possible disease signals and "
                "should receive closer inspection."
            )

        # ----------------------------------------------------
        # UNIQUE DETECTED DISEASES
        # ----------------------------------------------------

        detected_conditions = []

        for disease_name, count in disease_tally.items():

            detected_conditions.append({

                'disease': disease_name,

                'regions': count,

                'percentage': round(
                    (count / total) * 100,
                    1
                )

            })

        # ----------------------------------------------------
        # FIELD-LEVEL RECOMMENDATIONS
        # ----------------------------------------------------

        field_recommendations = []

        for tile in tiles_info:

            if not tile['healthy']:

                for recommendation in tile[
                    'recommendations'
                ]:

                    if recommendation not in field_recommendations:

                        field_recommendations.append(
                            recommendation
                        )

        # ----------------------------------------------------
        # FIELD SYMPTOMS
        # ----------------------------------------------------

        field_symptoms = []

        for tile in tiles_info:

            if not tile['healthy']:

                symptom = tile['symptoms']

                if symptom not in field_symptoms:

                    field_symptoms.append(symptom)

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            'image': img_b64,

            'rows': rows,

            'cols': cols,

            'total_tiles': total,

            'healthy_count': healthy_count,

            'diseased_count': diseased_count,

            'healthy_pct': healthy_pct,

            'diseased_pct': diseased_pct,

            'average_confidence':
                average_confidence,

            'most_common_disease':
                most_common,

            'detected_conditions':
                detected_conditions,

            'field_symptoms':
                field_symptoms,

            'field_recommendations':
                field_recommendations,

            'field_status':
                field_status,

            'tiles':
                tiles_info

        })

    except Exception as e:

        print(f"Field scan error: {e}")

        return jsonify({

            'error':
            'Unable to process the field image.'

        }), 500


# ============================================================
# FARMER CHAT
# ============================================================

@app.route('/chat', methods=['POST'])
def chat():

    data = request.get_json() or {}

    disease = data.get(
        'disease',
        'Unknown'
    )

    question = data.get(
        'question',
        ''
    )

    # Optional field information
    field_context = data.get(
        'field_context',
        ''
    )

    if not question:

        return jsonify({
            'error':
            'No question provided'
        }), 400

    # --------------------------------------------------------
    # NORMAL SINGLE-DIAGNOSIS CHAT
    # --------------------------------------------------------

    if not field_context:

        prompt = (

            "You are an agricultural assistant helping "
            "a farmer.\n\n"

            f"The crop leaf was classified by the "
            f"machine learning system as: {disease}.\n\n"

            "Do not claim that you personally diagnosed "
            "the image. Treat the machine learning result "
            "as a prediction that may require confirmation "
            "by an agricultural professional.\n\n"

            "Answer the farmer's question clearly and "
            "practically.\n\n"

            "Format your answer using Markdown. "
            "Use short headings where useful, blank lines "
            "between paragraphs, and numbered or bulleted "
            "lists for steps.\n\n"

            f"Farmer's question: {question}"
        )

    # --------------------------------------------------------
    # FIELD-SCREENING CHAT
    # --------------------------------------------------------

    else:

        prompt = (

            "You are an agricultural assistant helping "
            "a farmer interpret results from a crop field "
            "screening system.\n\n"

            "The field screening system divides a wider "
            "image into regions and applies an existing "
            "leaf-level machine learning classifier to "
            "each region.\n\n"

            "The results are screening predictions and "
            "should not be presented as a confirmed "
            "professional field diagnosis.\n\n"

            f"Field screening information:\n"
            f"{field_context}\n\n"

            "Answer the farmer's question based on the "
            "provided screening results.\n\n"

            "Do not invent diseases that are not present "
            "in the provided results.\n\n"

            "Format your response using Markdown with "
            "short headings, paragraphs and lists where "
            "appropriate.\n\n"

            f"Farmer's question: {question}"
        )

    # --------------------------------------------------------
    # GEMINI MODEL FALLBACK
    # --------------------------------------------------------

    models_to_try = [
        "gemini-3.6-flash",
        "gemini-2.5-flash"
    ]

    for model_name in models_to_try:

        for attempt in range(2):

            try:

                response = client.models.generate_content(

                    model=model_name,

                    contents=prompt
                )

                return jsonify({

                    'answer':
                    response.text

                })

            except Exception as e:

                print(
                    f"Gemini chat error "
                    f"[{model_name}] "
                    f"(attempt {attempt + 1}): {e}"
                )

                if (
                    "503" in str(e)
                    or
                    "UNAVAILABLE" in str(e)
                ):

                    time.sleep(2)

                    continue

                else:

                    break

    return jsonify({

        'error':
        'Could not get a response right now. '
        'Please try again.'

    }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )
