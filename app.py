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

# ============================================================
# MODEL SETUP
# ============================================================

MODEL_PATH = "model/efficientnet_best.tflite"

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
# DISPLAY NAMES
# ============================================================

display_names = {
    'Pepper,_bell___Bacterial_spot': 'Pepper Bell Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper Bell Healthy',

    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Potato Healthy',

    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite':
        'Tomato Spider Mites (Two-Spotted Spider Mite)',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus':
        'Tomato Mosaic Virus',
    'Tomato___healthy': 'Tomato Healthy'
}

# ============================================================
# SHORT LABELS FOR FIELD OVERLAY
# ============================================================

short_labels = {
    'Pepper,_bell___Bacterial_spot': 'Pepper Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper Healthy',

    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Potato Healthy',

    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite':
        'Tomato Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        'Tomato Yellow Leaf Curl',
    'Tomato___Tomato_mosaic_virus':
        'Tomato Mosaic Virus',
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
# RECOMMENDATIONS
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
        "Apply an appropriate fungicide according to the product label.",
        "Avoid overhead irrigation where possible.",
        "Ensure proper plant spacing for airflow."
    ],

    'Potato___Late_blight': [
        "Remove severely infected plants and dispose of them appropriately.",
        "Apply an appropriate fungicide according to local agricultural guidance.",
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
        "Apply an appropriate fungicide according to the product label.",
        "Add mulch around the base of plants to reduce soil splash.",
        "Water at the base of the plant rather than directly on the leaves."
    ],

    'Tomato___Late_blight': [
        "Remove severely infected plants and dispose of them appropriately.",
        "Apply an appropriate fungicide according to local agricultural guidance.",
        "Avoid overhead irrigation where possible.",
        "Do not compost heavily infected plant material."
    ],

    'Tomato___Leaf_Mold': [
        "Prune affected leaves and improve spacing for better air circulation.",
        "Reduce excessive humidity, especially in greenhouse conditions.",
        "Apply an appropriate fungicide if infection is severe.",
        "Avoid wetting the leaves during watering."
    ],

    'Tomato___Septoria_leaf_spot': [
        "Remove infected lower leaves.",
        "Apply an appropriate fungicide according to the product label.",
        "Avoid overhead watering.",
        "Practice crop rotation next season."
    ],

    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "Spray plants firmly with water to dislodge mites.",
        "Use an appropriate miticide or insecticidal soap when necessary.",
        "Encourage natural predators where practical.",
        "Monitor plants regularly, especially during dry weather."
    ],

    'Tomato___Target_Spot': [
        "Remove infected leaves.",
        "Improve air circulation around plants.",
        "Apply an appropriate fungicide according to the product label.",
        "Avoid overhead watering."
    ],

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Remove and dispose of severely infected plants.",
        "Control whitefly populations using appropriate integrated pest management methods.",
        "Use resistant tomato varieties where available.",
        "Remove weeds that may serve as alternative hosts."
    ],

    'Tomato___Tomato_mosaic_virus': [
        "Remove and dispose of infected plants.",
        "Disinfect tools between uses.",
        "Avoid handling healthy plants after touching infected plants.",
        "Control insect vectors and weeds according to local agricultural guidance."
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
    img = img.resize((224, 224)).convert('RGB')

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    return arr


# ============================================================
# MODEL PREDICTION
# ============================================================

def classify_image(img):
    input_data = preprocess_image(img)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_details[0]['index']
    )

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_labels[predicted_index]

    confidence = float(
        np.max(predictions)
    ) * 100

    return predicted_class, confidence


# ============================================================
# AUTOMATIC IMAGE TYPE ESTIMATION
# ============================================================

def estimate_image_type(img):
    """
    Estimates whether an uploaded image is more suitable for
    direct leaf classification or wider region screening.

    This is a simple computer-vision heuristic. It is NOT a
    separately trained image detector.

    Returns:
        "single" or "wide"
    """

    # Resize for inexpensive analysis
    small = img.copy()
    small.thumbnail((300, 300))

    arr = np.array(small.convert('RGB'))

    if arr.size == 0:
        return "single"

    # Calculate colour statistics
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    # Approximate vegetation pixels
    vegetation = (
        (g > r * 0.90) &
        (g > b * 0.90) &
        (g > 45)
    )

    vegetation_ratio = float(np.mean(vegetation))

    height, width = vegetation.shape

    # Divide image into 3x3 sections
    occupied_regions = 0

    for row in range(3):
        for col in range(3):

            y1 = row * height // 3
            y2 = (row + 1) * height // 3

            x1 = col * width // 3
            x2 = (col + 1) * width // 3

            region = vegetation[y1:y2, x1:x2]

            if region.size > 0:
                ratio = float(np.mean(region))

                if ratio > 0.05:
                    occupied_regions += 1

    # Wider image characteristics
    aspect_ratio = width / max(height, 1)

    # Decision rule
    if occupied_regions >= 4 and vegetation_ratio > 0.08:
        return "wide"

    if vegetation_ratio > 0.35 and occupied_regions >= 3:
        return "wide"

    if aspect_ratio >= 1.8 and occupied_regions >= 3:
        return "wide"

    return "single"


# ============================================================
# FIELD SCREENING
# ============================================================

def perform_field_scan(img, rows=3, cols=3):

    W, H = img.size

    tile_w = W // cols
    tile_h = H // rows

    if tile_w < 20 or tile_h < 20:
        raise ValueError(
            "Image is too small for region screening."
        )

    overlay = Image.new(
        'RGBA',
        img.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(overlay)

    try:
        font_size = max(
            14,
            min(28, tile_h // 9)
        )

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size
        )

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

            right = (
                W if c == cols - 1
                else left + tile_w
            )

            bottom = (
                H if r == rows - 1
                else top + tile_h
            )

            tile_img = img.crop(
                (left, top, right, bottom)
            )

            predicted_class, confidence = classify_image(
                tile_img
            )

            is_healthy = predicted_class.endswith(
                'healthy'
            )

            full_label = display_names.get(
                predicted_class,
                predicted_class
            )

            overlay_label = short_labels.get(
                predicted_class,
                predicted_class
            )

            if is_healthy:

                healthy_count += 1

                fill = (
                    34, 139, 34, 65
                )

                border = (
                    27, 94, 32, 255
                )

            else:

                diseased_count += 1

                disease_tally[full_label] = (
                    disease_tally.get(full_label, 0) + 1
                )

                fill = (
                    200, 30, 30, 85
                )

                border = (
                    183, 28, 28, 255
                )

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

            tag = (
                f"{overlay_label} "
                f"{confidence:.0f}%"
            )

            text_bbox = draw.textbbox(
                (0, 0),
                tag,
                font=font
            )

            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]

            tx = left + 5
            ty = top + 5

            draw.rectangle(
                [
                    tx - 3,
                    ty - 3,
                    tx + tw + 5,
                    ty + th + 6
                ],
                fill=(0, 0, 0, 160)
            )

            draw.text(
                (tx, ty),
                tag,
                fill=(255, 255, 255, 255),
                font=font
            )

            tiles_info.append({
                'row': r,
                'col': c,
                'disease': predicted_class,
                'display_name': full_label,
                'confidence': round(
                    confidence,
                    1
                ),
                'healthy': is_healthy
            })

    annotated = Image.alpha_composite(
        img.convert('RGBA'),
        overlay
    ).convert('RGB')

    buffer = io.BytesIO()

    annotated.save(
        buffer,
        format='JPEG',
        quality=88
    )

    image_b64 = base64.b64encode(
        buffer.getvalue()
    ).decode('utf-8')

    total = rows * cols

    healthy_pct = round(
        healthy_count / total * 100,
        1
    )

    diseased_pct = round(
        diseased_count / total * 100,
        1
    )

    if disease_tally:

        most_common = max(
            disease_tally,
            key=disease_tally.get
        )

    else:

        most_common = None

    if diseased_count == 0:

        field_status = (
            "No disease signals were detected "
            "in the screened regions."
        )

    elif diseased_pct < 20:

        field_status = (
            "A small number of regions were "
            "flagged for closer inspection."
        )

    elif diseased_pct < 50:

        field_status = (
            "Several regions were flagged and "
            "should be inspected more closely."
        )

    else:

        field_status = (
            "A large number of screened regions "
            "were flagged and require attention."
        )

    return {
        'image': image_b64,
        'rows': rows,
        'cols': cols,
        'total_tiles': total,
        'healthy_count': healthy_count,
        'diseased_count': diseased_count,
        'healthy_pct': healthy_pct,
        'diseased_pct': diseased_pct,
        'most_common_disease': most_common,
        'disease_tally': disease_tally,
        'field_status': field_status,
        'tiles': tiles_info
    }


# ============================================================
# HOME
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({
            'error': 'No image uploaded.'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'No image selected.'
        }), 400

    try:

        img = Image.open(
            file.stream
        ).convert('RGB')

        predicted_class, confidence = classify_image(
            img
        )

        full_name = display_names.get(
            predicted_class,
            predicted_class
        )

        steps = recommendations.get(
            predicted_class,
            ["No recommendation available."]
        )

        symptom_text = symptoms.get(
            predicted_class,
            "No symptom description available."
        )

        return jsonify({

            'mode': 'single',

            'disease': predicted_class,

            'display_name': full_name,

            'crop': full_name.split(
                ' '
            )[0],

            'confidence': f"{confidence:.2f}%",

            'symptoms': symptom_text,

            'recommendation_steps': steps
        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            'error': 'The image could not be analyzed.'
        }), 500


# ============================================================
# AUTOMATIC ANALYSIS
# ============================================================

@app.route('/analyze', methods=['POST'])
def analyze():

    if 'file' not in request.files:
        return jsonify({
            'error': 'No image uploaded.'
        }), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({
            'error': 'No image selected.'
        }), 400

    try:

        img = Image.open(
            file.stream
        ).convert('RGB')

        # Determine how to process image
        image_type = estimate_image_type(img)

        # ----------------------------------------------------
        # SINGLE LEAF
        # ----------------------------------------------------

        if image_type == 'single':

            predicted_class, confidence = classify_image(
                img
            )

            full_name = display_names.get(
                predicted_class,
                predicted_class
            )

            return jsonify({

                'mode': 'single',

                'display_name': full_name,

                'disease': predicted_class,

                'confidence': round(
                    confidence,
                    2
                ),

                'symptoms': symptoms.get(
                    predicted_class,
                    "No symptom description available."
                ),

                'recommendation_steps':
                    recommendations.get(
                        predicted_class,
                        ["No recommendation available."]
                    )
            })

        # ----------------------------------------------------
        # WIDE / MULTI-REGION IMAGE
        # ----------------------------------------------------

        result = perform_field_scan(
            img,
            rows=3,
            cols=3
        )

        result['mode'] = 'field'

        return jsonify(result)

    except Exception as e:

        print("Automatic analysis error:", e)

        return jsonify({
            'error':
                'The image could not be analyzed.'
        }), 500


# ============================================================
# CHAT
# ============================================================

@app.route('/chat', methods=['POST'])
def chat():

    data = request.get_json()

    if not data:
        return jsonify({
            'error': 'No question provided.'
        }), 400

    question = data.get(
        'question',
        ''
    ).strip()

    disease = data.get(
        'disease',
        'Unknown'
    )

    context = data.get(
        'context',
        ''
    )

    if not question:

        return jsonify({
            'error': 'Please enter a question.'
        }), 400

    prompt = f"""
You are an agricultural assistant inside a crop disease
detection, pesticide recommendation and management system.

The machine-learning system produced this diagnosis/context:

{disease}

Additional screening information:

{context}

Answer the farmer's question clearly and practically.

Important:
- Do not claim certainty beyond the provided diagnosis.
- Explain that machine-learning predictions should be confirmed
  by a qualified agricultural professional when necessary.
- Give practical crop-management guidance.
- Do not invent a disease that was not provided.
- Keep the language simple enough for an ordinary farmer.
- Use Markdown.
- Use short headings when useful.
- Use numbered or bullet lists for steps.

Farmer's question:

{question}
"""

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
                    'answer': response.text
                })

            except Exception as e:

                print(
                    f"Gemini error "
                    f"[{model_name}] "
                    f"(attempt {attempt + 1}): {e}"
                )

                if (
                    "503" in str(e)
                    or "UNAVAILABLE" in str(e)
                ):

                    time.sleep(2)

                    continue

                break

    return jsonify({
        'error':
            'The agricultural assistant is temporarily unavailable.'
    }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == '__main__':
    app.run(debug=True)
