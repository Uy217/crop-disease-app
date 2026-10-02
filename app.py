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

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)


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
# FARMER-FRIENDLY DISEASE NAMES
# ============================================================

display_names = {
    'Pepper,_bell___Bacterial_spot': 'Pepper Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper Healthy',

    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Potato Healthy',

    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Tomato Healthy'
}


# ============================================================
# SHORT LABELS FOR FIELD IMAGE
# ============================================================

short_labels = {
    'Pepper,_bell___Bacterial_spot': 'Pepper - Bacterial Spot',
    'Pepper,_bell___healthy': 'Pepper - Healthy',

    'Potato___Early_blight': 'Potato - Early Blight',
    'Potato___Late_blight': 'Potato - Late Blight',
    'Potato___healthy': 'Potato - Healthy',

    'Tomato___Bacterial_spot': 'Tomato - Bacterial Spot',
    'Tomato___Early_blight': 'Tomato - Early Blight',
    'Tomato___Late_blight': 'Tomato - Late Blight',
    'Tomato___Leaf_Mold': 'Tomato - Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato - Septoria Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato - Spider Mites',
    'Tomato___Target_Spot': 'Tomato - Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato - Yellow Leaf Curl',
    'Tomato___Tomato_mosaic_virus': 'Tomato - Mosaic Virus',
    'Tomato___healthy': 'Tomato - Healthy'
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
        "Dark brown spots with concentric rings on older, lower leaves first.",

    'Potato___Late_blight':
        "Large, irregular, water-soaked dark green to brown patches on leaves, often with white fungal growth on the underside.",

    'Potato___healthy':
        "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration.",

    'Tomato___Bacterial_spot':
        "Small, dark, greasy-looking spots on leaves and fruit, often with a yellow halo.",

    'Tomato___Early_blight':
        "Dark brown spots with concentric rings, usually starting on older lower leaves, which may yellow and drop.",

    'Tomato___Late_blight':
        "Large, irregular, water-soaked grey-green patches that can spread quickly, often with white mold on leaf undersides.",

    'Tomato___Leaf_Mold':
        "Pale green or yellow spots on the upper leaf surface, with olive-green to grey mold visible underneath.",

    'Tomato___Septoria_leaf_spot':
        "Small, circular spots with dark borders and grey centers, mainly on lower leaves.",

    'Tomato___Spider_mites Two-spotted_spider_mite':
        "Fine yellow speckling on leaves, sometimes with visible webbing on the underside in heavy infestations.",

    'Tomato___Target_Spot':
        "Brown spots with concentric rings similar to early blight, appearing on leaves, stems, and fruit.",

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus':
        "Upward curling and yellowing of leaves, stunted plant growth, and reduced fruit production.",

    'Tomato___Tomato_mosaic_virus':
        "Mottled light and dark green patches on leaves, with possible leaf curling and stunted growth.",

    'Tomato___healthy':
        "No visible symptoms. Leaves are uniformly green with no spots, wilting, or discoloration."
}


# ============================================================
# PESTICIDE / MANAGEMENT RECOMMENDATIONS
# ============================================================

recommendations = {
    'Pepper,_bell___Bacterial_spot': [
        "Remove and destroy infected leaves.",
        "Avoid overhead watering to reduce leaf wetness.",
        "Apply an appropriate copper-based bactericide according to its label.",
        "Use certified disease-free seeds and practice crop rotation."
    ],

    'Pepper,_bell___healthy': [
        "No disease detected — the crop appears healthy.",
        "Continue regular monitoring.",
        "Maintain proper spacing for airflow.",
        "Maintain balanced crop nutrition."
    ],

    'Potato___Early_blight': [
        "Remove infected leaves promptly.",
        "Use an appropriate fungicide according to the product label.",
        "Avoid overhead irrigation where practical.",
        "Ensure proper plant spacing and airflow."
    ],

    'Potato___Late_blight': [
        "Remove severely infected plant material promptly.",
        "Use an appropriate fungicide according to the product label.",
        "Avoid working among plants when foliage is wet.",
        "Monitor nearby plants closely."
    ],

    'Potato___healthy': [
        "No disease detected — the crop appears healthy.",
        "Maintain crop rotation practices.",
        "Continue regular monitoring."
    ],

    'Tomato___Bacterial_spot': [
        "Remove and dispose of infected plant debris.",
        "Use an appropriate copper-based bactericide according to its label.",
        "Avoid overhead watering where practical.",
        "Use disease-free seeds and resistant varieties where available."
    ],

    'Tomato___Early_blight': [
        "Prune and remove lower infected leaves.",
        "Use an appropriate fungicide according to the product label.",
        "Add mulch around the plant base to reduce soil splash.",
        "Water at the base of the plant rather than directly on leaves."
    ],

    'Tomato___Late_blight': [
        "Remove and destroy severely infected plant material promptly.",
        "Use an appropriate fungicide according to the product label.",
        "Avoid overhead irrigation where practical.",
        "Do not compost heavily infected plant material."
    ],

    'Tomato___Leaf_Mold': [
        "Improve spacing and air circulation.",
        "Reduce excessive humidity where possible.",
        "Use an appropriate fungicide if necessary.",
        "Avoid unnecessary wetting of leaves."
    ],

    'Tomato___Septoria_leaf_spot': [
        "Remove infected lower leaves.",
        "Use an appropriate fungicide according to its product label.",
        "Avoid overhead watering where practical.",
        "Practice crop rotation."
    ],

    'Tomato___Spider_mites Two-spotted_spider_mite': [
        "Use a firm water spray to dislodge mites.",
        "Use an appropriate miticide or insecticidal soap when necessary.",
        "Encourage natural predators where appropriate.",
        "Monitor plants regularly, especially during dry conditions."
    ],

    'Tomato___Target_Spot': [
        "Remove infected leaves.",
        "Improve air circulation.",
        "Use an appropriate fungicide according to the product label.",
        "Avoid unnecessary wetting of foliage."
    ],

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Remove severely infected plants where appropriate.",
        "Control whitefly populations using suitable integrated pest-management practices.",
        "Use resistant tomato varieties where available.",
        "Control weeds that may harbour insect vectors."
    ],

    'Tomato___Tomato_mosaic_virus': [
        "Remove and destroy infected plants.",
        "Disinfect tools between uses.",
        "Avoid handling healthy plants after touching infected plants.",
        "Control potential insect vectors and maintain field sanitation."
    ],

    'Tomato___healthy': [
        "No disease detected — the crop appears healthy.",
        "Continue regular monitoring.",
        "Maintain proper watering.",
        "Maintain balanced fertilization."
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
# CLASSIFICATION
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

    confidence = float(np.max(predictions)) * 100

    return predicted_class, confidence


# ============================================================
# AUTOMATIC IMAGE TYPE ANALYSIS
#
# This is NOT a new ML model.
# It is an image-processing routing mechanism.
#
# It estimates whether the uploaded picture is more suitable
# for direct single-leaf classification or region screening.
# ============================================================

def calculate_image_complexity(img):

    small = img.resize((160, 160)).convert('RGB')

    arr = np.asarray(small, dtype=np.float32) / 255.0

    # Colour variation
    colour_variation = float(np.mean(np.std(arr, axis=(0, 1))))

    # Convert to grayscale
    gray = (
        0.299 * arr[:, :, 0] +
        0.587 * arr[:, :, 1] +
        0.114 * arr[:, :, 2]
    )

    # Simple edge estimation without OpenCV
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))

    edge_strength = float(
        (np.mean(horizontal) + np.mean(vertical)) / 2
    )

    # Green vegetation proportion
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]

    green_pixels = (
        (green > red * 1.05) &
        (green > blue * 1.02) &
        (green > 0.20)
    )

    vegetation_ratio = float(np.mean(green_pixels))

    # Combine visual indicators
    complexity_score = (
        colour_variation * 0.45 +
        edge_strength * 1.8 +
        vegetation_ratio * 0.25
    )

    return complexity_score


def determine_processing_mode(img):

    width, height = img.size

    aspect_ratio = max(width, height) / max(1, min(width, height))

    complexity = calculate_image_complexity(img)

    # First perform a normal classification.
    direct_class, direct_confidence = classify_image(img)

    # --------------------------------------------------------
    # Routing logic
    # --------------------------------------------------------

    field_score = 0

    # Very wide images are more likely to contain multiple
    # plants/regions.
    if aspect_ratio >= 1.7:
        field_score += 2
    elif aspect_ratio >= 1.45:
        field_score += 1

    # Larger images are often wider photographs.
    if width >= 1800 or height >= 1800:
        field_score += 1

    # Visually complex images receive additional points.
    if complexity >= 0.16:
        field_score += 1

    if complexity >= 0.22:
        field_score += 1

    # A very confident direct leaf prediction is evidence that
    # the image may be a normal leaf photograph.
    if direct_confidence >= 85 and field_score <= 1:
        return "single", direct_class, direct_confidence

    # A combination of wide dimensions / complexity suggests
    # region screening.
    if field_score >= 2:
        return "field", direct_class, direct_confidence

    return "single", direct_class, direct_confidence


# ============================================================
# HOME
# ============================================================

@app.route('/')
def home():
    return render_template('index.html')


# ============================================================
# AUTOMATIC PREDICTION ROUTE
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    try:

        img = Image.open(file.stream).convert('RGB')

        mode, direct_class, direct_confidence = \
            determine_processing_mode(img)

        # ====================================================
        # SINGLE IMAGE
        # ====================================================

        if mode == "single":

            predicted_class = direct_class
            confidence = direct_confidence

            steps = recommendations.get(
                predicted_class,
                ["No recommendation available."]
            )

            symptom_text = symptoms.get(
                predicted_class,
                "No symptom description available."
            )

            return jsonify({

                'processing_mode': 'single',

                'mode_description':
                    'The uploaded image was processed as a single crop image.',

                'disease': predicted_class,

                'display_disease':
                    display_names.get(
                        predicted_class,
                        predicted_class
                    ),

                'confidence':
                    f"{confidence:.2f}%",

                'symptoms':
                    symptom_text,

                'recommendation_steps':
                    steps

            })


        # ====================================================
        # WIDER IMAGE
        # ====================================================

        return process_field_image(img)

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            'error':
                'The image could not be processed. Please try another crop image.'
        }), 500


# ============================================================
# FIELD PROCESSING
# ============================================================

def process_field_image(img):

    W, H = img.size

    # Automatic grid size.
    # We don't expose this to the farmer.

    if W * H >= 5000000:
        rows = 4
        cols = 4
    elif W * H >= 2000000:
        rows = 4
        cols = 4
    else:
        rows = 3
        cols = 3

    tile_w = W // cols
    tile_h = H // rows

    if tile_w < 40 or tile_h < 40:

        # Fall back to direct classification
        predicted_class, confidence = classify_image(img)

        return jsonify({

            'processing_mode': 'single',

            'mode_description':
                'The image was processed directly because it was too small for reliable region screening.',

            'disease': predicted_class,

            'display_disease':
                display_names.get(
                    predicted_class,
                    predicted_class
                ),

            'confidence':
                f"{confidence:.2f}%",

            'symptoms':
                symptoms.get(
                    predicted_class,
                    "No symptom description available."
                ),

            'recommendation_steps':
                recommendations.get(
                    predicted_class,
                    ["No recommendation available."]
                )

        })


    # ========================================================
    # ANNOTATION LAYER
    # ========================================================

    overlay = Image.new(
        'RGBA',
        img.size,
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(overlay)

    try:

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(14, tile_h // 9)
        )

    except Exception:

        font = ImageFont.load_default()


    tiles_info = []

    healthy_count = 0
    diseased_count = 0

    disease_tally = {}


    # ========================================================
    # PROCESS EACH REGION
    # ========================================================

    for r in range(rows):

        for c in range(cols):

            left = c * tile_w
            top = r * tile_h

            right = W if c == cols - 1 \
                else left + tile_w

            bottom = H if r == rows - 1 \
                else top + tile_h

            tile_img = img.crop(
                (left, top, right, bottom)
            )

            predicted_class, confidence = \
                classify_image(tile_img)

            is_healthy = predicted_class.endswith(
                'healthy'
            )

            label = short_labels.get(
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

                disease_tally[label] = \
                    disease_tally.get(label, 0) + 1

                fill = (
                    200, 30, 30, 90
                )

                border = (
                    183, 28, 28, 255
                )


            # Draw region
            draw.rectangle(
                [
                    left,
                    top,
                    right - 1,
                    bottom - 1
                ],
                fill=fill,
                outline=border,
                width=4
            )


            # Draw label
            tag = (
                f"{label} "
                f"{confidence:.0f}%"
            )

            bbox = draw.textbbox(
                (0, 0),
                tag,
                font=font
            )

            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            tx = left + 6
            ty = top + 6

            draw.rectangle(
                [
                    tx - 3,
                    ty - 3,
                    tx + tw + 5,
                    ty + th + 6
                ],
                fill=(0, 0, 0, 165)
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

                'disease':
                    predicted_class,

                'display_disease':
                    display_names.get(
                        predicted_class,
                        predicted_class
                    ),

                'label':
                    label,

                'confidence':
                    round(confidence, 1),

                'healthy':
                    is_healthy

            })


    # ========================================================
    # CREATE ANNOTATED IMAGE
    # ========================================================

    annotated = Image.alpha_composite(
        img.convert('RGBA'),
        overlay
    ).convert('RGB')


    buf = io.BytesIO()

    annotated.save(
        buf,
        format='JPEG',
        quality=88
    )

    img_b64 = base64.b64encode(
        buf.getvalue()
    ).decode('utf-8')


    # ========================================================
    # FIELD SUMMARY
    # ========================================================

    total = rows * cols

    healthy_pct = round(
        healthy_count / total * 100,
        1
    )

    diseased_pct = round(
        diseased_count / total * 100,
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


    if diseased_pct == 0:

        field_status = (
            "No disease signals were detected "
            "in the screened regions."
        )

    elif diseased_pct < 20:

        field_status = (
            "Most screened regions appear healthy. "
            "Some regions were flagged for closer inspection."
        )

    elif diseased_pct < 50:

        field_status = (
            "Several screened regions showed possible "
            "disease signals and should be inspected closely."
        )

    else:

        field_status = (
            "A large proportion of the screened regions "
            "showed possible disease signals and require attention."
        )


    # ========================================================
    # COLLECT MAIN DISEASES
    # ========================================================

    detected_conditions = []

    for disease_label, count in sorted(
        disease_tally.items(),
        key=lambda item: item[1],
        reverse=True
    ):

        detected_conditions.append({

            'disease':
                disease_label,

            'count':
                count

        })


    return jsonify({

        'processing_mode':
            'field',

        'mode_description':
            'The wider image was automatically divided into regions and screened using the existing crop disease classifier.',

        'image':
            img_b64,

        'rows':
            rows,

        'cols':
            cols,

        'total_tiles':
            total,

        'healthy_count':
            healthy_count,

        'diseased_count':
            diseased_count,

        'healthy_pct':
            healthy_pct,

        'diseased_pct':
            diseased_pct,

        'most_common_disease':
            most_common,

        'most_common_display':
            display_names.get(
                most_common,
                most_common
            ) if most_common else None,

        'field_status':
            field_status,

        'detected_conditions':
            detected_conditions,

        'tiles':
            tiles_info

    })


# ============================================================
# FARMER CHAT
# ============================================================

@app.route('/chat', methods=['POST'])
def chat():

    data = request.get_json()

    disease = data.get(
        'disease',
        'Unknown'
    )

    question = data.get(
        'question',
        ''
    )

    if not question:

        return jsonify({
            'error':
                'Please enter a question.'
        }), 400


    prompt = (

        f"You are an agricultural assistant helping a farmer. "

        f"The crop disease detected by the machine learning system is: "
        f"{disease}. "

        f"Answer the farmer clearly and practically. "

        f"Do not claim certainty beyond the diagnosis provided. "

        f"Do not invent information. "

        f"Use simple language suitable for a farmer. "

        f"Format the response using Markdown. "

        f"Use short headings where useful and bullet points for steps. "

        f"Farmer's question: {question}"

    )


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
                    f"Gemini error "
                    f"[{model_name}] "
                    f"attempt {attempt + 1}: {e}"
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
            'Could not get a response right now. Please try again.'

    }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':

    app.run(
        debug=True
    )
