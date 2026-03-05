import os
import uuid
from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from weather import get_weather

# ------------------- Flask Setup -------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------- Load CNN Model -------------------
cnn_model = tf.keras.models.load_model('model/pest_detector.h5')

with open('class_names.txt', 'r') as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

# ------------------- Prediction Function -------------------
def predict_pest(image_path):

    img = Image.open(image_path).convert('RGB')
    resized = img.resize((224, 224))

    arr = np.array(resized) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prediction = cnn_model.predict(arr, verbose=0)[0]

    index = np.argmax(prediction)
    pest = CLASS_NAMES[index]
    confidence = round(prediction[index] * 100, 2)

    # Confidence check
    if confidence < 60:
        pest = "Unknown Pest"

    # Draw prediction on image
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    label_text = f"{pest} {confidence}%"
    draw.text((20, 20), label_text, fill="red", font=font)

    annotated_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"annotated_{uuid.uuid4().hex}.jpg"
    )

    img.save(annotated_path)

    return pest, 1, confidence, annotated_path


# ------------------- Advice Database -------------------
ADVICE_DB = {

    "grasshopper": "Spray neem oil or garlic-chili solution. Encourage birds as natural predators.",

    "aphids": "Apply neem oil spray or insecticidal soap every 5 days.",

    "armyworm": "Use Bacillus thuringiensis (Bt) spray in early morning.",

    "whitefly": "Use yellow sticky traps and neem oil spray.",

    "thrips": "Use blue sticky traps and insecticidal soap.",

    "mites": "Spray strong water jet or apply miticide.",

    "caterpillar": "Apply Bt spray or remove manually.",

    "beetle": "Hand-pick beetles and apply neem oil.",

    "stinkbug": "Use kaolin clay spray and remove eggs manually.",

    "weevil": "Remove infected plant parts and apply neem oil.",

    "termite": "Consult pest control and apply soil treatment.",

    "slug": "Use beer traps or iron phosphate pellets.",

    "snail": "Use copper tape barriers around plants.",

    "ants": "Use boric acid bait or neem oil spray."
}


# ------------------- Advice Engine -------------------
def get_advice(pest, count, weather):

    base = ADVICE_DB.get(pest.lower(), "Monitor the crop and consult local agricultural expert.")

    extra = []

    if count > 5:
        extra.append("High infestation detected.")

    if weather["humidity"] > 80:
        extra.append("High humidity may increase pest spread.")

    if weather["temp"] > 30:
        extra.append("Spray pesticides in evening due to high temperature.")

    return f"{base} {' '.join(extra)}".strip()


# ------------------- Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        file = request.files['image']
        city = request.form['city'].strip()

        if not file or not city:
            return "Missing image or city", 400

        # Save uploaded image
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        # Predict pest
        pest, count, confidence, annotated_path = predict_pest(filepath)

        # Get weather
        weather = get_weather(city)

        # Generate advice
        advice = get_advice(pest, count, weather)

        annotated_filename = os.path.basename(annotated_path)

        return render_template(
            'result.html',
            pest=pest,
            count=count,
            confidence=confidence,
            weather=weather,
            advice=advice,
            filename=annotated_filename,
            city=city
        )

    return render_template('index.html')


# ------------------- Serve Uploaded Images -------------------
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)