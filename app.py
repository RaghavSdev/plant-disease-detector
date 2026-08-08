import os
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

# Load ONNX model with minimal RAM footprint (<50MB)
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Class labels (order matters)
class_labels = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_YellowLeaf_Curl_Virus', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite'
]

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Load and preprocess image with PIL & numpy (ultra fast, low memory)
    img = Image.open(filepath).convert('RGB').resize((200, 200))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fast ONNX inference
    prediction = session.run([output_name], {input_name: img_array})[0]
    pred_idx = int(np.argmax(prediction[0])) if prediction.ndim > 1 else int(np.argmax(prediction))
    predicted_class = class_labels[pred_idx] if pred_idx < len(class_labels) else f"Class {pred_idx}"
    confidence = round(float(100 * np.max(prediction)), 2)

    return render_template('index.html',
                           prediction=predicted_class,
                           confidence=confidence,
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True, port=3001)
