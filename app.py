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

# Class labels (order matches ImageDataGenerator sorted class indices)
class_labels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Clean, human-readable labels for the UI
display_labels = {
    'Pepper__bell___Bacterial_spot': 'Pepper Bell (Bacterial Spot)',
    'Pepper__bell___healthy': 'Pepper Bell (Healthy)',
    'Potato___Early_blight': 'Potato (Early Blight)',
    'Potato___Late_blight': 'Potato (Late Blight)',
    'Potato___healthy': 'Potato (Healthy)',
    'Tomato_Bacterial_spot': 'Tomato (Bacterial Spot)',
    'Tomato_Early_blight': 'Tomato (Early Blight)',
    'Tomato_Late_blight': 'Tomato (Late Blight)',
    'Tomato_Leaf_Mold': 'Tomato (Leaf Mold)',
    'Tomato_Septoria_leaf_spot': 'Tomato (Septoria Leaf Spot)',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato (Two-Spotted Spider Mite)',
    'Tomato__Target_Spot': 'Tomato (Target Spot)',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato (Yellow Leaf Curl Virus)',
    'Tomato__Tomato_mosaic_virus': 'Tomato (Mosaic Virus)',
    'Tomato_healthy': 'Tomato (Healthy)'
}

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

    # Load and preprocess image (MobileNetV2 [-1, 1] normalization)
    img = Image.open(filepath).convert('RGB').resize((200, 200))
    img_array = (np.array(img, dtype=np.float32) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fast ONNX inference
    prediction = session.run([output_name], {input_name: img_array})[0][0]
    pred_idx = int(np.argmax(prediction))
    
    raw_class = class_labels[pred_idx] if pred_idx < len(class_labels) else f"Class {pred_idx}"
    predicted_class = display_labels.get(raw_class, raw_class)
    confidence = round(float(100 * prediction[pred_idx]), 2)

    return render_template('index.html',
                           prediction=predicted_class,
                           confidence=confidence,
                           image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True, port=3001)
