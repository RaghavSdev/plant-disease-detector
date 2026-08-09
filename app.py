import os
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify, send_from_directory
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

# Detailed plant disease diagnostics and actionable care steps
disease_details = {
    'Pepper__bell___Bacterial_spot': {
        'status': 'danger',
        'severity': 'High Risk',
        'symptoms': 'Small dark water-soaked spots on leaves that turn brown/black with yellow halos.',
        'treatment': [
            'Apply copper-based bactericides early in the morning.',
            'Remove and safely destroy heavily infected leaves.',
            'Avoid overhead irrigation to keep foliage completely dry.'
        ]
    },
    'Pepper__bell___healthy': {
        'status': 'healthy',
        'severity': 'Optimal Health',
        'symptoms': 'Leaf shows rich green color, smooth foliage, and no signs of bacterial or fungal spots.',
        'treatment': [
            'Maintain consistent watering around the plant base.',
            'Ensure adequate sunlight (6-8 hours daily).',
            'Continue balanced organic fertilizer routine.'
        ]
    },
    'Potato___Early_blight': {
        'status': 'warning',
        'severity': 'Moderate Risk',
        'symptoms': 'Concentric target-like brown spots surrounded by yellow tissue on older lower leaves.',
        'treatment': [
            'Prune lower leaves touching soil to restrict fungal splash.',
            'Apply protective copper or chlorothalonil fungicide.',
            'Rotate crops annually with non-solanaceous plants.'
        ]
    },
    'Potato___Late_blight': {
        'status': 'danger',
        'severity': 'Critical Risk',
        'symptoms': 'Rapidly spreading dark water-soaked lesions with white moldy growth on leaf undersides.',
        'treatment': [
            'Immediately destroy severely affected vines to protect field.',
            'Apply systemic fungicides like metalaxyl or mancozeb.',
            'Ensure good soil drainage and hilling around tubers.'
        ]
    },
    'Potato___healthy': {
        'status': 'healthy',
        'severity': 'Optimal Health',
        'symptoms': 'Vibrant green leaves with crisp edges and healthy cellular structure.',
        'treatment': [
            'Keep soil consistently moist but never waterlogged.',
            'Inspect weekly for early signs of beetle or blight activity.',
            'Apply light organic mulch around base.'
        ]
    },
    'Tomato_Bacterial_spot': {
        'status': 'danger',
        'severity': 'High Risk',
        'symptoms': 'Dark, greasy-looking leaf spots causing foliage yellowing and premature leaf drop.',
        'treatment': [
            'Spray copper hydroxide combined with mancozeb.',
            'Disinfect pruning tools between plants using 10% bleach solution.',
            'Avoid working in foliage while plants are wet.'
        ]
    },
    'Tomato_Early_blight': {
        'status': 'warning',
        'severity': 'Moderate Risk',
        'symptoms': 'Dark brown spots with concentric ring pattern (bullseye target) on lower mature leaves.',
        'treatment': [
            'Prune lower 12 inches of foliage to improve airflow.',
            'Apply bio-fungicides containing Bacillus subtilis.',
            'Mulch heavily around tomato bases to prevent soil splashback.'
        ]
    },
    'Tomato_Late_blight': {
        'status': 'danger',
        'severity': 'Critical Risk',
        'symptoms': 'Large dark olive-brown patches on leaves with fuzzy white mildew underneath.',
        'treatment': [
            'Bag and dispose of infected plants immediately.',
            'Apply preventative copper spray on surrounding healthy plants.',
            'Ensure bright direct sunlight and low foliage humidity.'
        ]
    },
    'Tomato_Leaf_Mold': {
        'status': 'warning',
        'severity': 'Moderate Risk',
        'symptoms': 'Pale green yellow spots on upper leaf surfaces with velvety olive-brown mold on undersides.',
        'treatment': [
            'Improve greenhouse/garden air circulation with proper spacing.',
            'Keep ambient humidity below 85%.',
            'Spray sulphur or copper-based fungicides.'
        ]
    },
    'Tomato_Septoria_leaf_spot': {
        'status': 'warning',
        'severity': 'Moderate Risk',
        'symptoms': 'Numerous small circular spots with dark brown margins and light grey centers containing tiny black specks.',
        'treatment': [
            'Remove infected lower leaves at first sight.',
            'Spray copper or chlorothalonil fungicide weekly during rainy periods.',
            'Avoid overhead sprinkler watering.'
        ]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'status': 'warning',
        'severity': 'Moderate Risk',
        'symptoms': 'Yellow stippling/bronzing on upper leaf surfaces accompanied by fine silken webbing underneath.',
        'treatment': [
            'Hose down leaf undersides with strong water spray.',
            'Apply insecticidal soap or horticultural neem oil.',
            'Introduce natural predators like ladybugs or predatory mites.'
        ]
    },
    'Tomato__Target_Spot': {
        'status': 'warning',
        'severity': 'Moderate Risk',
        'symptoms': 'Small brown spots with light brown centers and dark rings expanding into target shapes.',
        'treatment': [
            'Maintain strict crop rotation and weed control.',
            'Apply protective fungicides early in disease cycle.',
            'Ensure adequate row spacing for quick canopy drying.'
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'status': 'danger',
        'severity': 'High Risk',
        'symptoms': 'Severe leaf curling upwards, stunted plant growth, yellowing margins, and reduced fruit set.',
        'treatment': [
            'Control whitefly vectors using yellow sticky traps or insect nets.',
            'Remove and burn infected virus-reservoir plants.',
            'Plant resistant tomato cultivars in future plantings.'
        ]
    },
    'Tomato__Tomato_mosaic_virus': {
        'status': 'danger',
        'severity': 'High Risk',
        'symptoms': 'Mottled dark green and yellow mosaic patterns on leaves with blistering or leaf distortion.',
        'treatment': [
            'Wash hands with soap after handling tobacco products before gardening.',
            'Discard infected plants (viruses cannot be cured with spray).',
            'Sterilize garden stakes and equipment thoroughly.'
        ]
    },
    'Tomato_healthy': {
        'status': 'healthy',
        'severity': 'Optimal Health',
        'symptoms': 'Deep green color, crisp foliage structure, and vigorous growth.',
        'treatment': [
            'Water deeply at the base early in the morning.',
            'Provide balanced N-P-K fertilizer every 2-3 weeks.',
            'Prune suckers to encourage fruit production.'
        ]
    }
}

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_and_predict(filepath):
    """Load, preprocess image and run ONNX model inference."""
    img = Image.open(filepath).convert('RGB').resize((200, 200))
    img_array = (np.array(img, dtype=np.float32) / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)

    # Fast ONNX inference
    prediction = session.run([output_name], {input_name: img_array})[0][0]
    pred_idx = int(np.argmax(prediction))
    
    raw_class = class_labels[pred_idx] if pred_idx < len(class_labels) else f"Class {pred_idx}"
    predicted_class = display_labels.get(raw_class, raw_class)
    confidence = round(float(100 * prediction[pred_idx]), 2)
    
    details = disease_details.get(raw_class, {
        'status': 'healthy' if 'healthy' in raw_class.lower() else 'warning',
        'severity': 'Analyzed',
        'symptoms': 'Pattern analyzed by neural network.',
        'treatment': ['Inspect plant regularly', 'Ensure adequate water and sunlight', 'Monitor foliage changes']
    })
    
    return {
        'raw_class': raw_class,
        'prediction': predicted_class,
        'confidence': confidence,
        'details': details
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'error': 'No file uploaded'}), 400
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'error': 'No file selected'}), 400
        return "No file selected", 400
        
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = process_and_predict(filepath)
    # Ensure forward slashes for HTML img src
    web_image_path = filepath.replace('\\', '/')

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'image_path': '/' + web_image_path if not web_image_path.startswith('/') else web_image_path,
            'details': result['details']
        })

    return render_template('index.html',
                           prediction=result['prediction'],
                           confidence=result['confidence'],
                           image_path='/' + web_image_path if not web_image_path.startswith('/') else web_image_path,
                           details=result['details'])

@app.route('/predict_sample/<sample_name>', methods=['POST', 'GET'])
def predict_sample(sample_name):
    allowed_samples = {
        'healthy': 'static/samples/sample_healthy.jpg',
        'pepper_spot': 'static/samples/sample_pepper_spot.jpg',
        'potato_blight': 'static/samples/sample_potato_blight.jpg'
    }
    
    if sample_name not in allowed_samples:
        return jsonify({'error': 'Invalid sample request'}), 400
        
    sample_path = allowed_samples[sample_name]
    if not os.path.exists(sample_path):
        return jsonify({'error': 'Sample file missing'}), 404
        
    result = process_and_predict(sample_path)
    web_image_path = '/' + sample_path.replace('\\', '/')
    
    return jsonify({
        'success': True,
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'image_path': web_image_path,
        'details': result['details']
    })

if __name__ == '__main__':
    app.run(debug=True, port=3001)

