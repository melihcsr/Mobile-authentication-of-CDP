import os
import sys
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Model parameters
MODEL_DIR = "checkpoints"
TARGET_SIZE = 224  # Typical target size for many models

# Function to find the latest model
def find_latest_model():
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory not found: {MODEL_DIR}")
        return None
    
    model_files = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith('.h5'):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print(f"No model files found in: {MODEL_DIR}")
        return None
    
    # Find the latest file by modification time
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_files[0]

# Process image using PIL instead of skimage
def process_image(image_data):
    try:
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize((TARGET_SIZE, TARGET_SIZE))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Process the prediction (assuming binary classification)
        if len(prediction[0]) == 2:
            result = {
                "isValid": bool(np.argmax(prediction[0]) == 0),
                "confidence": float(prediction[0][np.argmax(prediction[0])]),
                "predictions": prediction[0].tolist()
            }
        else:
            result = {
                "isValid": bool(np.argmax(prediction[0]) == 0),
                "class": int(np.argmax(prediction[0])),
                "confidence": float(prediction[0][np.argmax(prediction[0])]),
                "predictions": prediction[0].tolist()
            }
            
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e)}

# Routes
@app.route('/')
def index():
    return jsonify({"status": "API is running", "info": "Use /verify-qr endpoint to verify CDP images"})

@app.route('/verify-qr', methods=['POST'])
def verify_qr():
    # Check for qrImage in form data from SwiftQR app
    if 'qrImage' not in request.files:
        return jsonify({"error": "No image provided", "isValid": False}), 400
    
    image_file = request.files['qrImage']
    if image_file.filename == '':
        return jsonify({"error": "No image selected", "isValid": False}), 400
    
    # Read image bytes
    image_data = image_file.read()
    
    # Process image and get results
    result = process_image(image_data)
    
    return jsonify(result)

@app.route('/api/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    # Read image bytes
    image_data = image_file.read()
    
    # Process image and get results
    result = process_image(image_data)
    
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def status():
    model_path = find_latest_model()
    
    return jsonify({
        "status": "running",
        "model_loaded": model_path is not None,
        "model_path": model_path
    })

# Initialize model at startup
def initialize():
    global model
    model_path = find_latest_model()
    
    if model_path:
        print(f"Loading model: {model_path}")
        try:
            model = load_model(model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a dummy model for testing if real model can't be loaded
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            print("Created dummy model for testing")
            return False
    else:
        print("WARNING: No model found, creating dummy model")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        return False

if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=True) 