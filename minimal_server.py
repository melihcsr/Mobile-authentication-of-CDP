import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import random

app = Flask(__name__)
CORS(app)

# Process image (just a simulation for testing)
def process_image(image_data):
    try:
        # Convert to PIL Image for basic validation
        image = Image.open(io.BytesIO(image_data))
        
        # Simple validation - just check if it's a valid image
        width, height = image.size
        if width < 10 or height < 10:
            return {"isValid": False, "confidence": 0.95, "error": "Image too small"}
        
        # For testing: randomly return true/false with higher probability of true
        is_valid = random.random() > 0.3  # 70% chance of being valid
        confidence = random.uniform(0.7, 0.99)
        
        result = {
            "isValid": is_valid,
            "confidence": float(confidence),
            "predictions": [confidence if is_valid else 1-confidence, 
                           1-confidence if is_valid else confidence]
        }
            
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": str(e), "isValid": False}

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
    return jsonify({
        "status": "running",
        "model_loaded": True,
        "model_type": "simulation"
    })

if __name__ == '__main__':
    port = 5001  # Changed port to avoid conflicts
    print(f"Starting minimal verification server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True) 