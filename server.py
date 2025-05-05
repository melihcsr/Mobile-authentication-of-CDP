import os
import sys
import numpy as np
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import skimage.io
import yaml
sys.path.insert(0, '.')

from libs.utils import *
import libs.yaml_utils as yaml_utils
from libs.ClassificationModel import ClassificationModel

app = Flask(__name__)
CORS(app)

# Load configuration
CONFIG_PATH = "./supervised_classification/configuration.yml"
config = yaml_utils.Config(yaml.load(open(CONFIG_PATH), Loader=yaml.SafeLoader))

# Model parameters
class Args:
    def __init__(self):
        self.image_type = "rgb"
        self.n_classes = 2
        self.is_max_pool = True
        self.dir = "supervised_classifier_lr1e-4_one_by_one"
        self.checkpoint_dir = "rgb_supervised_classifier_n_2_one_by_one"
        self.is_debug = True

args = Args()

# Initialize the model
model = ClassificationModel(config, args)
Classifier = model.ClassifierModel

# Find and load the latest checkpoint
def find_latest_checkpoint():
    checkpoint_dir = f"checkpoints/{args.checkpoint_dir}"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    files = os.listdir(checkpoint_dir)
    checkpoint_files = [f for f in files if f.endswith('.weights.h5') and 'final' in f]
    
    if not checkpoint_files:
        checkpoint_files = [f for f in files if f.endswith('.weights.h5')]
    
    if not checkpoint_files:
        print(f"No checkpoint file found in: {checkpoint_dir}")
        return None
    
    # Find the latest checkpoint
    checkpoint_files.sort(reverse=True)
    return os.path.join(checkpoint_dir, checkpoint_files[0])

# Process image helper function
def process_image(image_data):
    try:
        # Convert to skimage format from bytes
        image = skimage.io.imread(io.BytesIO(image_data))
        
        # Center crop
        target_size = config.models['classifier']["target_size"]
        h, w = image.shape[:2]
        start_h = (h - target_size) // 2
        start_w = (w - target_size) // 2
        image = image[start_h:start_h+target_size, start_w:start_w+target_size]
        
        # Normalize
        image = image.astype(np.float64)
        
        # Add batch dimension and prepare for model
        image_batch = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = Classifier.predict(image_batch)
        
        # For binary classification
        if args.n_classes == 2:
            result = {
                "isValid": bool(np.argmax(prediction[0]) == 0),
                "confidence": float(prediction[0][np.argmax(prediction[0])]),
                "predictions": prediction[0].tolist()
            }
        else:
            # For multi-class classification
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

# Add a specific endpoint for the SwiftQR app
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

@app.route('/api/status', methods=['GET'])
def status():
    checkpoint_path = find_latest_checkpoint()
    
    return jsonify({
        "status": "running",
        "model_loaded": checkpoint_path is not None,
        "checkpoint_path": checkpoint_path,
        "model_classes": args.n_classes,
        "image_type": args.image_type
    })

# Initialize model at startup
def initialize():
    checkpoint_path = find_latest_checkpoint()
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        Classifier.load_weights(checkpoint_path)
        print("Model loaded successfully")
    else:
        print("WARNING: No checkpoint found, model will not work correctly")

if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=True) 