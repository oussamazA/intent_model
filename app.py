import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoConfig

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

classifier = None

def load_model():
    global classifier
    try:
        app.logger.info("Starting model load...")
        
        # Verify files exist
        required_files = [
            'config.json', 'model.safetensors',
            'tokenizer.json', 'vocab.txt'
        ]
        for f in required_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing file: {f}")

        # Load with reduced memory footprint
        config = AutoConfig.from_pretrained(".")
        classifier = pipeline(
            "text-classification",
            model=".",
            config=config,
            device=-1,  # Use CPU
            torch_dtype="auto"
        )
        
        app.logger.info("Model loaded successfully!")
        
    except Exception as e:
        app.logger.error(f"Model load failed: {str(e)}")
        classifier = None

load_model()

@app.route('/api/respond', methods=['POST'])
def respond():
    if not classifier:
        return jsonify({'error': 'Model failed to initialize'}), 503
    
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
            
        result = classifier(message[:256])[0]  # Limit input length
        return jsonify({
            'reply': f"Predicted label: {result['label']} (confidence: {result['score']:.2f})"
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
