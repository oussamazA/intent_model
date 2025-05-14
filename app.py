import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline  # Changed to use generic pipeline

app = Flask(__name__)
CORS(app)

# Initialize model pipeline
classifier = None

def load_model():
    global classifier
    try:
        # Use zero-code classification for simplicity
        classifier = pipeline("text-classification", model=".")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

load_model()

@app.route('/api/respond', methods=['POST'])
def respond():
    if classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True)
    message = data.get('message', '')
    
    try:
        # Get prediction
        result = classifier(message)[0]
        
        # Map label to responses
        responses = {
            "greeting": "Hello! How can I help you today?",
            "account": "I can help with account-related questions.",
            "support": "For technical support, please visit our help center.",
            "default": "Could you please clarify your request?"
        }
        
        reply = responses.get(result["label"], responses["default"])
        
        return jsonify({'reply': reply})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
