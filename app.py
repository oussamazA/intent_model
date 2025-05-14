import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)

# Load model and tokenizer during startup
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    try:
        # Load tokenizer from local files
        tokenizer = AutoTokenizer.from_pretrained(".")
        
        # Load model from safetensors
        model = AutoModelForSequenceClassification.from_pretrained(".")
        
        print("Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Load the model when the app starts
load_model()

@app.route('/api/respond', methods=['POST'])
def respond():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True)
    message = data.get('message', '')
    
    try:
        # Tokenize input
        inputs = tokenizer(
            message,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predicted label
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Add your custom responses based on predicted class
        responses = [
            "Hello! How can I help you today?",
            "I can help with account-related questions.",
            "For technical support, please visit our help center.",
            "Could you please clarify your request?"
        ]
        
        reply = responses[predicted_class % len(responses)]
        
        return jsonify({'reply': reply})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
