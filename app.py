import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# DigitalOcean requires 8080 port
PORT = int(os.environ.get("PORT", 8080))

# Simplified model loading
try:
    classifier = pipeline("text-classification", model=".")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model load failed: {str(e)}")
    classifier = None

@app.route('/')
def health_check():
    return jsonify({
        "status": "ready" if classifier else "error",
        "model": "loaded" if classifier else "missing"
    })

@app.route('/api/respond', methods=['POST'])
def respond():
    if not classifier:
        return jsonify({"error": "Model not loaded"}), 503
    
    data = request.get_json()
    message = data.get('message', '')[:500]  # Limit input length
    
    try:
        result = classifier(message)[0]
        return jsonify({
            "reply": f"Predicted: {result['label']} ({result['score']:.2f})"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=PORT, debug=False)
