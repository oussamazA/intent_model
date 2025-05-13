import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# If you're using PyTorch to load your model:
import torch

app = Flask(__name__)
CORS(app)

# Load your model once on startup (adjust path & loading as needed)
MODEL_PATH = "intent_model/pytorch_model.bin"
# Example: torch.load; replace with your actual loading code
model = torch.load(MODEL_PATH, map_location='cpu')
model.eval()

@app.route('/api/respond', methods=['POST'])
def respond():
    """
    Expects JSON: { "userId": "<id>", "message": "<text>" }
    Returns JSON: { "reply": "<generated reply>" }
    """
    data = request.get_json(force=True)
    user_id = data.get('userId', '')
    message = data.get('message', '')

    # TODO: Replace with your real inference logic:
    # inputs = tokenizer(message, return_tensors='pt')
    # outputs = model.generate(**inputs)
    # reply_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply_text = "This is a placeholder reply."

    return jsonify({'reply': reply_text})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
