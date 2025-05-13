import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/api/respond', methods=['POST'])
def respond():
    """
    Expects JSON: { "userId": "<id>", "message": "<text>" }
    Returns JSON: { "reply": "<generated reply>" }
    """
    data = request.get_json(force=True)
    message = data.get('message', '').lower()
    
    # Simple rule-based responses
    if any(word in message for word in ['hello', 'hi', 'hey']):
        reply = "Hello! How can I help you today?"
    elif 'help' in message:
        reply = "I'm here to assist you. What do you need help with?"
    else:
        reply = "I understand. Could you please clarify your request?"

    return jsonify({'reply': reply})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
