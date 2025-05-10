from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

app = Flask(__name__)

# load model & tokenizer from the folder you saved in Colab
MODEL_DIR = "intent_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        idx    = int(np.argmax(probs))
        confidence = float(probs[idx])

    # map back to tag
    id2tag = {v:k for k,v in model.config.label2id.items()} if hasattr(model.config, "label2id") else {}
    tag = id2tag.get(idx, str(idx))

    return jsonify({
        "intent": tag,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
