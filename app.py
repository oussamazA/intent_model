import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

generator = None
try:
    app.logger.info("Loading tokenizer & model…")
    tokenizer = AutoTokenizer.from_pretrained(".")
    model     = AutoModelForCausalLM.from_pretrained(".")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    app.logger.info("Model loaded successfully!")
except Exception as e:
    app.logger.error(f"Model load failed: {e}")

@app.route("/api/respond", methods=["POST"])
def respond():
    if generator is None:
        return jsonify({"error": "Model not initialized"}), 503

    data = request.get_json(force=True)
    msg  = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    try:
        out = generator(msg, num_return_sequences=1)
        return jsonify({"reply": out[0]["generated_text"]})
    except Exception as e:
        app.logger.error(f"Generation error: {e}")
        return jsonify({"error": "Generation failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
