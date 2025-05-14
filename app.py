import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

#
# 1) Configuration: set your HF model ID via an env var HF_MODEL_ID
#    e.g. "gpt2" or "username/my-finetuned-chatbot"
#
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "gpt2")

generator = None
try:
    app.logger.info(f"Loading tokenizer/model from HuggingFace: {HF_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        max_length=256,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load model `{HF_MODEL_ID}`: {e}")
    generator = None

@app.route("/api/respond", methods=["POST"])
def respond():
    if generator is None:
        return jsonify({"error": "Model not initialized"}), 503

    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    try:
        outputs = generator(message, max_length=256, num_return_sequences=1)
        reply = outputs[0].get("generated_text", "")
        return jsonify({"reply": reply})
    except Exception as e:
        app.logger.error(f"Generation error: {e}")
        return jsonify({"error": "Generation failed"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
