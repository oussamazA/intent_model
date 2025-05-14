import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# Load model/tokenizer once
generator = None
try:
    app.logger.info("Loading tokenizer and model…")
    tokenizer = AutoTokenizer.from_pretrained(".")
    model     = AutoModelForCausalLM.from_pretrained(
        ".", 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    generator = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      device=-1,
      return_full_text=False,
      max_length=256,
      do_sample=True,
      top_k=50,
      top_p=0.95,
    )
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Model load failed: {e}")
    generator = None

@app.route('/api/respond', methods=['POST'])
def respond():
    if generator is None:
        return jsonify({'error': 'Model not initialized'}), 503

    data = request.get_json(force=True)
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        out = generator(message, max_length=256, num_return_sequences=1)
        reply = out[0]['generated_text']
        return jsonify({'reply': reply})
    except Exception as e:
        app.logger.error(f"Generation error: {e}")
        return jsonify({'error': 'Generation failed'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
