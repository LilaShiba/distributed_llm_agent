from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load a small or quantized language model for Raspberry Pi
# You can replace "distilgpt2" with any lightweight model
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
generator = pipeline("text-generation", model=MODEL_NAME)

@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives a prompt and returns generated text from the LLM.
    """
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = generator(prompt, max_length=100, do_sample=True)
        text = result[0]['generated_text']
        return jsonify({"response": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "model": MODEL_NAME})
