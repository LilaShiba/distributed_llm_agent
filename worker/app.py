"""Zen worker - simple model inference."""

from flask import Flask, request, jsonify
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HOST, PORT, MODEL_NAME, MAX_LENGTH, TEMPERATURE, DO_SAMPLE, LOG_LEVEL
from logging_config import setup_logger, error_tracker

app = Flask(__name__)
log = setup_logger("worker", LOG_LEVEL)

model = None


def get_model():
    """Load model on first use."""
    global model
    if model is None:
        log.info(f"loading model: {MODEL_NAME}")
        try:
            # Import transformers lazily so the module can be imported without the
            # heavy dependency present. If the import fails, we surface a clear
            # error when attempting to use the model.
            from transformers import pipeline
            model = pipeline("text-generation", model=MODEL_NAME)
            log.info("model ready")
        except Exception as e:
            error_tracker.record("model_load_error")
            log.error(f"failed to load model: {e}")
            raise
    return model


@app.route("/chat", methods=["POST"])
def chat():
    """Generate text from prompt."""
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt")
        
        if not prompt:
            return jsonify({"error": "no prompt"}), 400

        log.info(f"prompt: {len(prompt)} chars")
        
        try:
            generator = get_model()
            result = generator(
                prompt,
                max_length=MAX_LENGTH,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE
            )
            text = result[0]["generated_text"]
            log.info(f"response: {len(text)} chars")
            return jsonify({"response": text, "model": MODEL_NAME})
        except Exception as e:
            error_tracker.record("inference_error")
            log.error(f"inference failed: {e}")
            return jsonify({"error": "inference failed"}), 500
            
    except Exception as e:
        log.exception("unexpected error")
        error_tracker.record("internal_error")
        return jsonify({"error": "internal error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    try:
        get_model()
        return jsonify({"status": "ok", "model": MODEL_NAME})
    except Exception as e:
        error_tracker.record("health_check_error")
        log.error(f"health check failed: {e}")
        return jsonify({"status": "error", "model": MODEL_NAME}), 500


if __name__ == "__main__":
    log.info(f"worker starting on {HOST}:{PORT}")
    log.info(f"model: {MODEL_NAME}")
    app.run(host=HOST, port=PORT, debug=False, threaded=False)
