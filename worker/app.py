"""Zen worker - simple model inference."""

import os
import sys
from typing import Any

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DO_SAMPLE,
    HOST,
    LOG_LEVEL,
    MAX_LENGTH,
    MODEL_NAME,
    PORT,
    TEMPERATURE,
)
from utils.logging_config import error_tracker, setup_logger

app = Flask(__name__)
log = setup_logger("worker", LOG_LEVEL)

model: Any = None


def get_model() -> Any:
    """Load model on first use (lazy loading)."""
    global model
    if model is None:
        log.info(f"loading model: {MODEL_NAME}")
        try:
            from transformers import pipeline

            model = pipeline("text-generation", model=MODEL_NAME)
            log.info("model ready")
        except Exception as e:
            error_tracker.record("model_load_error")
            log.error(f"failed to load model: {e}")
            raise

    return model


@app.route("/chat", methods=["POST"])
def chat() -> tuple:
    """Generate text from prompt."""
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        log.info(f"generating: {len(prompt)} chars")
        generator = get_model()
        result = generator(
            prompt,
            max_length=MAX_LENGTH,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
        )
        text = result[0]["generated_text"]
        return jsonify({"response": text, "model": MODEL_NAME}), 200

    except ValueError:
        return jsonify({"error": "invalid json"}), 400
    except Exception as e:
        error_tracker.record("inference_error")
        log.error(f"inference failed: {e}")
        return jsonify({"error": "inference failed"}), 500


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """Health check."""
    try:
        get_model()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        log.error(f"health check failed: {e}")
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    log.info(f"worker starting on {HOST}:{PORT} with model {MODEL_NAME}")
    app.run(host=HOST, port=PORT, debug=False, threaded=False, use_reloader=False)
