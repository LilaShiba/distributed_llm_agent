from flask import Flask, request, jsonify
import os
import requests
import random

app = Flask(__name__)

# Get worker URLs from environment variable
# Comma-separated list, e.g., "http://worker1:5000,http://worker2:5000"
WORKER_URLS = os.getenv("WORKER_URLS") or os.getenv("WORKER_SERVICE")
if WORKER_URLS:
    WORKERS = WORKER_URLS.split(",")
else:
    WORKERS = ["http://worker1:5000", "http://worker2:5000"]  # fallback

@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives a prompt and forwards it to a randomly chosen worker.
    Returns the worker's response.
    """
    data = request.get_json()
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Pick a random worker
    worker_url = random.choice(WORKERS)
    try:
        resp = requests.post(
            f"{worker_url}/chat",
            json={"prompt": prompt},
            timeout=10
        )
        return jsonify(resp.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e), "worker": worker_url}), 500

@app.route("/health", methods=["GET"])
def health():
    """Simple health check"""
    return jsonify({"status": "ok", "workers": WORKERS})
