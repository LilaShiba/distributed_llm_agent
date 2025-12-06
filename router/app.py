"""Zen router - simple request forwarding."""

from flask import Flask, request, jsonify
import requests
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HOST, PORT, WORKER_URLS, WORKER_SERVICE, WORKER_PORT, HEALTH_TIMEOUT, REQUEST_TIMEOUT, MAX_RETRIES, LOG_LEVEL
from logging_config import setup_logger, error_tracker

app = Flask(__name__)
log = setup_logger("router", LOG_LEVEL)

WORKERS = [w.strip() for w in WORKER_URLS.split(",")]


def is_healthy(url: str) -> bool:
    """Check if worker is healthy."""
    try:
        resp = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
        return resp.status_code == 200
    except Exception:
        return False


def get_healthy_workers() -> list:
    """Get list of healthy workers."""
    # If running in Docker Swarm, prefer the service name which Swarm will resolve
    if WORKER_SERVICE:
        service_url = f"http://{WORKER_SERVICE}:{WORKER_PORT}"
        if is_healthy(service_url):
            return [service_url]
        logger.warning("WORKER_SERVICE set but service unhealthy: %s", service_url)

    urls = [u.strip() for u in WORKER_URLS.split(",") if u.strip()]
    healthy = []
    for url in urls:
        if is_healthy(url):
            healthy.append(url)
    if not healthy:
        logger.warning("No healthy workers found via WORKER_URLS or WORKER_SERVICE")
    return healthy


@app.route("/chat", methods=["POST"])
def chat():
    """Forward chat requests to a worker."""
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt")
        
        if not prompt:
            return jsonify({"error": "no prompt"}), 400

        log.info(f"prompt: {len(prompt)} chars")
        
        for attempt in range(MAX_RETRIES):
            try:
                worker = random.choice(get_healthy_workers())
                log.info(f"attempt {attempt + 1}: {worker}")
                
                resp = requests.post(
                    f"{worker}/chat",
                    json={"prompt": prompt},
                    timeout=REQUEST_TIMEOUT
                )
                
                if resp.status_code == 200:
                    log.info(f"success: {worker}")
                    return jsonify(resp.json())
                else:
                    error_tracker.record(f"worker_status_{resp.status_code}")
                    log.warning(f"status {resp.status_code}")
                    
            except requests.Timeout:
                error_tracker.record("timeout")
                log.error("timeout")
            except Exception as e:
                error_tracker.record("error")
                log.error(str(e))
        
        error_tracker.record("no_workers")
        return jsonify({"error": "workers unavailable"}), 503
        
    except Exception as e:
        log.exception("unexpected error")
        return jsonify({"error": "internal error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    healthy = [w for w in WORKERS if is_healthy(w)]
    return jsonify({
        "status": "ok",
        "total": len(WORKERS),
        "healthy": len(healthy)
    })


@app.route("/workers", methods=["GET"])
def workers():
    """Worker status."""
    return jsonify({w: is_healthy(w) for w in WORKERS})


@app.route("/errors", methods=["GET"])
def errors():
    """Error summary."""
    return jsonify(error_tracker.get_summary())


if __name__ == "__main__":
    log.info(f"router starting on {HOST}:{PORT}")
    log.info(f"workers: {WORKERS}")
    app.run(host=HOST, port=PORT, debug=False)
