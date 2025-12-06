"""Router — forwards prompt to a healthy worker.

Simple behavior:
- Check worker health (GET /health)
- Forward POST /chat to a healthy worker
- Retry a few times before failing
"""

import os
import random
import sys
from typing import List

import requests
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    HOST,
    PORT,
    WORKER_SERVICE,
    WORKER_PORT,
    WORKER_URLS,
    HEALTH_TIMEOUT,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    LOG_LEVEL,
)
from utils.logging_config import error_tracker, setup_logger

app = Flask(__name__)
log = setup_logger("router", LOG_LEVEL)


def is_healthy(url: str) -> bool:
    """Return True if worker at url responds OK to /health quickly."""
    try:
        r = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False


def worker_list() -> List[str]:
    """Give candidate worker URLs.

If WORKER_SERVICE set (Swarm), prefer service name; otherwise use WORKER_URLS.
"""
    if WORKER_SERVICE:
        return [f"http://{WORKER_SERVICE}:{WORKER_PORT}"]
    return [u.strip() for u in WORKER_URLS.split(",") if u.strip()]


@app.route("/chat", methods=["POST"])
def chat() -> tuple:
    """Accept {"prompt": "..."} and return worker result.

Quick flow:
- validate prompt
- find healthy workers
- try a few and return first successful response
"""
    try:
        data = request.get_json(force=True)
        prompt = (data or {}).get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        candidates = worker_list()
        healthy = [w for w in candidates if is_healthy(w)]
        if not healthy:
            error_tracker.record("no_workers")
            return jsonify({"error": "no healthy workers"}), 503

        # Try up to MAX_RETRIES different workers (random order)
        tries = 0
        attempted = set()
        while tries < MAX_RETRIES and len(attempted) < len(healthy):
            w = random.choice([x for x in healthy if x not in attempted])
            attempted.add(w)
            tries += 1
            try:
                r = requests.post(f"{w}/chat", json={"prompt": prompt}, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    return jsonify(r.json()), 200
                error_tracker.record(f"worker_status_{r.status_code}")
            except requests.RequestException:
                error_tracker.record("worker_request_error")

        error_tracker.record("all_workers_failed")
        return jsonify({"error": "all workers failed"}), 503

    except Exception as exc:
        log.exception("router error: %s", exc)
        return jsonify({"error": "internal error"}), 500


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """Return router and worker summary."""
    candidates = worker_list()
    healthy = [w for w in candidates if is_healthy(w)]
    return jsonify({"status": "ok", "workers": len(candidates), "healthy": len(healthy)}), 200


@app.route("/workers", methods=["GET"])
def workers() -> tuple:
    """Per-worker health map."""
    candidates = worker_list()
    return jsonify({w: is_healthy(w) for w in candidates}), 200


@app.route("/errors", methods=["GET"])
def errors() -> tuple:
    """Error summary."""
    return jsonify(error_tracker.get_summary()), 200


if __name__ == "__main__":
    log.info("router starting on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
