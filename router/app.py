"""Zen router - simple request forwarding."""

import os
import random
import sys

import requests
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    HEALTH_TIMEOUT,
    HOST,
    LOG_LEVEL,
    MAX_RETRIES,
    PORT,
    REQUEST_TIMEOUT,
    WORKER_PORT,
    WORKER_SERVICE,
    WORKER_URLS,
)
from utils.logging_config import error_tracker, setup_logger

app = Flask(__name__)
log = setup_logger("router", LOG_LEVEL)


def is_healthy(url: str) -> bool:
    """Check if worker is healthy."""
    try:
        resp = requests.get(f"{url}/health", timeout=HEALTH_TIMEOUT)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def get_workers() -> list[str]:
    """Get list of healthy workers."""
    if WORKER_SERVICE:
        url = f"http://{WORKER_SERVICE}:{WORKER_PORT}"
        if is_healthy(url):
            return [url]

    urls = [u.strip() for u in WORKER_URLS.split(",") if u.strip()]
    return [url for url in urls if is_healthy(url)]


@app.route("/chat", methods=["POST"])
def chat() -> tuple:
    """Forward chat request to a healthy worker."""
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        log.info(f"routing: {len(prompt)} chars")

        for attempt in range(MAX_RETRIES):
            workers = get_workers()
            if not workers:
                error_tracker.record("no_workers")
                return jsonify({"error": "no workers available"}), 503

            worker = random.choice(workers)
            try:
                log.info(f"attempt {attempt + 1}/{MAX_RETRIES}: {worker}")
                resp = requests.post(
                    f"{worker}/chat",
                    json={"prompt": prompt},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    log.info("success")
                    return jsonify(resp.json()), 200

                error_tracker.record(f"worker_error_{resp.status_code}")

            except requests.Timeout:
                error_tracker.record("timeout")
                log.warning("timeout")
            except requests.RequestException as e:
                error_tracker.record("request_error")
                log.warning(f"request error: {e}")

        error_tracker.record("max_retries_exceeded")
        return jsonify({"error": "all workers failed"}), 503

    except ValueError:
        return jsonify({"error": "invalid json"}), 400
    except Exception as e:
        log.error(f"unexpected error: {e}")
        return jsonify({"error": "internal error"}), 500


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """Router health status."""
    workers = [u.strip() for u in WORKER_URLS.split(",") if u.strip()]
    healthy = [w for w in workers if is_healthy(w)]
    return (
        jsonify({"status": "ok", "workers": len(workers), "healthy": len(healthy)}),
        200,
    )


@app.route("/workers", methods=["GET"])
def workers() -> tuple:
    """Worker status details."""
    worker_list = [u.strip() for u in WORKER_URLS.split(",") if u.strip()]
    return jsonify({w: is_healthy(w) for w in worker_list}), 200


@app.route("/errors", methods=["GET"])
def errors() -> tuple:
    """Error summary."""
    return jsonify(error_tracker.get_summary()), 200


if __name__ == "__main__":
    log.info(f"router starting on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
