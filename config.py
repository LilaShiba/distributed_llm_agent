"""Zen configuration - simple and clear."""

import os

# System
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

# Network
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
WORKER_URLS = os.getenv("WORKER_URLS", "http://localhost:5001,http://localhost:5002")

# Docker Swarm: set a single service name (e.g. 'worker') to let Swarm LB handle routing
WORKER_SERVICE = os.getenv("WORKER_SERVICE", "")
WORKER_PORT = int(os.getenv("WORKER_PORT", 5000))

# Timeouts
HEALTH_TIMEOUT = int(os.getenv("HEALTH_TIMEOUT", 2))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 100))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
DO_SAMPLE = os.getenv("DO_SAMPLE", "True").lower() == "true"
