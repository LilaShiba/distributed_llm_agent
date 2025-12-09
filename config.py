"""Zen configuration - simple and clear."""

import os

# System
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGS_DIR = os.getenv("LOGS_DIR", "logs")

# Network
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
WORKER_URLS = os.getenv(
    "WORKER_URLS", "http://worker1:5000,http://worker2:5000"
)
WORKER_SERVICE = os.getenv("WORKER_SERVICE", "")
WORKER_PORT = int(os.getenv("WORKER_PORT", 5000))

# Timeouts (seconds)
HEALTH_TIMEOUT = int(os.getenv("HEALTH_TIMEOUT", 2))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 2))

# Model
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 100))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
DO_SAMPLE = os.getenv("DO_SAMPLE", "True").lower() == "true"

# DPR Models
DPR_QUESTION_ENCODER = os.getenv("DPR_QUESTION_ENCODER", "facebook/dpr-question_encoder-single-nq-base")
DPR_CONTEXT_ENCODER = os.getenv("DPR_CONTEXT_ENCODER", "facebook/dpr-ctx_encoder-single-nq-base")
PDF_DIR = os.getenv("PDF_DIR", "pdf_corpus")
DATA_DIR = os.getenv("DATA_DIR", "data")
ENCODE_BATCH = int(os.getenv("ENCODE_BATCH", "32"))

# All variables have defaults, so no need for strict checking
