"""
Worker — simple retrieval over PDFs using DPR.

Short and sweet:
- Put PDFs in pdf_corpus/
- Embeddings are built once and cached
- POST /chat {"prompt": "..."} → top passages
- POST /upload → add PDF and reload
- POST /reload → rebuild embeddings
- GET /health → readiness check

Security:
- Forces safetensors to avoid torch.load()
  (CVE-2025-32434)
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request
from pypdf import PdfReader
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
)
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------
# Correct project paths (as provided)
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from config import HOST, PORT, LOG_LEVEL  # noqa: E402
from utils.logging_config import error_tracker, setup_logger  # noqa: E402


# ---------------------------------------------------------------------
# App / logging
# ---------------------------------------------------------------------

app = Flask(__name__)
log = setup_logger("worker", LOG_LEVEL)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PDF_DIR = os.getenv("PDF_DIR", "pdf_corpus")
DATA_DIR = os.getenv("DATA_DIR", "data")

QUESTION_MODEL = os.getenv(
    "DPR_QUESTION_ENCODER",
    "facebook/dpr-question_encoder-single-nq-base",
)
CONTEXT_MODEL = os.getenv(
    "DPR_CONTEXT_ENCODER",
    "facebook/dpr-ctx_encoder-single-nq-base",
)

BATCH_SIZE = int(os.getenv("ENCODE_BATCH", "32"))

os.makedirs(DATA_DIR, exist_ok=True)

PERSIST_CHUNKS = os.path.join(DATA_DIR, "corpus_chunks.json")
PERSIST_EMB = os.path.join(DATA_DIR, "corpus_embeddings.npz")
PERSIST_META = os.path.join(DATA_DIR, "meta.json")


# ---------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------

_question_encoder: Optional[DPRQuestionEncoder] = None
_question_tokenizer: Optional[DPRQuestionEncoderTokenizer] = None
_context_encoder: Optional[DPRContextEncoder] = None
_context_tokenizer: Optional[DPRContextEncoderTokenizer] = None

corpus_chunks: List[str] = []
corpus_embeddings: Optional[np.ndarray] = None


# ---------------------------------------------------------------------
# Model loading (SAFE: safetensors only)
# ---------------------------------------------------------------------

def load_question_encoder() -> Tuple[
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
]:
    """Load DPR question encoder (singleton)."""
    global _question_encoder, _question_tokenizer

    if _question_encoder is None:
        log.info("Loading question encoder: %s", QUESTION_MODEL)
        _question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            QUESTION_MODEL
        )
        _question_encoder = DPRQuestionEncoder.from_pretrained(
            QUESTION_MODEL,
            use_safetensors=True,
        ).to(DEVICE)
        _question_encoder.eval()

    return _question_encoder, _question_tokenizer


def load_context_encoder() -> Tuple[
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
]:
    """Load DPR context encoder (singleton)."""
    global _context_encoder, _context_tokenizer

    if _context_encoder is None:
        log.info("Loading context encoder: %s", CONTEXT_MODEL)
        _context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            CONTEXT_MODEL
        )
        _context_encoder = DPRContextEncoder.from_pretrained(
            CONTEXT_MODEL,
            use_safetensors=True,
        ).to(DEVICE)
        _context_encoder.eval()

    return _context_encoder, _context_tokenizer


# ---------------------------------------------------------------------
# Corpus utilities
# ---------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 1_000) -> List[str]:
    """Split text into fixed-size chunks."""
    return [
        text[i:i + max_chars].strip()
        for i in range(0, len(text), max_chars)
        if text[i:i + max_chars].strip()
    ]


def load_corpus() -> List[str]:
    """Read PDFs and return text chunks."""
    passages: List[str] = []

    for path in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        try:
            reader = PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            chunks = chunk_text(text)
            passages.extend(chunks)
            log.info("Loaded %d chunks from %s", len(chunks), path)
        except Exception as exc:
            log.warning("Skipping %s: %s", path, exc)

    return passages


def scan_meta() -> List[dict]:
    """Return PDF path + mtime metadata."""
    return [
        {"path": p, "mtime": os.path.getmtime(p)}
        for p in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    ]


def load_persisted() -> Tuple[List[str], Optional[np.ndarray], List[dict]]:
    """Load cached corpus if present."""
    if not all(map(os.path.exists, (PERSIST_CHUNKS, PERSIST_EMB, PERSIST_META))):
        return [], None, []

    try:
        with open(PERSIST_CHUNKS, encoding="utf-8") as fh:
            chunks = json.load(fh)

        embeddings = np.load(PERSIST_EMB)["embeddings"]

        with open(PERSIST_META, encoding="utf-8") as fh:
            meta = json.load(fh)

        return chunks, embeddings, meta

    except Exception as exc:
        log.warning("Failed to load cache: %s", exc)
        return [], None, []


def save_persisted(
    chunks: List[str],
    embeddings: np.ndarray,
    meta: List[dict],
) -> None:
    """Persist corpus to disk (best effort)."""
    try:
        with open(PERSIST_CHUNKS, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh)

        np.savez_compressed(PERSIST_EMB, embeddings=embeddings)

        with open(PERSIST_META, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except Exception as exc:
        log.warning("Failed to persist corpus: %s", exc)


# ---------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------

def prepare_corpus() -> None:
    """Load cached corpus or rebuild if PDFs changed."""
    global corpus_chunks, corpus_embeddings

    current_meta = scan_meta()
    cached_chunks, cached_embeddings, cached_meta = load_persisted()

    if cached_meta == current_meta and cached_embeddings is not None:
        corpus_chunks = cached_chunks
        corpus_embeddings = cached_embeddings
        return

    if not current_meta:
        corpus_chunks = []
        corpus_embeddings = None
        return

    ctx_encoder, ctx_tokenizer = load_context_encoder()
    corpus_chunks = load_corpus()

    all_embeddings: List[np.ndarray] = []

    for i in range(0, len(corpus_chunks), BATCH_SIZE):
        batch = corpus_chunks[i:i + BATCH_SIZE]
        inputs = ctx_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            vecs = ctx_encoder(**inputs).pooler_output.cpu().numpy()
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            all_embeddings.append(vecs)

    corpus_embeddings = np.vstack(all_embeddings)
    save_persisted(corpus_chunks, corpus_embeddings, current_meta)


def find_top_k(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Return top-k passages by cosine similarity."""
    if corpus_embeddings is None:
        return [("No corpus loaded.", 0.0)]

    q_encoder, q_tokenizer = load_question_encoder()

    inputs = q_tokenizer(
        query,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        q_vec = q_encoder(**inputs).pooler_output[0].cpu().numpy()
        q_vec /= np.linalg.norm(q_vec) + 1e-8

    scores = corpus_embeddings @ q_vec
    top_idx = np.argsort(-scores)[:k]

    return [(corpus_chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/chat", methods=["POST"])
def chat() -> Tuple[Any, int]:
    """Return relevant passages for a query."""
    try:
        prepare_corpus()
        prompt = (request.get_json(force=True) or {}).get("prompt", "").strip()

        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        results = find_top_k(prompt)
        return jsonify(
            {"results": [{"passage": p, "score": s} for p, s in results]}
        ), 200

    except Exception as exc:
        error_tracker.record("retrieval_error")
        log.exception("chat failed: %s", exc)
        return jsonify({"error": "retrieval failed"}), 500


@app.route("/health", methods=["GET"])
def health() -> Tuple[Any, int]:
    """Health check endpoint."""
    try:
        prepare_corpus()
        return jsonify(
            {
                "status": "ok" if corpus_embeddings is not None else "no_corpus",
                "passages": len(corpus_chunks),
            }
        ), 200
    except Exception as exc:
        log.exception("health failed: %s", exc)
        return jsonify({"status": "error"}), 500


@app.route("/reload", methods=["POST"])
def reload_endpoint() -> Tuple[Any, int]:
    """Force corpus reload."""
    try:
        for path in (PERSIST_CHUNKS, PERSIST_EMB, PERSIST_META):
            if os.path.exists(path):
                os.remove(path)

        prepare_corpus()
        return jsonify({"status": "ok", "passages": len(corpus_chunks)}), 200
    except Exception as exc:
        log.exception("reload failed: %s", exc)
        return jsonify({"error": "reload failed"}), 500


@app.route("/upload", methods=["POST"])
def upload_pdf() -> Tuple[Any, int]:
    """Upload a PDF and trigger reload."""
    try:
        f = request.files.get("file")

        if not f or not f.filename:
            return jsonify({"error": "no file"}), 400

        if not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "only pdf allowed"}), 400

        os.makedirs(PDF_DIR, exist_ok=True)
        filename = secure_filename(f.filename)
        f.save(os.path.join(PDF_DIR, filename))

        prepare_corpus()
        return jsonify({"status": "uploaded", "file": filename}), 201

    except Exception as exc:
        log.exception("upload failed: %s", exc)
        return jsonify({"error": "upload failed"}), 500


# ---------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Worker starting on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, threaded=False)
