"""
Worker — DPR-based retrieval over PDFs.

Uses Dense Passage Retrieval (DPR) with separate encoders:
- Question encoder: encodes user queries
- Context encoder: encodes document passages

Short and sweet:
- Put PDFs in pdf_corpus/
- Worker builds embeddings once and saves them
- POST /chat with {"prompt": "..."} to get top passages
- POST /upload to add a PDF and auto-reload
- POST /reload to re-scan (use {"force": true} to rebuild)
"""

import glob
import json
import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from flask import Flask, jsonify, request
from pypdf import PdfReader
from transformers import AutoModel, AutoTokenizer
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HOST, PORT, LOG_LEVEL, DPR_QUESTION_ENCODER, DPR_CONTEXT_ENCODER, ENCODE_BATCH
from utils.logging_config import error_tracker, setup_logger

app = Flask(__name__)
log = setup_logger("worker", LOG_LEVEL)

# Simple config / globals
PDF_DIR = os.getenv("PDF_DIR", "pdf_corpus")
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

PERSIST_CHUNKS = os.path.join(DATA_DIR, "corpus_chunks.json")
PERSIST_EMB = os.path.join(DATA_DIR, "corpus_embeddings.npz")
PERSIST_META = os.path.join(DATA_DIR, "meta.json")

# DPR models (loaded on demand)
_question_encoder: Optional[AutoModel] = None
_question_tokenizer: Optional[AutoTokenizer] = None
_context_encoder: Optional[AutoModel] = None
_context_tokenizer: Optional[AutoTokenizer] = None

corpus_chunks: List[str] = []
corpus_embeddings: Optional[np.ndarray] = None


def load_dpr_models() -> Tuple[AutoModel, AutoTokenizer, AutoModel, AutoTokenizer]:
    """Load DPR question and context encoders (once).
    
    Returns:
        Tuple of (question_encoder, question_tokenizer, context_encoder, context_tokenizer)
    """
    global _question_encoder, _question_tokenizer, _context_encoder, _context_tokenizer
    
    if _question_encoder is None:
        log.info("Loading DPR question encoder: %s", DPR_QUESTION_ENCODER)
        _question_tokenizer = AutoTokenizer.from_pretrained(DPR_QUESTION_ENCODER)
        _question_encoder = AutoModel.from_pretrained(DPR_QUESTION_ENCODER)
        _question_encoder.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _question_encoder = _question_encoder.cuda()
            log.info("Question encoder moved to GPU")
    
    if _context_encoder is None:
        log.info("Loading DPR context encoder: %s", DPR_CONTEXT_ENCODER)
        _context_tokenizer = AutoTokenizer.from_pretrained(DPR_CONTEXT_ENCODER)
        _context_encoder = AutoModel.from_pretrained(DPR_CONTEXT_ENCODER)
        _context_encoder.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _context_encoder = _context_encoder.cuda()
            log.info("Context encoder moved to GPU")
    
    return _question_encoder, _question_tokenizer, _context_encoder, _context_tokenizer


def encode_passages(passages: List[str]) -> np.ndarray:
    """Encode passages using DPR context encoder."""
    _, _, ctx_model, ctx_tokenizer = load_dpr_models()
    
    embeddings_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process in batches
    for i in range(0, len(passages), ENCODE_BATCH):
        batch = passages[i : i + ENCODE_BATCH]
        
        # Tokenize batch
        inputs = ctx_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # Encode
        with torch.no_grad():
            outputs = ctx_model(**inputs)
            # Use CLS token (first token) from last hidden state for DPR models
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings_list.append(batch_embeddings)
        
        if (i // ENCODE_BATCH + 1) % 10 == 0:
            log.info("Encoded %d/%d passages", min(i + ENCODE_BATCH, len(passages)), len(passages))
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings_list)
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    return embeddings


def encode_question(question: str) -> np.ndarray:
    """Encode a question using DPR question encoder."""
    q_model, q_tokenizer, _, _ = load_dpr_models()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenize question
    inputs = q_tokenizer(
        question,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Encode
    with torch.no_grad():
        outputs = q_model(**inputs)
        # Use CLS token (first token) from last hidden state for DPR models
        embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    
    # Normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    embedding = embedding / (norm + 1e-8)
    
    return embedding


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Simple fixed-size chunking; keeps implementation tiny and predictable."""
    return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i : i + max_chars].strip()]


def load_corpus(pdf_dir: str = PDF_DIR) -> List[str]:
    """Read PDFs and return a list of text chunks (ordered by filename)."""
    passages: List[str] = []
    for path in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf"))):
        try:
            reader = PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            chunks = chunk_text(text)
            passages.extend(chunks)
            log.info("Loaded %d chunks from %s", len(chunks), path)
        except Exception as exc:
            log.warning("Skipping %s: %s", path, exc)
    return passages


def _scan_meta(pdf_dir: str) -> List[dict]:
    """Return sorted list of {path, mtime} for PDFs in the corpus directory."""
    paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    return [{"path": p, "mtime": os.path.getmtime(p)} for p in paths]


def _load_persisted() -> Tuple[List[str], Optional[np.ndarray], List[dict]]:
    """Load persisted chunks, embeddings, and meta if present and valid."""
    if not (os.path.exists(PERSIST_CHUNKS) and os.path.exists(PERSIST_EMB) and os.path.exists(PERSIST_META)):
        return [], None, []
    try:
        with open(PERSIST_CHUNKS, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)
        npz = np.load(PERSIST_EMB)
        embeddings = npz["embeddings"]
        with open(PERSIST_META, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return chunks, embeddings, meta
    except Exception as exc:
        log.warning("Failed to load persisted corpus: %s", exc)
        return [], None, []


def _save_persisted(chunks: List[str], embeddings: np.ndarray, meta: List[dict]) -> None:
    """Persist chunks, embeddings and metadata. Best-effort atomic writes."""
    try:
        tmp_chunks = PERSIST_CHUNKS + ".tmp"
        with open(tmp_chunks, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh)
        os.replace(tmp_chunks, PERSIST_CHUNKS)
    except Exception as exc:
        log.warning("Failed to save chunks: %s", exc)

    try:
        # save with key 'embeddings' for easy load
        np.savez_compressed(PERSIST_EMB, embeddings=embeddings)
    except Exception as exc:
        log.warning("Failed to save embeddings: %s", exc)

    try:
        tmp_meta = PERSIST_META + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(tmp_meta, PERSIST_META)
    except Exception as exc:
        log.warning("Failed to save meta: %s", exc)


def prepare_corpus() -> None:
    """Load or build corpus embeddings.

    Compare current PDF list (path + mtime) with saved metadata.
    If unchanged, reuse saved embeddings. Otherwise rebuild and save.
    """
    global corpus_chunks, corpus_embeddings
    load_dpr_models()
    current_meta = _scan_meta(PDF_DIR)
    persisted_chunks, persisted_embeddings, persisted_meta = _load_persisted()

    if persisted_meta and current_meta == persisted_meta and persisted_chunks and persisted_embeddings is not None:
        corpus_chunks = persisted_chunks
        corpus_embeddings = persisted_embeddings
        log.info("Using cached corpus (%d passages).", len(corpus_chunks))
        return

    # rebuild from scratch
    log.info("Building corpus from PDFs (%d files)...", len(current_meta))
    corpus_chunks = load_corpus(PDF_DIR)
    if not corpus_chunks:
        corpus_embeddings = None
        log.warning("No PDFs found in %s", PDF_DIR)
        return

    corpus_embeddings = encode_passages(corpus_chunks)
    _save_persisted(corpus_chunks, corpus_embeddings, current_meta)
    log.info("Corpus built and saved (%d passages).", len(corpus_chunks))


def find_top_k(query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Encode query using DPR question encoder and return top-k passages with cosine scores."""
    if not corpus_chunks or corpus_embeddings is None:
        return [("No corpus loaded.", 0.0)]
    
    q_vec = encode_question(query)
    sims = np.dot(corpus_embeddings, q_vec)  # cosine since both normalized
    top_idx = np.argsort(-sims)[:k]
    return [(corpus_chunks[int(i)], float(sims[int(i)])) for i in top_idx]


@app.route("/chat", methods=["POST"])
def chat() -> Tuple[Any, int]:
    """Return most relevant passages for the prompt."""
    try:
        prepare_corpus()
        data = request.get_json(force=True)
        prompt = (data or {}).get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt required"}), 400
        results = find_top_k(prompt, k=3)
        return jsonify({"results": [{"passage": p, "score": s} for p, s in results]}), 200
    except Exception as exc:
        error_tracker.record("retrieval_error")
        log.exception("chat failed: %s", exc)
        return jsonify({"error": "retrieval failed"}), 500


@app.route("/finetune", methods=["POST"])
def finetune() -> Tuple[Any, int]:
    """Finetune stub — real finetuning should be done offline with proper training scripts."""
    return jsonify({"status": "finetune must be run offline"}), 501


@app.route("/health", methods=["GET"])
def health() -> Tuple[Any, int]:
    """Health check: ensure model loads and corpus present if any PDFs exist."""
    try:
        prepare_corpus()
        ok = corpus_embeddings is not None and len(corpus_chunks) > 0
        return jsonify({"status": "ok" if ok else "no_corpus", "passages": len(corpus_chunks)}), 200
    except Exception as exc:
        log.exception("health check failed: %s", exc)
        return jsonify({"status": "error"}), 500


def reload_corpus(force: bool = False) -> dict:
    """Reload corpus; if force=True remove persisted files then rebuild."""
    if force:
        for path in (PERSIST_CHUNKS, PERSIST_EMB, PERSIST_META):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    log.info("Removed persisted file: %s", path)
            except Exception as exc:
                log.warning("Could not remove %s: %s", path, exc)
    prepare_corpus()
    count = len(corpus_chunks) if corpus_chunks else 0
    status = "ok" if corpus_embeddings is not None and count > 0 else "no_corpus"
    return {"status": status, "passages": count}


@app.route("/reload", methods=["POST"])
def reload_endpoint() -> Tuple[Any, int]:
    """Trigger corpus reload. JSON body optional: {'force': true}."""
    try:
        data = request.get_json(silent=True) or {}
        force = bool(data.get("force", False))
        result = reload_corpus(force=force)
        return jsonify(result), 200
    except Exception as exc:
        log.exception("reload failed: %s", exc)
        return jsonify({"error": "reload failed"}), 500


# New: simple upload endpoint to add PDFs to the corpus directory and trigger reload.
@app.route("/upload", methods=["POST"])
def upload_pdf() -> Tuple[Any, int]:
    """Upload a single PDF and trigger a reload.

    Form field: file (multipart/form-data). Only .pdf allowed.
    Returns: {"status": "uploaded", "file": "<name>", "reload": {...}}
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "no file field"}), 400

        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify({"error": "no filename"}), 400

        filename = secure_filename(f.filename)
        if not filename.lower().endswith(".pdf"):
            return jsonify({"error": "only pdf allowed"}), 400

        os.makedirs(PDF_DIR, exist_ok=True)
        save_path = os.path.join(PDF_DIR, filename)
        f.save(save_path)
        log.info("Saved uploaded PDF: %s", filename)

        result = reload_corpus(force=False)
        return jsonify({"status": "uploaded", "file": filename, "reload": result}), 201

    except Exception as exc:
        log.exception("upload failed")
        return jsonify({"error": "upload failed"}), 500


if __name__ == "__main__":
    log.info("worker starting on %s:%s", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, threaded=False, use_reloader=False)
