"""Zen worker - DPR-based retrieval over PDF corpus."""

import os
import sys
from typing import Any, List, Tuple, Optional

from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HOST, PORT, LOG_LEVEL
from utils.logging_config import error_tracker, setup_logger

app = Flask(__name__)
log = setup_logger("worker", LOG_LEVEL)

# Globals for models and corpus
question_encoder: Optional[Any] = None
context_encoder: Optional[Any] = None
corpus_chunks: List[str] = []
corpus_embeddings: Optional[Any] = None

PDF_DIR = os.getenv("PDF_DIR", "pdf_corpus")
DPR_QUESTION_ENCODER = os.getenv(
    "DPR_QUESTION_ENCODER", "facebook/dpr-question_encoder-single-nq-base"
)
DPR_CONTEXT_ENCODER = os.getenv(
    "DPR_CONTEXT_ENCODER", "facebook/dpr-ctx_encoder-single-nq-base"
)


def load_pdfs(pdf_dir: str) -> List[str]:
    """Load and chunk all PDFs in the directory into passages."""
    from pypdf import PdfReader
    import glob

    passages: List[str] = []
    for pdf_path in glob.glob(os.path.join(pdf_dir, "*.pdf")):
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            # Chunk text into ~1000 character passages
            for i in range(0, len(text), 1000):
                chunk = text[i:i + 1000].strip()
                if chunk:
                    passages.append(chunk)
            log.info("Loaded %d passages from %s", len(passages), pdf_path)
        except Exception as exc:
            log.error("Failed to load %s: %s", pdf_path, exc)
    return passages


def encode_corpus(passages: List[str]) -> Any:
    """Encode all passages with the context encoder."""
    import torch
    batch_size = 16
    embeddings = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        emb = context_encoder(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            vecs = context_encoder.model(**emb).pooler_output
        embeddings.append(vecs.cpu())
    return torch.cat(embeddings, dim=0)


def get_models_and_corpus() -> Tuple[Any, Any, List[str], Any]:
    """
    Load DPR models and encode the PDF corpus.
    Returns:
        question_encoder, context_encoder, corpus_chunks, corpus_embeddings
    """
    global question_encoder, context_encoder, corpus_chunks, corpus_embeddings
    if (
        question_encoder is not None
        and context_encoder is not None
        and corpus_embeddings is not None
    ):
        return question_encoder, context_encoder, corpus_chunks, corpus_embeddings

    from transformers import (
        DPRQuestionEncoder,
        DPRQuestionEncoderTokenizer,
        DPRContextEncoder,
        DPRContextEncoderTokenizer,
        pipeline,
    )

    # Load models
    log.info("Loading DPR question encoder: %s", DPR_QUESTION_ENCODER)
    question_encoder_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
        DPR_QUESTION_ENCODER
    )
    question_encoder_model = DPRQuestionEncoder.from_pretrained(DPR_QUESTION_ENCODER)
    question_encoder = pipeline(
        "feature-extraction",
        model=question_encoder_model,
        tokenizer=question_encoder_tokenizer,
    )

    log.info("Loading DPR context encoder: %s", DPR_CONTEXT_ENCODER)
    context_encoder_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        DPR_CONTEXT_ENCODER
    )
    context_encoder_model = DPRContextEncoder.from_pretrained(DPR_CONTEXT_ENCODER)
    context_encoder = pipeline(
        "feature-extraction",
        model=context_encoder_model,
        tokenizer=context_encoder_tokenizer,
    )

    # Load and encode corpus
    log.info("Loading PDF corpus from %s", PDF_DIR)
    corpus_chunks = load_pdfs(PDF_DIR)
    if not corpus_chunks:
        log.warning("No PDF passages found in corpus.")
        corpus_embeddings = None
    else:
        log.info("Encoding %d passages...", len(corpus_chunks))
        import torch

        with torch.no_grad():
            corpus_embeddings_list = []
            for passage in corpus_chunks:
                emb = context_encoder(passage)
                # emb: [1, tokens, 768], take mean over tokens
                emb_vec = sum(emb[0]) / len(emb[0])
                corpus_embeddings_list.append(emb_vec)
            corpus_embeddings = torch.tensor(corpus_embeddings_list)
        log.info("Corpus encoding complete.")

    return question_encoder, context_encoder, corpus_chunks, corpus_embeddings


def find_best_passages(query: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """Find top_k most relevant passages for the query."""
    import torch

    question_encoder_, context_encoder_, passages, embeddings = get_models_and_corpus()
    if not passages or embeddings is None:
        return [("No corpus loaded.", 0.0)]

    # Encode query
    emb = question_encoder_(query)
    emb_vec = sum(emb[0]) / len(emb[0])
    emb_vec = torch.tensor(emb_vec).unsqueeze(0)  # [1, 768]

    # Compute cosine similarity
    sim = torch.nn.functional.cosine_similarity(embeddings, emb_vec)
    topk = torch.topk(sim, min(top_k, len(passages)))
    results = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        results.append((passages[idx], float(score)))
    return results


@app.route("/chat", methods=["POST"])
def chat() -> Tuple[Any, int]:
    """Retrieve most relevant passage(s) for the prompt."""
    try:
        data = request.get_json(force=True)
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "prompt required"}), 400

        log.info("retrieving for: %s...", prompt[:60])
        results = find_best_passages(prompt, top_k=3)
        response = [{"passage": p, "score": s} for p, s in results]
        return jsonify({"results": response}), 200

    except ValueError:
        return jsonify({"error": "invalid json"}), 400
    except Exception as exc:
        error_tracker.record("retrieval_error")
        log.error("retrieval failed: %s", exc)
        return jsonify({"error": "retrieval failed"}), 500


@app.route("/finetune", methods=["POST"])
def finetune() -> Tuple[Any, int]:
    """Stub endpoint for finetuning DPR on new data."""
    # In practice, finetuning DPR requires offline training.
    # Here, just acknowledge the request.
    return jsonify({"status": "finetune not implemented in API; run offline"}), 501


@app.route("/health", methods=["GET"])
def health() -> Tuple[Any, int]:
    """Health check."""
    try:
        get_models_and_corpus()
        return jsonify({"status": "ok"}), 200
    except Exception as exc:
        log.error("health check failed: %s", exc)
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    log.info("worker starting on %s:%s with DPR model", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, threaded=False, use_reloader=False)
