# Worker & Router Guide — Zen LLM Agent

This guide explains, step-by-step and with plain language, how the router and worker nodes work and how data flows through the system. It is written for developers and operators who want a clear mental model and practical commands.

---

## Overview — short

- Router: receives client requests (POST /chat), finds healthy workers, forwards prompt, returns worker response.
- Worker: loads DPR-style encoders, indexes PDF corpus into vector embeddings, answers retrieval queries by returning top-k relevant passages.
- Persistence: worker saves encoded corpus to disk to avoid re-encoding on every restart.
- Reload: worker provides `/reload` endpoint to re-scan or force rebuild corpus without restarting.

---

## Architecture (high-level)

1. Client → Router (/chat)
2. Router → Healthy Worker (/chat)
3. Worker → (Encode query using question encoder)
4. Worker → (Compute cosine similarity with persisted corpus embeddings)
5. Worker → Respond with top-k passages
6. Router → Return result to client

Router does not run any model; workers host models and corpus.

---

## Worker: responsibilities & behavior

- Load DPR-style models (question encoder, context encoder) via HuggingFace AutoTokenizer/AutoModel.
- Load PDFs from `pdf_corpus/`, chunk large texts into fixed-size passages (~1000 chars).
- Encode all passages with the context encoder into normalized vectors and persist:
  - DATA_DIR/corpus_chunks.json  — list of passage texts
  - DATA_DIR/corpus_embeddings.npz — numpy array with embeddings
  - DATA_DIR/meta.json — metadata (list of {path, mtime})
- On startup, worker compares current PDF metadata with `meta.json`. If identical, the embeddings are reused; otherwise worker rebuilds the entire corpus.
- Provide endpoints:
  - POST /chat — accepts JSON { "prompt": "..." } and returns top passages with scores.
  - POST /reload — accepts optional JSON { "force": true } to remove persisted files and rebuild.
  - GET /health — returns worker status and passage count.
  - POST /finetune — stub (finetune offline).

Why persistence?
- Encoding can be slow and costly. Persisting embeddings avoids re-encoding unchanged PDFs.
- The meta.json check is simple (path + mtime). For stronger guarantees use checksums.

Example worker commands:
- Add PDFs: copy into `pdf_corpus/`
- Normal reload: curl -X POST http://worker:5000/reload
- Force reload: curl -X POST http://worker:5000/reload -H "Content-Type: application/json" -d '{"force": true}'
- Test query: curl -X POST http://worker:5000/chat -H "Content-Type: application/json" -d '{"prompt":"What is the core idea?"}'

Notes:
- GPU optional: code uses CUDA if available.
- If you need incremental fine-grained updates, replace meta-based check with checksums or a small DB.

---

## Router: responsibilities & behavior

- Exposes endpoints for clients:
  - POST /chat — forwards prompt to a healthy worker and returns the worker response.
  - GET /health — returns counts of total vs healthy workers.
  - GET /workers — returns health map of configured workers.
  - GET /errors — returns the error summary (in-memory).
- Worker discovery:
  - If `WORKER_SERVICE` is set (Swarm), router prefers `http://{WORKER_SERVICE}:{WORKER_PORT}`.
  - Otherwise router uses `WORKER_URLS` (comma-separated).
- Health check:
  - Router queries `{worker}/health` with a short timeout and considers worker healthy if HTTP 200.
- Retry logic:
  - Router attempts up to `MAX_RETRIES` different workers (random choice) before giving up.

Common router commands:
- Test router: curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Hello"}'
- Check workers: curl http://localhost:8000/workers
- Check errors: curl http://localhost:8000/errors

---

## Data flow — detailed with examples

1. Client sends:
   POST /chat -> {"prompt":"What are the main findings?"}
2. Router receives request:
   - Loads worker list (service or URLs)
   - Calls worker /health to filter healthy workers
   - Picks a healthy worker and forwards the prompt
3. Worker receives:
   - Encodes prompt via question encoder → q_vec (normalized)
   - If corpus not prepared, runs `prepare_corpus()` → loads PDFs → chunks → encodes context embeddings → saves to disk
   - Computes cosine similarity between q_vec and stored context embeddings
   - Selects top-k passages and returns JSON:
     {"results": [{"passage": "...", "score": 0.92}, ...]}
4. Router returns worker JSON to client unchanged.

---

## Files & Locations (where things live)

- Router:
  - router/app.py — routing logic, health checks, error endpoints
  - router/Dockerfile, router/requirements.txt
- Worker:
  - worker/app.py — DPR-like retrieval, PDF loading, persistence, /reload
  - worker/Dockerfile, worker/requirements.txt
  - pdf_corpus/ — place PDFs here (mounted or copied into container)
  - data/ — worker writes persistence files here (DATA_DIR)
- Utilities:
  - utils/logging_config.py — simple logger and error tracker

---

## Environment variables (relevant)

- WORKER_URLS — comma-separated worker addresses (router)
- WORKER_SERVICE — Swarm service name (router)
- HOST, PORT — service binding
- DPR_QUESTION_ENCODER — question encoder model id
- DPR_CONTEXT_ENCODER — context encoder model id
- PDF_DIR — directory for PDFs (worker)
- DATA_DIR — directory to persist corpus embeddings (worker)
- ENCODE_BATCH — batch size for encoding

---

## Practical tips & troubleshooting

- If "no corpus loaded" return → copy PDFs to pdf_corpus/ and call /reload.
- If timeouts happen increase REQUEST_TIMEOUT in config or env.
- If embeddings fail to save, ensure container has write permission to DATA_DIR.
- If models fail to download in Docker, ensure network access or build images on a machine that already has model caches.

---

## Offline finetuning (short note)

- DPR finetuning is done offline using the HuggingFace training loop (not supported via API).
- Typical steps:
  - Convert PDFs → (question, positive_passage, negative_passage) training examples.
  - Use HF Trainer with DPR question/context encoders.
  - Push fine-tuned weights and set DPR_QUESTION_ENCODER and DPR_CONTEXT_ENCODER to the new model paths.

---

## Example workflows

1. Local dev:
   - Place PDFs → `pdf_corpus/`
   - Start worker locally: python worker/app.py
   - POST /reload if needed; query /chat

2. Docker Compose:
   - Build images (make build), docker compose up
   - Ensure pdf_corpus is added into worker build context or mounted

3. Swarm:
   - Use `swarm-stack.yml`, ensure worker service mounts or copies PDFs to worker nodes
   - Prefer pushing images to a registry for production

---

## Quick reference commands

- Health:
  curl http://localhost:8000/health
- Chat:
  curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"prompt":"Explain X"}'
- Worker reload:
  curl -X POST http://worker:5000/reload
- Worker force reload:
  curl -X POST http://worker:5000/reload -H "Content-Type: application/json" -d '{"force":true}'

---

Thank you for using Zen. Keep things simple: encode once, persist, query, and scale workers. If you want, add a small UI to upload PDFs and call /reload automatically.
