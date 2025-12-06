# Zen LLM Agent

A lightweight, distributed text generation system. Send prompts to a router, which forwards them to healthy workers running language models.

## What You Get

✅ **Distributed** - Multiple workers share the load  
✅ **Reliable** - Automatic failover and retries  
✅ **Observable** - Built-in logging and error tracking  
✅ **Simple** - ~500 lines of clean code  
✅ **Fast** - Ready in seconds, inference in 1-5 seconds  

## 30-Second Start

```bash
# 1. Build
make build

# 2. Run
make up

# 3. Test
make test

# Done! Open another terminal to check logs
make logs
```

Stop with:
```bash
make down
```

## How to Use

### Send a Message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!"}'
```

Response:
```json
{
  "response": "Hello, world! How can I help you today?",
  "model": "distilgpt2"
}
```

### Check Status

```bash
# Router health
curl http://localhost:8000/health

# Worker status
curl http://localhost:8000/workers

# Error count
curl http://localhost:8000/errors
```

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key settings:
- `MODEL_NAME` - Which model to use (default: `distilgpt2`)
- `REQUEST_TIMEOUT` - Seconds to wait for response (default: 30)
- `LOG_LEVEL` - How much to log (default: INFO)

## What's Running?

- **Router** (port 8000): Receives requests, picks a worker
- **Worker 1** (port 5000): Runs the language model
- **Worker 2** (port 5000): Backup worker for redundancy

All three run in Docker containers.

## What Files Do What?

| File | Purpose |
|------|---------|
| `router/app.py` | Request routing (85 lines) |
| `worker/app.py` | Text generation (69 lines) |
| `utils/logging_config.py` | Logging setup (52 lines) |
| `config.py` | Configuration (23 lines) |
| `docker-compose.yml` | Local deployment |
| `swarm-stack.yml` | Distributed deployment |

## Common Commands

```bash
make build          # Build Docker images
make up             # Start services
make down           # Stop services
make logs           # Watch all logs
make logs-router    # Router logs only
make logs-worker    # Worker logs only
make health         # Check status
make test           # Send test message
make errors         # Show errors
make clean          # Clean up
```

## Troubleshooting

### "No workers available"

Workers are starting. Wait 30 seconds, then try again.

```bash
make logs-worker    # Check worker logs
make health         # Check status
```

### "Timeout error"

Response took too long. Increase timeout in `.env`:
