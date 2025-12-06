# zen llm agent

distributed text generation across multiple workers.

## quick start

```bash
make build       # build images
make up          # start services
make test        # send test prompt
make logs        # watch logs
make down        # stop services
```

## local development

If you want to run and test locally (not in Docker), create a virtualenv and install requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r router/requirements.txt
pip install -r worker/requirements.txt
```

Then run the router and worker (in separate terminals):

```bash
python router/app.py
python worker/app.py
```

## what is it?

- **router** - forwards chat requests to healthy workers
- **workers** - run language models, generate text
- **simple** - minimal code, maximum clarity

## api

### chat (POST /chat)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello"}'
```

### health (GET /health)
```bash
curl http://localhost:8000/health
```

### workers (GET /workers)
```bash
curl http://localhost:8000/workers
```

### errors (GET /errors)
```bash
curl http://localhost:8000/errors
```

## configuration

create `.env` from `.env.example`:

```bash
cp .env.example .env
```

edit values as needed:
- `WORKER_URLS` - comma-separated worker addresses
- `MODEL_NAME` - huggingface model name
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR
- `REQUEST_TIMEOUT` - seconds to wait for response

## logs

logs are written to `logs/` directory:
- `router.log` - router activity
- `worker.log` - worker activity

view in real-time:
```bash
make logs         # all logs
make logs-router  # router only
make logs-worker  # workers only
```

## error handling

router automatically retries failed requests with other workers.

get error summary:
```bash
make errors
```

## develop

### requirements
- docker & docker-compose
- python 3.9+
- (optional) jq for pretty json output

### structure
```
.
├── config.py              # configuration
├── logging_config.py      # logging setup
├── docker-compose.yml     # services
├── Makefile               # commands
├── router/
│   ├── app.py            # request forwarding
│   ├── Dockerfile
│   └── requirements.txt
└── worker/
    ├── app.py            # text generation
    ├── Dockerfile
    └── requirements.txt
```

## tips

1. check health during development
   ```bash
   watch -n 1 'make health'
   ```

2. test specific prompts
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"prompt": "your prompt here"}'
   ```

3. monitor errors
   ```bash
   watch -n 5 'make errors'
   ```

## troubleshooting

**workers not responding?**
```bash
make logs-worker    # check worker logs
make health         # verify health check
```

**timeout errors?**
increase `REQUEST_TIMEOUT` in `.env` and restart

**out of memory?**
use a smaller model in `.env`:
```
MODEL_NAME=distilgpt2
```

## principles

- **zen** - minimal, clear, focused
- **simple** - easy to understand and modify
- **reliable** - graceful error handling
- **observable** - built-in logging and health checks

---

made with care. keep it simple.
