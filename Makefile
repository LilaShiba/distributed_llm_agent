.PHONY: build up down logs logs-router logs-worker health test errors clean help swarm-build swarm-deploy swarm-remove

# === Local Docker Compose ===
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

logs-router:
	docker compose logs -f router

logs-worker:
	docker compose logs -f worker1 worker2

health:
	curl -s http://localhost:8000/health | jq .

test:
	curl -s -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello"}' | jq .

errors:
	curl -s http://localhost:8000/errors | jq .

# === Docker Swarm ===
swarm-build:
	docker build -t llm-router:latest ./router
	docker build -t llm-worker:latest ./worker

swarm-deploy: swarm-build
	docker stack deploy -c swarm-stack.yml llm_agent
	@echo "Stack deployed. Checking status..."
	sleep 3
	docker stack ps llm_agent

swarm-remove:
	docker stack rm llm_agent

swarm-logs-router:
	docker service logs -f llm_agent_router

swarm-logs-worker:
	docker service logs -f llm_agent_llm_worker

swarm-health:
	curl -s http://localhost:8000/health | jq .

swarm-test:
	curl -s -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello"}' | jq .

swarm-workers:
	docker service ps llm_agent_llm_worker

# === Cleanup ===
clean:
	docker system prune -af
	rm -rf logs/*.log*

help:
	@echo "Zen LLM Agent - Commands"
	@echo ""
	@echo "Local Docker Compose:"
	@echo "  make build         - Build Docker images"
	@echo "  make up            - Start services"
	@echo "  make down          - Stop services"
	@echo "  make logs          - View all logs"
	@echo "  make health        - Check router health"
	@echo "  make test          - Send test prompt"
	@echo "  make errors        - View error summary"
	@echo ""
	@echo "Docker Swarm:"
	@echo "  make swarm-deploy  - Deploy to Swarm"
	@echo "  make swarm-remove  - Remove from Swarm"
	@echo "  make swarm-health  - Check health"
	@echo "  make swarm-test    - Send test prompt"
	@echo "  make swarm-workers - List worker tasks"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Clean up Docker"
	@echo "  make help          - Show this help"
