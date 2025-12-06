.PHONY: build up down logs logs-router logs-worker health test errors clean help swarm-build swarm-deploy swarm-remove

# === Local Docker Compose ===
build:
	docker build -t llm-router:latest ./router
	docker build -t llm-worker:latest ./worker

up:
	docker compose -f routes/docker-compose.yml up -d

down:
	docker compose -f routes/docker-compose.yml down

logs:
	docker compose -f routes/docker-compose.yml logs -f

logs-router:
	docker compose -f routes/docker-compose.yml logs -f router

logs-worker:
	docker compose -f routes/docker-compose.yml logs -f worker1 worker2

health:
	curl -s http://localhost:8000/health | jq .

test:
	curl -s -X POST http://localhost:8000/chat \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Hello"}' | jq .

errors:
	curl -s http://localhost:8000/errors | jq .

clean:
	docker system prune -af
	rm -rf logs/*.log* data/*

# === Docker Swarm ===
swarm-build:
	@echo "Building router and worker images for swarm..."
	docker build -t distributed_llm_agent_router:latest ./router
	docker build -t distributed_llm_agent_worker:latest ./worker
	@echo "Done"

swarm-deploy: swarm-build
	@echo "Deploying stack to swarm..."
	docker stack deploy -c routes/swarm-stack.yml llm_agent
	@echo "Done"

swarm-remove:
	@echo "Removing stack from swarm..."
	docker stack rm llm_agent || true
	@echo "Done"

help:
	@echo "zen llm agent"
	@echo ""
	@echo "make build        build docker images"
	@echo "make up           start services (routes/docker-compose.yml)"
	@echo "make down         stop services"
	@echo "make logs         watch all logs"
	@echo "make logs-router  router logs only"
	@echo "make logs-worker  worker logs only"
	@echo "make health       check router health"
	@echo "make test         send test prompt"
	@echo "make errors       show error summary"
	@echo "make clean        cleanup docker/logs/data"
	@echo ""
	@echo "make swarm-build  build images for swarm"
	@echo "make swarm-deploy deploy to swarm (routes/swarm-stack.yml)"
	@echo "make swarm-remove remove swarm stack"
