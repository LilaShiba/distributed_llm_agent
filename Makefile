.PHONY: build up down logs health test clean help

build:
	docker build -t llm-router:latest ./router
	docker build -t llm-worker:latest ./worker

up:
	docker compose up -d --remove-orphans

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

clean:
	docker system prune -af
	rm -rf logs/*.log*

# Docker Swarm convenience targets
swarm-deploy:
	@echo "Building router and worker images for swarm..."
	docker build -t distributed_llm_agent_router:latest ./router
	docker build -t distributed_llm_agent_worker:latest ./worker
	@echo "Deploying stack to swarm..."
	docker stack deploy -c swarm-stack.yml llm_stack

swarm-remove:
	@echo "Removing stack from swarm..."
	docker stack rm llm_stack || true
	@echo "Done"

help:
	@echo "zen llm agent"
	@echo ""
	@echo "make build        build docker images"
	@echo "make up           start services"
	@echo "make down         stop services"
	@echo "make logs         watch all logs"
	@echo "make health       check router health"
	@echo "make test         send test prompt"
	@echo "make errors       show error summary"
	@echo "make clean        cleanup docker and logs"
