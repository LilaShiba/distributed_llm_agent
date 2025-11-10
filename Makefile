# Makefile for Raspberry Pi LLM Chatbot Hub
# Provides convenient shortcuts for building, testing, and deploying.

# Variables
PROJECT_NAME = llm_hub
COMPOSE_FILE = docker-compose.yml
STACK_FILE = swarm-stack.yml

# Default registry (optional, can be overridden on command line)
REGISTRY ?= local

# 🧱 Build all Docker images (router + worker)
build:
	docker build -t $(REGISTRY)/rpi-llm-router:latest ./router
	docker build -t $(REGISTRY)/rpi-llm-worker:latest ./worker

# 🧱 Run locally using Docker Compose
up:
	docker compose -f $(COMPOSE_FILE) up -d --remove-orphans

# 🧱 Stop and remove all containers
down:
	docker compose -f $(COMPOSE_FILE) down

# 🧱 View logs from all services
logs:
	docker compose -f $(COMPOSE_FILE) logs -f

# 🧱 Deploy to Docker Swarm (Pi cluster)
swarm-deploy:
	docker stack deploy -c $(STACK_FILE) $(PROJECT_NAME)

# 🧱 Remove Swarm stack
swarm-remove:
	docker stack rm $(PROJECT_NAME)

# 🧱 List running containers and nodes
status:
	docker ps
	docker node ls || true

# 🧱 Clean up dangling images and containers
clean:
	docker system prune -af
