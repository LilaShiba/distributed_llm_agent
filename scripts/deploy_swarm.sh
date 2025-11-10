#!/bin/bash
# deploy_swarm.sh
# Builds and pushes Docker images, then deploys the swarm stack

# Usage:
# ./deploy_swarm.sh [registry]
# If no registry provided, defaults to 'local'

REGISTRY=${1:-local}

echo "Using registry: $REGISTRY"

# Build images
echo "Building router image..."
docker build -t $REGISTRY/rpi-llm-router:latest ../router
echo "Building worker image..."
docker build -t $REGISTRY/rpi-llm-worker:latest ../worker

# Push images to registry
echo "Pushing router image..."
docker push $REGISTRY/rpi-llm-router:latest
echo "Pushing worker image..."
docker push $REGISTRY/rpi-llm-worker:latest

# Deploy the swarm stack
echo "Deploying swarm stack..."
docker stack deploy -c ../swarm-stack.yml llm_hub

echo "Deployment complete!"
