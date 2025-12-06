# Deployment Checklist

Verify everything works before production deployment.

## Local Docker Compose Test

```bash
# 1. Build
make build

# Should show:
# [+] Building 2/2
# Successfully tagged llm-router:latest
# Successfully tagged llm-worker:latest

# 2. Start
make up

# Should show 3 containers running
docker compose ps
# STATUS: Up

# 3. Wait for health checks to pass
sleep 10
docker compose ps
# All should show "healthy"

# 4. Test router health
make health

# Expected response:
# {
#   "status": "ok",
#   "workers": 2,
#   "healthy": 2
# }

# 5. Test chat API
make test

# Expected response:
# {
#   "response": "...",
#   "model": "distilgpt2"
# }

# 6. Check for errors
make errors

# Should show minimal or empty errors

# 7. Stop
make down
```

## Docker Swarm Test

```bash
# 1. Initialize Swarm
docker swarm init

# 2. Build images
make swarm-build

# 3. Deploy stack
make swarm-deploy

# Should show:
# Creating service llm_agent_router
# Creating service llm_agent_llm_worker

# 4. Wait for services to start
sleep 10
docker stack ps llm_agent

# Should show:
# - 1 router task (Running)
# - 2 worker tasks (Running)

# 5. Test health
make swarm-health

# Expected response:
# {
#   "status": "ok",
#   "workers": 1,
#   "healthy": 1
# }

# Note: In Swarm, workers show as 1 service (llm_worker)

# 6. Test chat
make swarm-test

# Expected response:
# {
#   "response": "...",
#   "model": "distilgpt2"
# }

# 7. Check worker distribution
docker service ps llm_agent_llm_worker

# Should show 2 tasks (possibly on different nodes)

# 8. Scale workers
docker service scale llm_agent_llm_worker=3

# Verify
docker service ps llm_agent_llm_worker

# Should show 3 tasks

# 9. Clean up
make swarm-remove
docker swarm leave --force
```

## Service Connectivity Verification

```bash
# Check Docker Compose network
docker network inspect llm_network

# Should show all containers connected

# Check service DNS in container
docker exec router nslookup worker1

# Should resolve to worker1 IP

# Check inter-service connectivity
docker exec router curl -f http://worker1:5000/health

# Should return health status
```

## Swarm Service Discovery Test

```bash
# Check overlay network
docker network inspect llm_network

# Should show overlay driver and connected nodes

# Check service DNS from router
docker exec <ROUTER_TASK_ID> nslookup llm_worker

# Should resolve to load balancer IP (usually 10.0.9.x)

# Test router can reach workers
docker exec <ROUTER_TASK_ID> curl -f http://llm_worker:5000/health

# Should return health status
```

## Load Testing

```bash
# Send 10 concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test"}' &
done
wait

# All should succeed with 200 status

# Check error counts
make errors

# Should show no new errors
```

## Verification Checklist

- [ ] Local docker-compose builds without errors
- [ ] All containers start and show "healthy"
- [ ] Router responds to /health with 2 healthy workers
- [ ] Chat endpoint generates text
- [ ] Error tracking works
- [ ] Docker Swarm initializes
- [ ] Stack deploys without errors
- [ ] Router task runs on manager node
- [ ] Worker tasks spread across nodes
- [ ] Service DNS resolution works
- [ ] Router can reach workers via service name
- [ ] Chat works via Swarm
- [ ] Scaling workers works
- [ ] Load test succeeds

## Common Issues and Fixes

**Issue**: "no workers available"
- **Cause**: Workers still loading model
- **Fix**: Wait 30 seconds, retry

**Issue**: "timeout error"
- **Cause**: Worker is slow
- **Fix**: Increase REQUEST_TIMEOUT in config

**Issue**: Service DNS not resolving
- **Cause**: Overlay network not created
- **Fix**: Ensure swarm-stack.yml has network defined

**Issue**: Tasks won't start
- **Cause**: Image pull failed
- **Fix**: Check docker build output, rebuild images

**Issue**: Different response between docker-compose and swarm
- **Cause**: Environment variable differences
- **Fix**: Ensure both configs have same values
