# Docker Swarm Deployment

Run the LLM agent across multiple machines using Docker Swarm.

## When to Use Swarm

Use Docker Swarm when you want to:
- Run on **multiple machines**
- **Scale** workers automatically
- **Distribute** load across a cluster
- Have **high availability**

For single-machine testing, use `docker-compose` instead (see readme.md).

## Quick Start (Single Machine)

Test Swarm on your local machine first:

```bash
# Initialize Swarm
docker swarm init

# Build images
docker build -t llm-router:latest ./router
docker build -t llm-worker:latest ./worker

# Deploy
docker stack deploy -c swarm-stack.yml llm_agent

# Check status
docker stack ps llm_agent

# Test
curl http://localhost:8000/health
```

Stop with:
```bash
docker stack rm llm_agent
```

## Multi-Machine Setup

### Prerequisites

- Docker installed on all machines
- All machines can reach each other (same network)
- Manager machine has IP address (e.g., `111.111.1.11`)

### Step 1: Initialize Manager

On the **manager machine** only:

```bash
docker swarm init --advertise-addr 111.111.1.11
```

Replace `111.111.1.11` with your manager's IP address.

Copy the output. You'll see:

```
Swarm initialized: current node is now the manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-0g... 111.111.1.11:2377

To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
```

### Step 2: Join Worker Nodes

On each worker node, run the join command from the manager's output:

```bash
docker swarm join --token SWMTKN-1-0g... 111.111.1.11:2377
```

### Step 3: Verify Cluster

Back on the manager node, check the cluster status:

```bash
docker node ls
```

## Deployment

### Stack Configuration

Sample `swarm-stack.yml`:

```yaml
version: '3'

services:
  router:
    image: llm-router:latest
    ports:
      - "8000:8000"
  worker:
    image: llm-worker:latest
```

### Deploying the Stack

From the directory with `swarm-stack.yml`:

```bash
docker stack deploy -c swarm-stack.yml llm_agent
```

## Testing

### Health Check

After deployment, check if the services are running:

```bash
docker service ls
```

### Accessing the Router

Open a browser or use `curl`:

```bash
curl http://<MANAGER_IP>:8000/health
```

## Monitoring

### Docker Stats

To monitor resource usage:

```bash
docker stats
```

### Logging

For real-time logs:

```bash
docker service logs -f llm_agent_router
```

## Scaling

### Adjusting Service Replicas

To scale the number of router replicas:

```bash
docker service scale llm_agent_router=<NUMBER>
```

### Resizing the Cluster

Add or remove nodes as needed, then adjust the service replicas accordingly.

## Troubleshooting

### Common Issues

- **Swarm not initializing**: Ensure Docker is running and you have the right permissions.
- **Nodes not joining**: Check network connectivity and firewall settings.

### Logs and Diagnostics

Use Docker's built-in logging and monitoring tools to diagnose issues. Check individual container logs with:

```bash
docker logs <CONTAINER_ID>
```

## Advanced

### Overlay Networks

For multi-host communication, configure overlay networks in your `docker-compose` file.

### Secrets and Configs

Use Docker secrets and configs for sensitive data and configuration files.

### Health Checks

Define health checks in your `docker-compose` to ensure services are running as expected.

### Resource Limits

Set CPU and memory limits for services in the `docker-compose` file to prevent resource hogging.

### Logging Drivers

Configure logging drivers for centralized logging solutions.

### Backup and Restore

Regularly backup your Docker volumes and Swarm configurations. Use `docker stack rm` to remove stacks and `docker swarm leave --force` to remove nodes from the swarm.