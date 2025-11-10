This project sets up a distributed **LLM chatbot cluster** across multiple Raspberry Pis.  
A central **router** node manages incoming chat requests and forwards them to **worker** nodes,  
each hosting a lightweight or quantized LLM for inference.

---

## 🧠 Architecture Overview



```
              ┌────────────────────────────┐
              │        Your Laptop         │
              │(SSH + Docker management)   │
              └────────────┬───────────────┘
                           │
                     Ethernet Cable
                           │
                   ┌───────┴────────┐
                   │   Ethernet     │
                   │     Switch     │
                   └──────┬─────────┘
        ┌─────────────────┼───────────────────────┐
        │                 │                       │
┌───────┴───────┐ ┌───────┴───────┐     ┌─────────┴─────┐
│ Raspberry Pi 1│ │ Raspberry Pi 2│ ... │ Raspberry Pi N│
│ Docker Node   │ │ Docker Node   │     │ Docker Node   │
└───────────────┘ └───────────────┘     └───────────────┘


```
## Build

```

- **Router**: Load balances and manages chat sessions.  
- **Workers**: Run small or quantized language models (e.g., DistilGPT2, TinyLLaMA).  
- **Network**: All Pis connected to a local Ethernet switch.  
- **Optionally**: You can add a lightweight web UI or memory module later.

---
```

## ⚙️ Quickstart (Local Test)

```bash
# Build Docker images
make build

# Start router + two workers locally
make up

# Send a test chat
./scripts/send_prompt.sh "Hello, how are you?"


```
🐍 Deploying to Raspberry Pi Swarm


### On Pi manager node
docker swarm init

### On each worker Pi
docker swarm join --token <TOKEN> <MANAGER_IP>:2377

### Deploy the stack
make swarm-deploy
