#!/bin/bash
# send_prompt.sh
# Sends a prompt to the router API and prints the response

# Usage:
# ./send_prompt.sh "Hello world!"

# Check if argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 \"Your prompt here\""
  exit 1
fi

PROMPT="$1"

# Router URL (default localhost:8000, can be changed via env var)
ROUTER_URL="${ROUTER_URL:-http://localhost:8000/chat}"

# Send the prompt to the router using curl
RESPONSE=$(curl -s -X POST "$ROUTER_URL" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\"}")

# Pretty print if jq is available, otherwise raw output
if command -v jq &> /dev/null; then
  echo "$RESPONSE" | jq
else
  echo "$RESPONSE"
fi
