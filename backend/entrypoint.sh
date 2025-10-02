#!/bin/bash
set -e

# Start Ollama serve in the background
OLLAMA_HOST=0.0.0.0:11435 ollama serve &

# Wait a few seconds for Ollama to be ready
sleep 5

# Start FastAPI app with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
