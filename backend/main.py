from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import json
import numpy as np
import requests
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from embedder import embed_text

# Add project root to path for network_stat imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from network_stat.network_rag import NetworkTopologyRAG, NetworkConfigRequest

# Load environment variables
load_dotenv()

app = FastAPI()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")

# Load FAISS index
index = faiss.read_index("docs_index.faiss")

# Load documents metadata (one doc per line)
documents = {}
with open("docs_metadata.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        documents[str(i)] = line.strip()

# Initialize Network Topology RAG
try:
    network_rag = NetworkTopologyRAG("network_stat/topo.yaml")
except Exception as e:
    print(f"Warning: Could not initialize Network RAG: {e}")
    network_rag = None

class RAGRequest(BaseModel):
    prompt: str
    model: str = "llama3.1:8b"
    max_tokens: int = 256
    continuation_token: str = None

@app.post("/rag")
def rag(req: RAGRequest):
    # Use continuation token as prompt if exists
    prompt_text = req.continuation_token if req.continuation_token else req.prompt

    # Embed query and search FAISS
    query_vec = embed_text(prompt_text).reshape(1, -1)
    k = 5  # number of docs to retrieve
    distances, indices = index.search(query_vec, k)

    retrieved_docs = [documents.get(str(i), "") for i in indices[0]]
    context = "\n".join(retrieved_docs)

    # Compose prompt with context and user prompt
    full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt_text}"

    # Call Ollama API running locally
    try:
        response = requests.post(OLLAMA_API_URL, json={
            "model": req.model,
            "prompt": full_prompt,
            "max_tokens": req.max_tokens
        })
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {e}")

    generated_text = result.get("response", "")

    # Simple continuation heuristic
    needs_continue = len(generated_text) >= req.max_tokens - 50

    return {
        "text": generated_text,
        "continue": needs_continue,
        "continuation_token": generated_text if needs_continue else None
    }


# ==================== NETWORK TOPOLOGY ENDPOINTS ====================

@app.get("/network/topology")
def get_topology_info():
    """Get network topology information"""
    if not network_rag:
        raise HTTPException(status_code=503, detail="Network RAG not initialized")
    
    return {
        "topology_summary": network_rag.get_topology_summary(),
        "all_devices": {
            d_id: d.to_dict() 
            for d_id, d in network_rag.parser.get_all_devices().items()
        }
    }


@app.get("/network/device/{device_id}")
def get_device_config(device_id: str):
    """Get configuration for specific device"""
    if not network_rag:
        raise HTTPException(status_code=503, detail="Network RAG not initialized")
    
    try:
        device_info = network_rag.get_device_info(device_id)
        if "error" in device_info:
            raise HTTPException(status_code=404, detail=device_info["error"])
        
        return {
            "device_info": device_info,
            "cli_commands": network_rag.get_device_config(device_id)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/network/devices")
def get_all_devices(device_type: str = None):
    """Get all devices, optionally filtered by type"""
    if not network_rag:
        raise HTTPException(status_code=503, detail="Network RAG not initialized")
    
    try:
        if device_type:
            devices = network_rag.get_devices_by_type(device_type)
            return {
                "type": device_type,
                "count": len(devices),
                "devices": devices
            }
        else:
            all_devices = network_rag.parser.get_all_devices()
            return {
                "total_count": len(all_devices),
                "devices": {d_id: d.to_dict() for d_id, d in all_devices.items()}
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/network/configure")
def configure_device(request: NetworkConfigRequest):
    """Configure device with optional LLM assistance"""
    if not network_rag:
        raise HTTPException(status_code=503, detail="Network RAG not initialized")
    
    try:
        # Get configuration
        config_result = network_rag.process_configuration_request(request)
        
        # Generate LLM prompt if needed
        llm_prompt = network_rag.generate_llm_prompt(request)
        
        return {
            "configuration": config_result,
            "llm_prompt_for_assistance": llm_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NetworkQuery(BaseModel):
    """Natural language query about network"""
    query: str
    model: str = "llama3.1:8b"
    max_tokens: int = 512


@app.post("/network/query")
def query_network(req: NetworkQuery):
    """Query about network topology using LLM with context"""
    if not network_rag:
        raise HTTPException(status_code=503, detail="Network RAG not initialized")
    
    try:
        # Get topology context
        topology_context = network_rag.get_llm_context()
        
        # Build prompt
        full_prompt = f"""
You are a network configuration expert. Use the following network topology information to answer the question.

NETWORK TOPOLOGY INFORMATION:
{topology_context}

QUESTION: {req.query}

ANSWER:"""
        
        # Call Ollama API
        response = requests.post(OLLAMA_API_URL, json={
            "model": req.model,
            "prompt": full_prompt,
            "stream": False,
            "max_tokens": req.max_tokens
        })
        response.raise_for_status()
        result = response.json()
        
        return {
            "question": req.query,
            "answer": result.get("response", ""),
            "topology_context_used": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/network/context")
def get_network_context():
    """Get full topology context for embedding/retrieval"""
    if not network_rag:
        raise HTTPException(status_code=503, detail="Network RAG not initialized")
    
    return {
        "context": network_rag.get_llm_context(),
        "devices_count": len(network_rag.parser.get_all_devices()),
        "switches": len(network_rag.parser.get_devices_by_type('switch')),
        "routers": len(network_rag.parser.get_devices_by_type('router')),
        "hosts": len(network_rag.parser.get_devices_by_type('host'))
    }
