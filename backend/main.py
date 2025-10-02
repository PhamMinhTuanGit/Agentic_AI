from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import json
import numpy as np
import requests
from embedder import embed_text

app = FastAPI()

# Load FAISS index
index = faiss.read_index("docs_index.faiss")

# Load documents metadata (one doc per line)
documents = {}
with open("docs_metadata.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        documents[str(i)] = line.strip()

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
        response = requests.post("http://localhost:11435/api/generate", json={
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
