from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import faiss
import numpy as np
import requests
from embedder import PDFEmbedder
import json
import os
app = FastAPI()

index = faiss.read_index("docs_index.faiss")

documents = {}
with open("docs_metadata.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        documents[str(i)] = line.strip()

OLLAMA_API_URL = "http://ollama:11434/api/generate"

class RAGRequest(BaseModel):
    prompt: str
    model: str = "tinyllama"
    max_tokens: int = 256

embedder = PDFEmbedder(folder_path="./documents")

HISTORY_PATH = "/app/rag_backend/history"
MAX_HISTORY_SIZE = 512 * 1024  # 512KB
os.makedirs(HISTORY_PATH, exist_ok=True)
class HistoryManager:
    def __init__(self, path=HISTORY_PATH, max_size=MAX_HISTORY_SIZE):
        self.path = path + "history.txt"
        self.max_size = max_size
        # Ensure file exists
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                pass

    def append(self, context):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(context + "\n")
        self.trim()

    def get(self):
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()

    def trim(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = f.read()
        if len(data.encode("utf-8")) > self.max_size:
            # Trim oldest lines
            lines = data.splitlines()
            while len("\n".join(lines).encode("utf-8")) > self.max_size:
                lines.pop(0)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

history_manager = HistoryManager()

@app.post("/rag")
def rag(req: RAGRequest):
    query_vec = embedder.get_embedding(req.prompt)
    query_vec_np = np.array(query_vec, dtype="float32").reshape(1, -1)
    k = 5
    distances, indices = index.search(query_vec_np, k)

    retrieved_docs = [documents.get(str(i), "") for i in indices[0]]
    context = "\n".join(retrieved_docs)
    history = history_manager.get()

    # Combine both retrieved docs and historical context
    full_prompt = (
        f"Context:\n{context}\n\n"
        f"History:\n{history}\n\n"
        f"Question:\n{req.prompt}"
    )

    # Save context to history
    history_manager.append(context)

    print("Full Prompt:", full_prompt)
    try:
        with requests.post(
            OLLAMA_API_URL,
            json={
                "model": req.model,
                "prompt": full_prompt,
                "max_tokens": req.max_tokens,
                "stream": True
            },
            stream=True,
        ) as response:
            response.raise_for_status()
            accumulated_response = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                if "response" in chunk:
                    accumulated_response.append(chunk["response"])
                if chunk.get("done", False):
                    break
            full_response = "".join(accumulated_response).strip()
            return {"answer": full_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {e}")

@app.post("/clear_history")
def clear_history():
    with open(history_manager.path, "w", encoding="utf-8") as f:
        f.write("")
    return {"status": "History cleared"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>RAG Query UI</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            textarea { width: 100%; height: 80px; }
            .answer { margin-top: 20px; padding: 10px; background: #f0f0f0; }
            button { margin-top: 10px; }
        </style>
    </head>
    <body>
        <h2>Ask a Question</h2>
        <form id="rag-form">
            <textarea name="prompt" id="prompt" placeholder="Enter your question..."></textarea><br>
            <button type="submit">Submit</button>
        </form>
        <button id="clear-btn">Clear History</button>
        <div class="answer" id="answer"></div>
        <script>
            document.getElementById('rag-form').onsubmit = async function(e) {
                e.preventDefault();
                const prompt = document.getElementById('prompt').value;
                document.getElementById('answer').innerText = "Loading...";
                const res = await fetch('/rag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: prompt })
                });
                const data = await res.json();
                document.getElementById('answer').innerText = data.answer || data.detail || "No answer";
            }
            document.getElementById('clear-btn').onclick = async function() {
                const res = await fetch('/clear_history', { method: 'POST' });
                const data = await res.json();
                document.getElementById('answer').innerText = data.status;
            }
        </script>
    </body>
    </html>
    """
