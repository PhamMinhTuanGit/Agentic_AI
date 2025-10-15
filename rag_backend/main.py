from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import faiss
import numpy as np
import requests
from embedder import PDFEmbedder
import json
import os
from typing import List, Tuple
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
    use_reranker: bool = True
    top_k: int = 10
    rerank_top_k: int = 5

embedder = PDFEmbedder(folder_path="./documents")

# Query condensation using LLM
def condense_question(question: str, history: str, model: str = "tinyllama") -> str:
    """Use LLM to condense/improve the question based on conversation history"""
    if not history.strip():
        return question
    
    condensation_prompt = f"""Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all relevant context.

Conversation History:
{history[-1000:]}  # Use last 1000 chars to avoid token limits

Follow-up Question: {question}

Standalone Question:"""
    
    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": condensation_prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        condensed = response.json().get("response", question).strip()
        print(f"Condensed question: {condensed}")
        return condensed
    except Exception as e:
        print(f"Question condensation failed: {e}, using original question")
        return question

# Reranker using cross-encoder scoring
def rerank_documents(query: str, docs: List[Tuple[int, str, float]], top_k: int = 5, model: str = "tinyllama") -> List[Tuple[int, str, float]]:
    """Rerank documents using LLM-based scoring for better relevance"""
    if len(docs) <= top_k:
        return docs
    
    reranked = []
    for idx, doc, distance in docs:
        # Score each document with a simple relevance prompt
        score_prompt = f"""Rate the relevance of this document to the query on a scale of 0-10, where 10 is highly relevant and 0 is not relevant at all. Only respond with a number.

Query: {query}

Document: {doc[:500]}

Relevance Score:"""
        
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": score_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            score_text = response.json().get("response", "5").strip()
            # Extract number from response
            score = float(''.join(filter(str.isdigit, score_text[:2])) or 5)
        except Exception as e:
            print(f"Reranking failed for doc {idx}: {e}")
            score = 5.0  # Default middle score
        
        reranked.append((idx, doc, score))
    
    # Sort by score descending
    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked[:top_k]

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
    # Step 1: Get historical context from cache
    history = history_manager.get()
    
    # Step 2: Condense/improve the question using LLM and historical context
    condensed_query = condense_question(req.prompt, history, req.model)
    
    # Step 3: Embed the condensed query
    query_vec = embedder.get_embedding(condensed_query)
    query_vec_np = np.array(query_vec, dtype="float32").reshape(1, -1)
    
    # Step 4: Retrieve top-k documents from database
    k = req.top_k
    distances, indices = index.search(query_vec_np, k)
    
    # Step 5: Prepare documents with metadata for reranking
    retrieved_docs = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        doc_text = documents.get(str(idx), "")
        retrieved_docs.append((int(idx), doc_text, float(distance)))
    
    # Step 6: Rerank documents if enabled
    if req.use_reranker and len(retrieved_docs) > req.rerank_top_k:
        print(f"Reranking {len(retrieved_docs)} documents to top {req.rerank_top_k}...")
        final_docs = rerank_documents(condensed_query, retrieved_docs, req.rerank_top_k, req.model)
    else:
        final_docs = retrieved_docs[:req.rerank_top_k]
    
    # Step 7: Build context from reranked documents
    context = "\n\n".join([doc[1] for doc in final_docs])
    
    # Step 8: Build full prompt with context and history
    full_prompt = f"""Based on the following context and conversation history, answer the question.

Context:
{context}

Conversation History:
{history[-2000:] if history else "No previous conversation"}

Question: {req.prompt}

Answer:"""

    # Step 9: Save current interaction to history cache
    interaction = f"Q: {req.prompt}\nContext used: {context[:200]}..."
    history_manager.append(interaction)

    print("Full Prompt:", full_prompt[:500], "...")
    
    # Step 10: Generate response using LLM
    try:
        with requests.post(
            OLLAMA_API_URL,
            json={
                "model": req.model,
                "prompt": full_prompt,
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
            
            # Save response to history
            history_manager.append(f"A: {full_response}\n---")
            
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
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px;
                max-width: 900px;
            }
            textarea { 
                width: 100%; 
                height: 100px;
                padding: 10px;
                font-size: 14px;
            }
            .answer { 
                margin-top: 20px; 
                padding: 15px; 
                background: #f0f0f0;
                border-radius: 5px;
                white-space: pre-wrap;
            }
            button { 
                margin-top: 10px;
                padding: 10px 20px;
                font-size: 14px;
                cursor: pointer;
            }
            .submit-btn {
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            .clear-btn {
                background: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
            }
            .settings {
                margin: 20px 0;
                padding: 15px;
                background: #e8f4f8;
                border-radius: 5px;
            }
            .settings label {
                display: block;
                margin: 10px 0;
            }
            .settings input[type="number"] {
                width: 80px;
                padding: 5px;
            }
            h2 {
                color: #333;
            }
        </style>
    </head>
    <body>
        <h2>🤖 RAG Query Interface</h2>
        
        <div class="settings">
            <h3>Settings</h3>
            <label>
                <input type="checkbox" id="use_reranker" checked> Use Reranker (improves result quality)
            </label>
            <label>
                Top-K Documents: <input type="number" id="top_k" value="10" min="1" max="50">
            </label>
            <label>
                Rerank Top-K: <input type="number" id="rerank_top_k" value="5" min="1" max="20">
            </label>
            <label>
                Model: 
                <select id="model">
                    <option value="tinyllama">TinyLlama (Fast)</option>
                    <option value="llama3.1:8b">Llama 3.1 8B (Better)</option>
                    <option value="mistral">Mistral</option>
                </select>
            </label>
        </div>
        
        <form id="rag-form">
            <textarea name="prompt" id="prompt" placeholder="Enter your question..."></textarea><br>
            <button type="submit" class="submit-btn">Submit Query</button>
        </form>
        <button id="clear-btn" class="clear-btn">Clear History</button>
        
        <div class="answer" id="answer">Your answer will appear here...</div>
        
        <script>
            document.getElementById('rag-form').onsubmit = async function(e) {
                e.preventDefault();
                const prompt = document.getElementById('prompt').value;
                const use_reranker = document.getElementById('use_reranker').checked;
                const top_k = parseInt(document.getElementById('top_k').value);
                const rerank_top_k = parseInt(document.getElementById('rerank_top_k').value);
                const model = document.getElementById('model').value;
                
                document.getElementById('answer').innerText = "🔄 Processing your query...";
                
                try {
                    const res = await fetch('/rag', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            prompt: prompt,
                            use_reranker: use_reranker,
                            top_k: top_k,
                            rerank_top_k: rerank_top_k,
                            model: model
                        })
                    });
                    const data = await res.json();
                    document.getElementById('answer').innerText = data.answer || data.detail || "No answer";
                } catch (error) {
                    document.getElementById('answer').innerText = "❌ Error: " + error.message;
                }
            }
            
            document.getElementById('clear-btn').onclick = async function() {
                if (!confirm('Are you sure you want to clear the conversation history?')) {
                    return;
                }
                try {
                    const res = await fetch('/clear_history', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('answer').innerText = "✅ " + data.status;
                } catch (error) {
                    document.getElementById('answer').innerText = "❌ Error clearing history: " + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
