# 🎬 Getting Started with RAG Pipeline UI

Quick setup guide to get the Streamlit UI running in 5 minutes.

## ⚡ Quick Start (5 minutes)

### Step 1: Ensure Ollama is Running
```bash
# Terminal 1
ollama serve
```

### Step 2: Install UI Dependencies
```bash
# Terminal 2
cd Agentic
pip install -r ui/requirements.txt
```

### Step 3: Run the UI
```bash
# In the same terminal
streamlit run ui/app.py
```

The UI will open automatically at **http://localhost:8501**

---

## 🎯 First Steps

1. **Wait for Pipeline Initialization**
   - On first run, it initializes the RAG pipeline
   - Takes ~30-60 seconds
   - You'll see: "🚀 Initializing RAG Pipeline..."

2. **Enter a Question**
   - Type in the input field: "What is BGP protocol?"
   - Click the 🔍 **Submit** button

3. **View Results**
   - See the generated answer (green box)
   - Check cache status (✅ HIT or ❌ MISS)
   - View performance metrics
   - Explore retrieved documents

4. **Explore Sidebar**
   - Adjust Retriever/Reranker Top-K
   - Change LLM temperature
   - View pipeline statistics

5. **Check Analytics Page**
   - Click 📊 **Analytics** in sidebar
   - See performance charts and metrics

---

## 🛠️ Alternative Startup Methods

### Method 1: Using Python Script (Recommended)
```bash
python ui/start.py
```

Checks dependencies and starts with validation.

### Method 2: Using Bash Script
```bash
chmod +x run_ui.sh
./run_ui.sh
```

### Method 3: With Custom Port
```bash
streamlit run ui/app.py --server.port 8080
```

### Method 4: With Debug Logging
```bash
streamlit run ui/app.py --logger.level=debug
```

---

## 📋 Prerequisites Checklist

- [ ] Python 3.8+ installed
- [ ] Ollama installed and running
- [ ] Required models pulled:
  - [ ] `nomic-embed-text`
  - [ ] `qwen2.5-coder:3b`
- [ ] FAISS index built: `database/document/hybrid_docs_index.faiss`
- [ ] Dependencies installed: `pip install -r ui/requirements.txt`

### Check Prerequisites

```bash
# Check Python version
python --version  # Should be 3.8+

# Check Ollama running
curl http://localhost:11434/api/tags  # Should return list of models

# Check models installed
ollama list  # Should show both models

# Check FAISS index
ls -lh database/document/hybrid_docs_index.faiss  # Should exist
```

---

## 🚀 Full Setup from Scratch

If this is your first time:

```bash
# 1. Navigate to project
cd /path/to/Agentic

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# 3. Install main dependencies
pip install -r requirements.txt

# 4. Install UI dependencies
pip install -r ui/requirements.txt

# 5. Start Ollama (in separate terminal)
ollama serve

# 6. Pull models (in another terminal)
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b

# 7. Build FAISS index (if not done)
python -m ingest.embedder

# 8. Run the UI
streamlit run ui/app.py
```

---

## 📊 What You'll See

### First Time Launch
```
INFO:rag.pipeline:🚀 Initializing RAG Pipeline
INFO:rag.pipeline:======================================================================
[1/4] 🔍 Initializing Retriever...
✅ Loaded FAISS index from database/document/hybrid_docs_index.faiss
✅ Loaded metadata with 2213 text chunks
[2/4] 🤖 Initializing Reranker...
✅ Initialized LLM Reranker with model: qwen2.5-coder:3b
[3/4] 💬 Initializing LLM Client...
✅ Initialized LLM Client
[4/4] 💾 Initializing Cache...
✅ Initialized Cache Manager
✅ Pipeline initialized successfully!
```

### Main Interface
```
┌─────────────────────────────────────────────┐
│  🚀 RAG Pipeline Interactive UI             │
│                                             │
│  Ask questions about your documents         │
│                                             │
│  [Enter your question...] [🔍 Submit]       │
└─────────────────────────────────────────────┘

Sidebar:
├─ ⚙️ Configuration
├─ 📊 Statistics  
├─ 📜 Query History
├─ 🗑️ Clear Cache
└─ 📋 Clear History
```

### Result Display
```
💬 ANSWER
BGP is a routing protocol used for inter-domain routing...

✅ Cache HIT (or ❌ Cache MISS)

⏱️ PERFORMANCE METRICS
├─ Total Time: 0.01s (cached) or 11.32s (full)
├─ Retrieval: 0.32s
├─ Reranking: 2.95s
└─ Generation: 8.05s

📄 RETRIEVED & RERANKED DOCUMENTS
Document 1: BGP Overview (Score: 0.892)
Document 2: BGP Configuration (Score: 0.856)
...
```

---

## ⚙️ Configuration Tips

### For First-Time Users
```python
# Default settings (good balance)
Retriever Top-K: 10
Reranker Top-K: 5
Temperature: 0.7
Cache TTL: 86400s (24h)
Enable Cache: ✓
```

### For Speed (Chat Application)
```python
Retriever Top-K: 10
Reranker Top-K: 3
Temperature: 0.9
Cache TTL: 86400s
Enable Cache: ✓
# Result: ~7s full pipeline
```

### For Quality (Research/Reports)
```python
Retriever Top-K: 20
Reranker Top-K: 8
Temperature: 0.3
Cache TTL: 86400s
Enable Cache: ✓
# Result: Better answers (~18s)
```

---

## 🐛 Common Issues

### Issue: "Connection refused" Error
```
❌ Error: Failed to connect to Ollama at http://localhost:11434

Solution:
1. Make sure Ollama is running: ollama serve
2. Check if it's accessible: curl http://localhost:11434/api/tags
3. If using different port, update LLM_API_URL env var
```

### Issue: "No documents retrieved"
```
❌ No documents found

Solution:
1. Check FAISS index exists: ls database/document/hybrid_docs_index.faiss
2. If missing, rebuild: python -m ingest.embedder
3. Check chunk count in metadata
```

### Issue: Slow First Response
```
⏳ Takes 30+ seconds first time

Reason: Pipeline initialization includes:
- Loading FAISS index
- Loading models
- Initializing LLM client
- Building cache

Solution: This is normal on first run. Subsequent queries are faster.
```

### Issue: Memory/RAM Issues
```
💾 Running out of memory

Solutions:
1. Clear cache: 🗑️ Clear All Cache button
2. Reduce Reranker Top-K: 5 → 3
3. Restart Ollama or browser
4. Monitor: top or Activity Monitor
```

### Issue: Poor Answer Quality
```
❌ Answers don't match documents

Solutions:
1. Check retrieved documents (expand "All Retrieved Docs")
2. Increase Retriever Top-K: 10 → 20
3. Lower Temperature: 0.7 → 0.3
4. Regenerate embeddings: python -m ingest.embedder
```

---

## 🔍 Debugging

### Enable Debug Logging
```bash
streamlit run ui/app.py --logger.level=debug
```

### Check Debug Console in UI
1. Scroll down to "Debug Information"
2. Expand each section:
   - Raw Result JSON
   - Query History
   - Logs
   - System Info

### Check System Logs
```bash
# View tail of debug log
tail -f /tmp/streamlit/logs.txt

# Or check stdout in terminal
# Streamlit logs appear in the terminal where you ran the command
```

---

## 📦 Environment Variables (Optional)

Create `.env` file in project root:

```bash
# LLM & Embedding APIs
LLM_API_URL=http://localhost:11434/api/generate
EMBEDDING_API_URL=http://localhost:11434/api/embeddings

# Models
RERANK_MODEL=qwen2.5-coder:3b
EMBEDDING_MODEL=nomic-embed-text

# Paths
FAISS_INDEX_PATH=database/document/hybrid_docs_index.faiss
METADATA_PATH=database/document/hybrid_docs_metadata.json
CACHE_DB_PATH=cache/rag_cache.db

# Parameters
CACHE_TTL=86400
RETRIEVER_TOP_K=10
RERANKER_TOP_K=5
```

---

## 🎨 UI Customization

### Change Sidebar Width
Edit `ui/app.py`, modify:
```python
st.set_page_config(
    layout="wide"  # or "centered"
)
```

### Change Theme Colors
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B35"        # Main color
backgroundColor = "#FFFFFF"     # Page background
secondaryBackgroundColor = "#F0F2F6"  # Sidebar/Cards
textColor = "#262730"           # Text color
```

### Add Custom Logo
Modify in `app.py`:
```python
st.sidebar.image("path/to/logo.png", width=250)
```

---

## 📱 Mobile Access

Access from other devices on your network:

```bash
# Find your local IP
ifconfig | grep "inet "  # macOS/Linux
ipconfig                  # Windows

# Run with that IP
streamlit run ui/app.py --server.address 192.168.1.100

# Access from mobile/other device
# http://192.168.1.100:8501
```

---

## 🚢 Deployment

### Streamlit Cloud (Free)
```bash
streamlit deploy
```
Requires GitHub repo and Streamlit account.

### Docker
```bash
docker build -t rag-ui .
docker run -p 8501:8501 rag-ui
```

### AWS/Cloud
Same Docker image, deploy to:
- AWS App Runner
- Google Cloud Run
- Azure Container Instances
- DigitalOcean

---

## ✅ Next Steps

1. **Run the UI** (this page)
2. **Read Settings page** in UI for tuning options
3. **Check Analytics page** for performance metrics
4. **Review main README** for architecture
5. **Explore code** in `ui/app.py` for customization

---

## 📞 Support

Having issues? Check:
1. This file (SETUP.md)
2. UI README (ui/README.md)
3. Main README (README.md)
4. Debug section in UI (expand at bottom)
5. Troubleshooting in Settings page

---

**Ready?** Run: `streamlit run ui/app.py` 🚀
