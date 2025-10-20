# 🎨 RAG Pipeline Streamlit UI

Professional interactive web interface for the RAG (Retrieval-Augmented Generation) pipeline built with Streamlit.

## ✨ Features

### 🎯 Main Features
- **Interactive Query Interface**: Ask questions about your documents in real-time
- **Smart Caching**: Automatically caches query results for 600x speedup on repeat queries
- **Real-time Metrics**: Performance monitoring and statistics
- **Document Visualization**: See retrieved and reranked documents with scores
- **Debug Mode**: Inspect raw results, query history, and system logs
- **Session Persistence**: Maintains query history and settings across sessions

### 🔧 Sidebar Configuration
- **Pipeline Settings**: Customize retriever top-k, reranker top-k, temperature
- **Model Selection**: Choose between Qwen models (3B/7B)
- **Cache Management**: Enable/disable cache, set TTL, clear cache
- **Live Statistics**: Real-time hit rate, latency, and token tracking
- **Query History**: Quick access to previous queries

### 📊 Advanced Features
- **Debug Console**: Raw JSON results, full query history, system logs

---

## 🚀 Installation & Setup

### 1. Prerequisites

```bash
# Ensure Ollama is running
ollama serve

# In another terminal, pull models if not already done
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b
```

### 2. Install UI Dependencies

```bash
cd Agentic

# Install Streamlit and visualization libraries
pip install -r ui/requirements.txt

# Or install individually
pip install streamlit>=1.28.0 streamlit-option-menu plotly pandas
```

### 3. Run the UI

```bash
# Option A: Using script
./run_ui.sh

# Option B: Direct command
streamlit run ui/app.py

# Option C: With custom port
streamlit run ui/app.py --server.port 8080

# Option D: With no browser auto-open
streamlit run ui/app.py --logger.level=debug --client.showErrorDetails=true
```

The UI will open at `http://localhost:8501`

---

## 📖 Usage Guide

### Main Interface

1. **Enter Question**: Type your question in the text input field
   - Example: "What is BGP protocol?"
   - Can be any question about your documents

2. **Click Submit**: Press the 🔍 Submit button to process
   - Shows real-time status indicator
   - "Processing..." → "Complete" or error

3. **View Results**:
   - **Answer**: Generated response from LLM (green box)
   - **Cache Status**: ✅ HIT or ❌ MISS badge
   - **Performance Metrics**: Time breakdown per stage
   - **Retrieved Documents**: Top 5 reranked + all retrieved (10)

### Sidebar Configuration

```
⚙️ CONFIGURATION
├─ 🔧 Pipeline Settings
│  ├─ Retriever Top-K: 10 (documents to retrieve)
│  ├─ Reranker Top-K: 5 (final documents)
│  ├─ LLM Temperature: 0.7 (creativity)
│  └─ Cache TTL: 86400s (24 hours)
│
├─ 🤖 Model Selection
│  ├─ Rerank Model: qwen2.5-coder:3b
│  └─ Generation Model: qwen2.5-coder:3b
│
├─ 💾 Cache Settings
│  └─ Enable Caching: [checkbox]
│
└─ 📊 Statistics
   ├─ Total Queries, Cache Hits, Hit Rate
   ├─ Avg Latency
   └─ Detailed Breakdown
```

### Debug Section

Expand "Debug Information" at the bottom to:
- View **Raw JSON**: Complete result data structure
- Check **Query History**: Table of all queries
- See **Logs**: Application logs
- Inspect **System Info**: Pipeline/Cache status

---

## 📊 Result Display

### Answer Section
```
💬 ANSWER
┌─────────────────────────────────┐
│ Generated answer from LLM       │
│ Based on reranked documents     │
│ Formatted for readability       │
└─────────────────────────────────┘
```

### Performance Metrics
```
⏱️ PERFORMANCE METRICS
┌─────────────┬──────────┬──────────┬──────────────┐
│ Total Time  │ Retrieval│ Reranking│ Generation   │
│ 11.32s      │ 0.32s    │ 2.95s    │ 8.05s        │
└─────────────┴──────────┴──────────┴──────────────┘
```

### Retrieved Documents
```
📄 RETRIEVED & RERANKED DOCUMENTS
Reranked Docs (Top-5) | All Retrieved Docs (Top-10)

Document 1: BGP Protocol Overview
Score: 0.892 | Source: documents/networking.pdf
"BGP is a routing protocol that operates at the application layer..."

Document 2: BGP Configuration
Score: 0.856 | Source: documents/bgp_setup.pdf
"To configure BGP, you need to set up neighbors..."
```

---

## ⚙️ Configuration Guide

### Performance Tuning

#### 🚀 Speed Optimization
```
Good for: Chat applications, real-time responses

Changes:
- Retriever Top-K: 10 → 10 (no change)
- Reranker Top-K: 5 → 3 (30% faster)
- Temperature: 0.7 → 0.9
- Cache TTL: Keep high (86400+)

Result: ~7s full pipeline (vs 11s)
```

#### 🎯 Quality Optimization
```
Good for: Accuracy, professional reports

Changes:
- Retriever Top-K: 10 → 20 (better candidates)
- Reranker Top-K: 5 → 8 (more thorough)
- Model: qwen2.5-coder:3b → 7b
- Temperature: 0.7 → 0.3 (more factual)

Result: Better answers (slower, ~18s)
```

#### 💾 Memory Optimization
```
Good for: Limited resources

Changes:
- Reranker Top-K: 5 → 3
- Clear cache: 🗑️ Clear All Cache button
- Monitor: Check Activity Monitor

Result: Lower memory footprint
```

### Cache Configuration

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| **Enable Cache** | True | Bool | Use query caching |
| **Cache TTL** | 86400s | 60-86400 | Seconds to keep cached answers |
| **Cache Backend** | SQLite | - | Persistent on disk |

Benefits:
- ✅ 600x speedup on repeat queries
- ✅ Reduces LLM API calls
- ✅ Better user experience
- ⚠️ May return stale answers

---

## 📱 Pages & Navigation

### 🏠 Home (main app.py)
- Main query interface
- Result display
- Debug console

**Navigate**: Streamlit single-page application

---

## 🐛 Troubleshooting

### Issue: "Pipeline not initialized"
```bash
# Solution: Ensure Ollama is running
ollama serve

# Check models
ollama list

# Should show:
# nomic-embed-text
# qwen2.5-coder:3b
```

### Issue: Slow responses
```
1. Check cache hit rate (Analytics page)
2. If hit rate low:
   - Run more similar queries
   - Increase Cache TTL
3. If still slow:
   - Reduce Reranker Top-K (5→3)
   - Use faster model (3B instead of 7B)
```

### Issue: "Connection refused"
```bash
# Solution: Start Ollama
ollama serve

# Verify with
curl http://localhost:11434/api/tags
```

### Issue: Poor answer quality
```
1. Check retrieved documents (see all 10)
2. Adjust Retriever Top-K up (10→20)
3. Lower Temperature (0.7→0.3)
4. Regenerate embeddings:
   python -m ingest.embedder
```

### Issue: Out of memory
```
1. Clear cache: 🗑️ Clear All Cache
2. Restart Ollama
3. Reduce Reranker Top-K
4. Monitor: top or Activity Monitor
```

---

## 🏗️ Architecture

```
ui/app.py (Main)
├── Sidebar Configuration
│   ├── Pipeline settings
│   ├── Model selection
│   └── Statistics
│
├── Main Query Interface
│   ├── Text input
│   ├── Submit button
│   └── Result display
│
├── Result Sections
│   ├── Answer (green box)
│   ├── Performance metrics
│   ├── Retrieved documents
│   └── Query metadata
│
└── Debug Console
    ├── Raw JSON
    ├── Query history
    ├── Logs
    └── System info
```

---

## 💻 System Architecture

```
Browser (Streamlit)
       ↓
Session State (query history, config)
       ↓
Main Interface (app.py)
       ↓
RAG Pipeline (rag/pipeline.py)
├─ Cache Check (rag/cache.py)
├─ Retrieve (agent/retriever.py)
├─ Rerank (agent/reranker.py)
├─ Generate (rag/llm_client.py)
└─ Cache Save (rag/cache.py)
       ↓
FAISS Index (database/document/)
SQLite Cache (cache/rag_cache.db)
Ollama API (localhost:11434)
```

---

## 📊 Performance Metrics

| Metric | Typical | With Cache |
|--------|---------|-----------|
| Query Response | ~11s | 0.01s (HIT) |
| Retrieval Time | ~0.3s | - |
| Reranking Time | ~3s | - |
| Generation Time | ~8s | - |
| Cache Hit Rate | - | >50% typical |

---

## 🔒 Security Notes

- ✅ All queries are hashed (SHA-256) before caching
- ✅ No sensitive data should be in query text
- ✅ Cache is local (SQLite in cache/rag_cache.db)
- ✅ No remote API calls except to Ollama
- ⚠️ For production: Use HTTPS, add authentication

---

## 📦 Dependencies

Main:
- `streamlit>=1.28.0` - Web UI framework
- `streamlit-option-menu>=0.3.6` - Sidebar navigation
- `plotly` - Interactive charts
- `pandas` - Data tables

Plus requirements from parent project:
- `torch`, `transformers`, `faiss-cpu`
- `requests`, `numpy`, `scikit-learn`
- All from `/requirements.txt`

---

## 🚀 Deployment

### Local Development
```bash
streamlit run ui/app.py
```

### Streamlit Cloud
```bash
# Deploy to Streamlit Cloud
streamlit deploy
```

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt -r ui/requirements.txt
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501"]
```

---

## 📚 File Structure

```
ui/
├── app.py                  # Main Streamlit application
├── __init__.py            # Package initialization
├── requirements.txt       # Streamlit dependencies
└── README.md              # This file
```

---

## 🎨 Customization

### Change Theme
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### Add Custom CSS
In `app.py`, add to `st.markdown()`:
```python
st.markdown("""
<style>
  /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

### Extend Pages
Add to `ui/pages/`:
```python
# new_page.py
import streamlit as st

st.title("New Page")
# Your content here
```

---

## ❓ FAQ

**Q: How does caching work?**
A: Queries are normalized, hashed with SHA-256, and stored in SQLite. Repeat queries return cached answers instantly.

**Q: Can I modify the retrieved documents?**
A: You can see them in the "Retrieved & Reranked Documents" section. Rerank scores show relevance ranking.

**Q: How do I export results?**
A: Results are in Debug → Raw Result JSON. You can copy/export as needed. Consider adding export feature.

**Q: Can I use different LLMs?**
A: Yes! Modify model selection in sidebar, then update RAGPipeline initialization.

**Q: Is this production-ready?**
A: Almost! For production, add: authentication, HTTPS, better error handling, rate limiting, audit logs.

---

## 🤝 Contributing

Ideas for improvements:
- [ ] Export results to PDF/JSON
- [ ] Query templates/examples
- [ ] Multi-user support with auth
- [ ] Real-time chart updates
- [ ] Query feedback/ratings
- [ ] Document upload interface
- [ ] Advanced search filters
- [ ] Response regeneration

---

## 📝 License

Same as parent project (MIT License)

---

## 🔗 Related

- [Main README](../README.md) - Full project documentation
- [RAG Pipeline Docs](../RAG_PIPELINE_DOCUMENTATION.md) - Technical details
- [Streamlit Docs](https://docs.streamlit.io/) - Streamlit reference

---

**Ready to use!** Run `streamlit run ui/app.py` and enjoy! 🚀
