# RAG Pipeline với Caching - Quick Start

## 🎯 Tổng Quan

Pipeline RAG hoàn chỉnh với caching để trả lời câu hỏi dựa trên documents, tối ưu hiệu suất và chi phí.

### ⚡ Highlights
- **Hybrid Retrieval**: Dense + Sparse embeddings
- **LLM Reranking**: Đánh giá lại độ liên quan
- **Intelligent Caching**: SQLite với TTL
- **Multiple Modes**: CLI, Interactive, Batch

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repo
cd Agentic

# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull models
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b
```

### 2. Prepare Documents

```bash
# Run embedding pipeline (if not done)
python -m ingest.embedder
```

### 3. Run Pipeline

#### Single Query (CLI Mode)
```bash
python main.py --query "What is BGP protocol?"
```

#### Interactive Mode
```bash
python main.py --interactive
```

#### Batch Mode
```bash
python main.py --batch example_questions.txt --output results.json
```

---

## 📊 Luồng Xử Lý

```
User Query
    ↓
┌─────────────┐
│ 1. Cache    │ → HIT  → Return (0.01s) ✨ FAST!
│    Check    │ → MISS → Continue ↓
└─────────────┘
    ↓
┌─────────────┐
│ 2. Retrieve │ → Top 10 docs (0.3s)
│   (Hybrid)  │
└─────────────┘
    ↓
┌─────────────┐
│ 3. Rerank   │ → Top 5 docs (3s)
│    (LLM)    │
└─────────────┘
    ↓
┌─────────────┐
│ 4. Generate │ → Answer (8s)
│    (LLM)    │
└─────────────┘
    ↓
┌─────────────┐
│ 5. Cache    │ → Save for next time
│    Save     │
└─────────────┘
    ↓
  Result
```

**Performance:**
- Cache HIT: ~0.01s (600x faster!)
- Cache MISS: ~6-15s (full pipeline)

---

## 📁 Cấu Trúc

```
Agentic/
├── rag/
│   ├── cache.py          # 💾 Caching với SQLite
│   ├── llm_client.py     # 🤖 LLM API client
│   └── pipeline.py       # 🔄 RAG orchestration
├── agent/
│   ├── retriever.py      # 🔍 Hybrid retrieval
│   └── reranker.py       # 📊 LLM reranking
├── ingest/
│   └── embedder.py       # 📚 Document embedding
├── main.py               # 🚀 Entry point
└── cache/
    └── rag_cache.db      # SQLite cache
```

---

## 🎨 Interactive Mode Commands

```bash
python main.py --interactive
```

**Available Commands:**
- Type question → Get answer
- `stats` → Pipeline statistics
- `cache` → Cache statistics  
- `clear` → Clear cache
- `help` → Show commands
- `quit` → Exit

---

## ⚙️ Configuration

### Environment Variables (.env)
```bash
LLM_API_URL=http://localhost:11434/api/generate
EMBEDDING_API_URL=http://localhost:11434/api/embeddings
RERANK_MODEL=qwen2.5-coder:3b
```

### Command Line Options
```bash
python main.py \
    --retriever-top-k 10 \      # Retrieve top-10
    --reranker-top-k 5 \        # Rerank to top-5
    --model qwen2.5-coder:3b \  # LLM model
    --temperature 0.7 \         # LLM temperature
    --no-cache                  # Disable cache
```

---

## 📊 Component Details

### 1. Cache Manager (cache.py)
- **Storage**: SQLite database
- **Key**: SHA-256 hash of normalized query
- **TTL**: Configurable (default: 24 hours)
- **Stats**: Hit/miss tracking

### 2. LLM Client (llm_client.py)
- **API**: Ollama (extensible to OpenAI)
- **Timeout**: 60s with retry
- **Prompt**: Structured with context
- **Metrics**: Tokens, latency tracking

### 3. RAG Pipeline (pipeline.py)
- **Flow**: Cache → Retrieve → Rerank → Generate → Cache
- **Stats**: Time breakdown per stage
- **Error Handling**: Graceful degradation

---

## 🎯 Best Practices

### Cache Management
✅ **DO**: Set appropriate TTL for your use case
✅ **DO**: Monitor hit rate (target >50%)
✅ **DO**: Clean up expired entries periodically
❌ **DON'T**: Cache sensitive/time-critical data

### Retrieval
✅ **DO**: Use K1=10 for retrieval, K2=5 for reranking
✅ **DO**: Monitor retrieval quality
❌ **DON'T**: Set K too high (slower, more noise)

### LLM Usage
✅ **DO**: Use structured prompts
✅ **DO**: Set low temperature (0.1-0.3) for factual answers
✅ **DO**: Track token usage
❌ **DON'T**: Generate without context

---

## 📈 Performance Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Cache Hit Rate | >50% | Depends on query patterns |
| Cache Hit Latency | <0.1s | SQLite lookup |
| Full Pipeline | <15s | Retrieval + Rerank + Gen |
| Token Usage | <2500 | Per query |

---

## 🐛 Troubleshooting

### "No documents retrieved"
```bash
# Check if FAISS index exists
ls database/document/hybrid_docs_index.faiss

# Regenerate if missing
python -m ingest.embedder
```

### "Connection refused" (Ollama)
```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

### "Cache errors"
```bash
# Clear cache
rm -rf cache/rag_cache.db

# Pipeline will recreate it
```

---

## 📚 Documentation

Xem chi tiết đầy đủ tại: [RAG_PIPELINE_DOCUMENTATION.md](RAG_PIPELINE_DOCUMENTATION.md)

Bao gồm:
- Luồng dữ liệu chi tiết
- Nhiệm vụ từng module
- Best practices
- Sơ đồ kiến trúc
- Performance tuning

---

## 🔧 Advanced Usage

### Custom System Prompt
```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Custom prompt for generation
system_prompt = """You are an expert in networking.
Answer concisely based on context only."""

result = pipeline.query(
    "What is BGP?",
    system_prompt=system_prompt
)
```

### Programmatic Usage
```python
from rag.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline(
    retriever_top_k=10,
    reranker_top_k=5,
    enable_cache=True
)

# Single query
result = pipeline.query("What is BGP?")
print(result['answer'])

# Batch processing
questions = ["Q1?", "Q2?", "Q3?"]
results = pipeline.batch_query(questions)

# Statistics
pipeline.print_stats()
```

---

## 🤝 Contributing

Contributions welcome! Priority areas:
- [ ] Redis cache backend
- [ ] FastAPI server mode
- [ ] Semantic caching
- [ ] More LLM providers
- [ ] Evaluation metrics

---

## 📝 License

MIT License

---

**Questions?** Check [RAG_PIPELINE_DOCUMENTATION.md](RAG_PIPELINE_DOCUMENTATION.md) for detailed explanations!
