# 🚀 Agentic AI - RAG Pipeline với Hybrid Embeddings & Caching

Hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh với hybrid embeddings, intelligent caching, và LLM reranking để trả lời câu hỏi dựa trên tài liệu một cách nhanh chóng và chính xác.

---

## 📑 Mục Lục

1. [Tổng Quan](#-tổng-quan)
2. [Tính Năng Chính](#-tính-năng-chính)
3. [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
4. [Quick Start](#-quick-start)
5. [Hybrid Embeddings](#-hybrid-embeddings)
6. [RAG Pipeline](#-rag-pipeline)
7. [Cấu Hình & Tuning](#-cấu-hình--tuning)
8. [Troubleshooting](#-troubleshooting)

---

## 📊 Tổng Quan

Agentic AI là một pipeline RAG production-ready kết hợp:

### ⚡ Highlights
- **Hybrid Retrieval**: Kết hợp Dense (768-dim semantic) + Sparse (5000-dim TF-IDF) embeddings
- **LLM Reranking**: Sử dụng qwen2.5-coder:3b để đánh giá lại độ liên quan
- **Intelligent Caching**: SQLite cache với TTL, hit rate tracking, 600x tăng tốc
- **Multiple Modes**: CLI, Interactive, Batch processing
- **Semantic Chunking**: Chia document dựa trên ngữ nghĩa, không chỉ độ dài
- **Production Ready**: Logging, error handling, statistics, monitoring

---

## 🎯 Tính Năng Chính

### 1. **Hybrid PDF Embedding Pipeline**
- **Semantic Chunking**: Chia tài liệu thông minh dựa trên cosine similarity
- **Hybrid Embeddings**: Dense (nomic-embed-text) + Sparse (TF-IDF với SVD)
- **Alpha Blending**: Công thức `0.7 × dense + 0.3 × sparse`
- **FAISS Indexing**: Lưu trữ hiệu quả cho tìm kiếm vector nhanh

### 2. **Retriever-Reranker System**
- **HybridRetriever**: Tìm top-10 documents bằng hybrid search
- **LLMReranker**: Đánh giá lại độ liên quan với LLM, giữ top-5
- **Relevance Scoring**: Score 0-100 dựa trên semantic matching

### 3. **Intelligent Caching**
- **SQLite Backend**: Persistent cache với TTL management
- **SHA-256 Hashing**: Normalized query keys
- **Hit/Miss Tracking**: Statistics và performance monitoring
- **Auto-Cleanup**: Tự động xóa expired entries

### 4. **RAG Pipeline Orchestration**
- **5-Stage Processing**: Cache → Retrieve → Rerank → Generate → Cache
- **Context Building**: Tự động format context từ reranked docs
- **Prompt Engineering**: System + context + query formatting
- **Statistics Breakdown**: Timing per stage, token usage, cache metrics

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  CLI Mode (main.py)                                     │ │
│  │  - Interactive mode: prompt-based chat                  │ │
│  │  - Single query: --query "text"                        │ │
│  │  - Batch mode: --batch file.txt                        │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                   RAG Pipeline (pipeline.py)                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Stage 1: Cache Check (cache.py)                         │ │
│  │ ├─ SHA-256 hash of normalized query                    │ │
│  │ ├─ HIT: Return cached answer (0.01s) ✨               │ │
│  │ └─ MISS: Continue to retrieval                         │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Stage 2: Retrieve (agent/retriever.py)                 │ │
│  │ ├─ Hybrid search: dense + sparse                       │ │
│  │ ├─ Top-10 documents (0.3s)                            │ │
│  │ └─ Return: [doc1, doc2, ..., doc10]                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Stage 3: Rerank (agent/reranker.py)                    │ │
│  │ ├─ LLM evaluation of relevance                         │ │
│  │ ├─ Top-5 documents (3s)                               │ │
│  │ └─ Return: [best_doc1, best_doc2, ..., best_doc5]     │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Stage 4: Generate Answer (llm_client.py)               │ │
│  │ ├─ Build context from reranked docs                   │ │
│  │ ├─ System prompt + context + query                    │ │
│  │ ├─ LLM generation (8s)                                │ │
│  │ └─ Return: answer text                                │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Stage 5: Cache Save                                    │ │
│  │ ├─ Save query → answer mapping                        │ │
│  │ ├─ Store metadata + timestamp                         │ │
│  │ └─ Ready for next hit                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                   Storage Layer                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ FAISS Index (database/document/)                      │   │
│  │ - Dense embeddings (768 dims)                         │   │
│  │ - Sparse embeddings (768 dims after SVD)              │   │
│  │ - 2213 text chunks indexed                            │   │
│  └───────────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ SQLite Cache (cache/rag_cache.db)                     │   │
│  │ - Query → Answer mappings                             │   │
│  │ - SHA-256 keys, TTL support                           │   │
│  │ - Hit/miss statistics                                 │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Thư Mục

```
Agentic/
├── agent/
│   ├── retriever.py          # HybridRetriever - hybrid search
│   ├── reranker.py           # LLMReranker - semantic reranking
│   ├── request.py            # Request models
│   └── __pycache__/
├── rag/
│   ├── cache.py              # CacheManager - SQLite caching
│   ├── llm_client.py         # LLMClient - API client
│   ├── pipeline.py           # RAGPipeline - orchestration
│   ├── __init__.py           # Module exports
│   └── __pycache__/
├── ingest/
│   ├── embedder.py           # HybridPDFEmbedder - document processing
│   ├── __pycache__/
│   └── requirements.txt
├── database/
│   └── document/
│       ├── hybrid_docs_index.faiss      # FAISS vector index
│       ├── hybrid_docs_metadata.json    # Metadata + chunks
│       └── svd_transformer.pkl          # SVD model
├── cache/
│   └── rag_cache.db                     # SQLite cache
├── documents/                           # Input PDF folder
├── main.py                              # CLI entry point
├── requirements.txt                     # Python dependencies
├── README.md                            # This file (tổng hợp)
├── RAG_PIPELINE_DOCUMENTATION.md        # Detailed technical docs
├── HYBRID_EMBEDDING_README.md           # Embedding details (deprecated)
├── example_questions.txt                # Test queries
└── docker-compose.yml, Dockerfile, etc.
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# macOS with Homebrew
brew install python3 pip3

# Ollama (for embeddings & LLM)
brew install ollama
# or download from https://ollama.ai
```

### 2. Installation

```bash
cd Agentic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Start Ollama server (in a separate terminal)
ollama serve

# Pull required models (in another terminal)
ollama pull nomic-embed-text       # Dense embeddings (768-dim)
ollama pull qwen2.5-coder:3b       # LLM for reranking & generation
```

### 3. Prepare Documents

```bash
# Place PDF files in documents/ folder
cp /path/to/documents/*.pdf documents/

# Run embedding pipeline (creates FAISS index)
python -m ingest.embedder

# Output:
# ✅ Loaded 50 documents
# ✅ Created 2213 chunks
# ✅ Generated embeddings (2213 x 768)
# ✅ Built FAISS index
```

### 4. Run the Pipeline

#### **Option A: Interactive Mode** (Recommended for testing)
```bash
python main.py --interactive

# Commands:
# - Type any question and press Enter
# - stats     → Show pipeline statistics
# - cache     → Show cache statistics
# - clear     → Clear cache
# - quit      → Exit
```

#### **Option B: Single Query**
```bash
python main.py --query "What is BGP protocol?"

# Output:
# Query: What is BGP protocol?
# Answer: BGP (Border Gateway Protocol) is...
# Stats: Retrieval: 0.3s, Rerank: 3s, Generation: 8s
```

#### **Option C: Batch Processing**
```bash
python main.py --batch example_questions.txt --output results.json

# Processes all questions in file, saves results to JSON
```

---

## 🔍 Hybrid Embeddings

### Architecture

```
Document
  ↓
┌─────────────────────────────────────────┐
│        Semantic Chunking                │
│  (Cosine Similarity based)              │
│                                         │
│  - Split into sentences                 │
│  - Calculate TF-IDF vectors             │
│  - Compute cosine similarity            │
│  - Merge sentences w/ high similarity   │
│  - Result: ~800 token chunks            │
└─────────────────────────────────────────┘
  ↓                                   ↓
┌──────────────────┐          ┌──────────────────┐
│ Dense Embeddings │          │ Sparse Embeddings│
│                  │          │                  │
│ nomic-embed-text │          │ TF-IDF Vectorizer│
│ (768 dimensions) │          │ (5000 features)  │
│                  │          │                  │
│ Semantic meaning │          │ Keyword matching │
└──────────────────┘          └──────────────────┘
  ↓                                   ↓
┌──────────────────────────────────────────┐
│   SVD Dimension Alignment                │
│   (5000 → 768 dimensions)                │
│                                          │
│   TruncatedSVD(n_components=768)         │
└──────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────┐
│   Hybrid Embedding Combination           │
│                                          │
│   hybrid = 0.7 × dense + 0.3 × sparse   │
│                                          │
│   Result: 768-dimensional vector         │
└──────────────────────────────────────────┘
  ↓
┌──────────────────────────────────────────┐
│   FAISS Index (IndexFlatL2)              │
│                                          │
│   - 2213 chunks indexed                  │
│   - Fast L2 distance search              │
│   - Memory efficient                     │
└──────────────────────────────────────────┘
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 800 | Maximum tokens per chunk |
| `min_chunk_size` | 200 | Minimum tokens per chunk |
| `similarity_threshold` | 0.5 | Cosine similarity threshold (0-1) |
| `alpha` | 0.7 | Dense embedding weight (0-1) |
| `embedding_model` | nomic-embed-text | Dense embedding model |

### Tuning Guide

```python
# For better keyword matching (legal/technical docs)
embedder = HybridPDFEmbedder(
    alpha=0.5,  # More sparse
    similarity_threshold=0.3  # Larger chunks
)

# For better semantic matching (narratives/descriptions)
embedder = HybridPDFEmbedder(
    alpha=0.9,  # More dense
    similarity_threshold=0.7  # Smaller chunks
)
```

---

## 🔄 RAG Pipeline

### Pipeline Flow

```
Input Query
    ↓
    ├─ Stage 1: Cache Lookup
    │  ├─ Normalize query
    │  ├─ SHA-256 hash
    │  ├─ Check SQLite
    │  ├─ HIT → Return cached answer (0.01s) ✨
    │  └─ MISS → Continue
    │
    ├─ Stage 2: Retrieval
    │  ├─ Create hybrid embedding of query
    │  ├─ FAISS search (top-10)
    │  ├─ Return ranked documents
    │  └─ Time: ~0.3s
    │
    ├─ Stage 3: Reranking
    │  ├─ Score each doc with LLM
    │  ├─ Re-sort by relevance
    │  ├─ Keep top-5
    │  └─ Time: ~3s (for 5 evals)
    │
    ├─ Stage 4: Answer Generation
    │  ├─ Build context string
    │  ├─ Format system prompt
    │  ├─ Call LLM
    │  ├─ Generate answer
    │  └─ Time: ~8s
    │
    └─ Stage 5: Cache Save
       ├─ Save query-answer pair
       ├─ Set TTL (24 hours default)
       └─ Update statistics
           ↓
        Output Answer
```

### Performance

| Metric | Time | Notes |
|--------|------|-------|
| **Cache HIT** | ~0.01s | 600x faster! |
| **Retrieval** | ~0.3s | Hybrid FAISS search |
| **Reranking** | ~3s | 5 LLM evaluations |
| **Generation** | ~8s | LLM answer creation |
| **Cache MISS** | ~11.3s | Full pipeline |
| **Expected Hit Rate** | >50% | With typical usage |

### Code Example

```python
from rag.pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline(
    retriever_top_k=10,
    reranker_top_k=5,
    enable_cache=True
)

# Query
result = pipeline.query("What is BGP?")

# Access results
print(result['answer'])                    # Generated answer
print(result['sources'])                   # Top 5 reranked docs
print(result['timing'])                    # Stage timings
print(result['cache_hit'])                 # Boolean

# Statistics
pipeline.print_stats()
```

---

## ⚙️ Cấu Hình & Tuning

### Environment Variables (.env)

```bash
# LLM & Embedding APIs
LLM_API_URL=http://localhost:11434/api/generate
EMBEDDING_API_URL=http://localhost:11434/api/embeddings

# Model selection
RERANK_MODEL=qwen2.5-coder:3b
EMBEDDING_MODEL=nomic-embed-text

# Paths
FAISS_INDEX_PATH=database/document/hybrid_docs_index.faiss
METADATA_PATH=database/document/hybrid_docs_metadata.json
CACHE_DB_PATH=cache/rag_cache.db

# Parameters
CACHE_TTL=86400          # 24 hours
RETRIEVER_TOP_K=10
RERANKER_TOP_K=5
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=60
```

### Command Line Options

```bash
python main.py \
    --retriever-top-k 10 \           # Retrieve top-10 (default)
    --reranker-top-k 5 \             # Rerank to top-5 (default)
    --model qwen2.5-coder:3b \       # LLM model
    --temperature 0.7 \              # LLM temperature (0-1)
    --no-cache                       # Disable caching

# Interactive mode (default)
python main.py --interactive

# Single query
python main.py --query "Your question here?"

# Batch processing
python main.py --batch questions.txt --output results.json
```

### Performance Tuning

| Goal | Adjustment | Impact |
|------|------------|--------|
| **Faster responses** | ↓ `reranker_top_k` (5→3) | Faster but less accurate |
| **Better quality** | ↑ `retriever_top_k` (10→20) | Slower but better candidates |
| **More creative** | ↑ `temperature` (0.7→0.9) | More varied but less factual |
| **More factual** | ↓ `temperature` (0.7→0.3) | More consistent answers |
| **Reduce latency** | ↑ `cache_ttl` | Better hit rate |
| **Fresh answers** | ↓ `cache_ttl` | Real-time but slower |

---

## 🎨 Interactive Mode Commands

When running `python main.py --interactive`:

```
Available Commands:
─────────────────────────────────────────────
? <question>     → Ask a question (type naturally)
stats            → Show detailed pipeline statistics
cache            → Show cache statistics (hit rate, etc)
clear            → Clear all cached answers
help             → Show this help message
quit             → Exit the application

Example:
>>> What is Border Gateway Protocol?
Answer: BGP is a routing protocol that...
Stats: Cache: MISS, Retrieval: 0.3s, Rerank: 3s, Generation: 8s

>>> stats
Pipeline Statistics:
├─ Total Queries: 5
├─ Cache Hits: 2 (40%)
├─ Avg Retrieval: 0.32s
├─ Avg Rerank: 2.95s
├─ Avg Generation: 8.15s
└─ Total Tokens: 12,450

>>> cache
Cache Statistics:
├─ Entries: 3
├─ Hit Rate: 40%
├─ Memory: 2.3 MB
└─ Expired: 0
```

---

## 🐛 Troubleshooting

### Issue: "No documents retrieved"

**Symptom**: Empty or no results from retriever

```bash
# 1. Check if FAISS index exists
ls -lh database/document/hybrid_docs_index.faiss

# 2. If missing, regenerate
python -m ingest.embedder

# 3. Check document count
grep "total_chunks" database/document/hybrid_docs_metadata.json
```

### Issue: "Connection refused" (Ollama)

**Symptom**: `Error: Failed to connect to Ollama at http://localhost:11434`

```bash
# 1. Start Ollama server
ollama serve

# 2. In another terminal, verify it's running
curl http://localhost:11434/api/tags

# 3. Check models are installed
ollama list
# Should show:
# nomic-embed-text
# qwen2.5-coder:3b

# 4. If missing, pull them
ollama pull qwen2.5-coder:3b
```

### Issue: "LLMClient timeout" errors

**Symptom**: Generation takes too long, timeout at 60s

```bash
# 1. Increase timeout in config
# Edit main.py or use environment:
export LLM_TIMEOUT=120

# 2. Or use command line
python main.py --interactive  # Currently hardcoded, see llm_client.py

# 3. Check Ollama resource usage
# May need to give Ollama more CPU/RAM in Docker/settings
```

### Issue: "Cache errors" or corrupted DB

**Symptom**: SQLite errors when saving cache

```bash
# 1. Clear cache entirely
rm -f cache/rag_cache.db

# 2. Pipeline will auto-recreate on next run

# 3. Check permissions
ls -l cache/
chmod 755 cache/
```

### Issue: "Low retrieval quality"

**Symptom**: Wrong documents are being retrieved

```python
# 1. Adjust alpha (dense vs sparse balance)
# Edit ingest/embedder.py or regenerate:
from ingest.embedder import HybridPDFEmbedder

embedder = HybridPDFEmbedder(
    alpha=0.8,  # More dense (semantic)
    similarity_threshold=0.4
)
embedder.process_documents()
embedder.save_to_faiss()

# 2. Increase retriever_top_k for more candidates
python main.py --retriever-top-k 20

# 3. Adjust reranker_top_k to see top choices
python main.py --reranker-top-k 10
```

### Issue: "High memory usage"

**Symptom**: FAISS index or LLM using too much RAM

```bash
# 1. Check FAISS index size
ls -lh database/document/hybrid_docs_index.faiss

# 2. If very large, try:
# - Reduce document count
# - Use smaller embedding dimension (currently 768)
# - Implement index sharding

# 3. Monitor LLM memory
# Qwen 3B model uses ~4GB RAM when loaded
# Check system resources: top, Activity Monitor
```

---

## 📚 Additional Resources

### Detailed Documentation

- **[RAG_PIPELINE_DOCUMENTATION.md](RAG_PIPELINE_DOCUMENTATION.md)** - Complete technical guide
  - Detailed flow diagrams
  - Module responsibilities
  - Best practices
  - Performance tuning
  - Architecture decisions

### Example Usage

```bash
# See example_questions.txt for sample queries
cat example_questions.txt

# Test with batch mode
python main.py --batch example_questions.txt
```

### Code Examples

#### Single Query
```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
result = pipeline.query("What is BGP?")
print(result['answer'])
```

#### Batch Processing
```python
questions = [
    "What is BGP?",
    "How does OSPF work?",
    "Explain routing protocols"
]
results = pipeline.batch_query(questions)
for q, r in zip(questions, results):
    print(f"Q: {q}\nA: {r['answer']}\n")
```

#### Custom System Prompt
```python
system_prompt = """You are an expert network engineer.
Answer concisely with specific technical details.
Cite sources when relevant."""

result = pipeline.query(
    "What is BGP?",
    system_prompt=system_prompt
)
```

#### Cache Management
```python
from rag.cache import CacheManager

cache = CacheManager()

# Get statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")

# Cleanup expired entries
cache.cleanup_expired()

# Check specific query
cached = cache.get("What is BGP?")
if cached:
    print(f"Cached: {cached['answer']}")
```

---

## 🔧 Advanced Features

### Custom Retriever

```python
from agent.retriever import HybridRetriever

retriever = HybridRetriever(
    alpha=0.8,  # More weight on dense
    similarity_threshold=0.4
)

docs, scores = retriever.search("BGP protocol", top_k=10)
for doc, score in zip(docs, scores):
    print(f"{doc}: {score:.2f}")
```

### Custom Reranker

```python
from agent.reranker import LLMReranker

reranker = LLMReranker(model="qwen2.5-coder:3b")

# Rerank a list of documents
docs = [...]  # From retriever
reranked = reranker.rerank(
    query="What is BGP?",
    documents=docs,
    top_k=5
)
```

### Cache Statistics & Monitoring

```python
pipeline = RAGPipeline()

# After running queries...
stats = pipeline.get_stats()
print(f"""
Pipeline Statistics:
─ Total queries: {stats['total_queries']}
─ Cache hit rate: {stats['cache_hit_rate']:.1%}
─ Avg retrieval: {stats['avg_retrieval_time']:.2f}s
─ Avg rerank: {stats['avg_rerank_time']:.2f}s
─ Avg generation: {stats['avg_generation_time']:.2f}s
─ Total tokens: {stats['total_tokens']}
""")
```

---

## 🤝 Contributing

Interested in contributing? Priority areas:

- [ ] Redis cache backend (for distributed deployment)
- [ ] FastAPI server mode (REST API)
- [ ] Semantic caching (reduce redundant generations)
- [ ] More LLM providers (Claude, GPT, etc.)
- [ ] Evaluation metrics (BLEU, ROUGE, METEOR)
- [ ] Web UI (Streamlit/Gradio frontend)
- [ ] Docker Compose setup improvements
- [ ] Multilingual support

---

## 📝 License

MIT License - See LICENSE file for details

---

## ❓ FAQ

**Q: Why hybrid embeddings instead of just dense?**
A: Hybrid combines semantic understanding (dense) with keyword matching (sparse). This handles both conceptual and terminology-based queries better.

**Q: Can I use different LLMs?**
A: Yes! Modify `llm_model` parameter or change the API in `rag/llm_client.py`. Currently supports Ollama; can extend to OpenAI, Claude, etc.

**Q: How do I improve cache hit rate?**
A: Run more similar queries, increase TTL, and normalize queries better. Monitor hit rate with `stats` command.

**Q: Can I deploy this in production?**
A: Yes! See docker-compose.yml. For production scale, consider:
- Redis cache instead of SQLite
- FastAPI wrapper
- Load balancing
- Query logging & monitoring

**Q: What if I want to update documents?**
A: Re-run `python -m ingest.embedder` with new documents. It will regenerate the FAISS index.

---

**Have questions?** Open an issue or check [RAG_PIPELINE_DOCUMENTATION.md](RAG_PIPELINE_DOCUMENTATION.md)

**Latest Update**: October 19, 2025
