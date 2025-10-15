# RAG System - Quick Start Guide

## What Changed

Your RAG system now implements an advanced architecture with the following enhancements:

### 1. **Question Condensation** 
- Automatically improves follow-up questions using conversation history
- Makes queries self-contained for better retrieval
- Example: "What about pricing?" → "What are the pricing details for ZebOS features?"

### 2. **Two-Stage Retrieval**
- **Stage 1**: Fast vector search retrieves top-K documents (default: 10)
- **Stage 2**: LLM-based reranking scores and selects best documents (default: 5)
- Result: Better precision and relevance

### 3. **Persistent Conversation History**
- Saves all Q&A interactions to disk (survives container restarts)
- Maximum 512KB with automatic trimming
- Enables multi-turn conversations with context awareness

### 4. **Enhanced UI**
New web interface with controls for:
- Enabling/disabling reranker
- Adjusting top-k values
- Selecting LLM model
- Clearing history

## How to Use

### Starting the System

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f rag_backend

# Access the UI
open http://localhost:8000
```

### Web Interface

1. **Ask Questions**: Type your query in the text area
2. **Adjust Settings**:
   - Toggle reranker for quality vs speed
   - Increase top-k for broader search
   - Change model for different quality/speed trade-offs
3. **Submit**: Click "Submit Query"
4. **Clear History**: Remove conversation context when starting a new topic

### API Usage

#### Basic Query
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is ZebOS?",
    "model": "tinyllama"
  }'
```

#### Advanced Query with All Options
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the routing protocols",
    "model": "llama3.1:8b",
    "use_reranker": true,
    "top_k": 15,
    "rerank_top_k": 7,
    "max_tokens": 512
  }'
```

#### Clear History
```bash
curl -X POST http://localhost:8000/clear_history
```

## Configuration Options

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Your question |
| `model` | string | "tinyllama" | LLM model (tinyllama, llama3.1:8b, mistral) |
| `max_tokens` | int | 256 | Maximum response length |
| `use_reranker` | bool | true | Enable LLM-based reranking |
| `top_k` | int | 10 | Initial documents to retrieve |
| `rerank_top_k` | int | 5 | Final documents after reranking |

### Model Selection Guide

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| tinyllama | ⚡⚡⚡ Fast | ⭐⭐ Good | Quick queries, testing |
| llama3.1:8b | ⚡⚡ Medium | ⭐⭐⭐⭐ Excellent | Production, complex questions |
| mistral | ⚡⚡ Medium | ⭐⭐⭐ Very Good | Balanced performance |

### Performance Tuning

**For Speed** (Real-time responses):
```json
{
  "model": "tinyllama",
  "use_reranker": false,
  "top_k": 5,
  "max_tokens": 128
}
```

**For Quality** (Best answers):
```json
{
  "model": "llama3.1:8b",
  "use_reranker": true,
  "top_k": 15,
  "rerank_top_k": 7,
  "max_tokens": 512
}
```

**Balanced** (Recommended):
```json
{
  "model": "tinyllama",
  "use_reranker": true,
  "top_k": 10,
  "rerank_top_k": 5,
  "max_tokens": 256
}
```

## Example Conversations

### Single Question
```
Q: What is OSPF?
A: OSPF (Open Shortest Path First) is a link-state routing protocol...
```

### Multi-Turn Conversation
```
Q: What is OSPF?
A: OSPF is a link-state routing protocol...

Q: How does it differ from RIP?
[System automatically understands "it" refers to OSPF due to history]
A: OSPF differs from RIP in several ways: it uses link-state instead of distance-vector...

Q: What are the configuration steps?
[System knows you're asking about OSPF configuration]
A: To configure OSPF, follow these steps...
```

## Troubleshooting

### Slow Responses
- Disable reranker: `"use_reranker": false`
- Reduce top_k values
- Use tinyllama model
- Check Ollama service is running

### Poor Quality Answers
- Enable reranker: `"use_reranker": true`
- Increase top_k and rerank_top_k
- Use better model (llama3.1:8b)
- Clear history if context is confusing

### Out of Context Responses
- Clear history and start fresh
- Rephrase question to be more specific
- Check if documents contain relevant information

### History Not Persisting
- Check Docker volume is mounted correctly
- Verify `/app/rag_backend/history/history.txt` is writable
- Check container logs for permissions errors

## Best Practices

1. **Clear History** when switching topics completely
2. **Use Reranker** for important/complex queries
3. **Increase top_k** if answers seem incomplete
4. **Use Better Models** for critical information
5. **Review Settings** before important queries

## Monitoring

Check the backend logs to see the RAG pipeline in action:

```bash
docker-compose logs -f rag_backend
```

You'll see:
- Condensed questions
- Number of documents retrieved
- Reranking status
- Prompt preview
- Response generation

## Architecture Flow

```
User Query → Load History → Condense Question → 
Embed Query → Vector Search (Top-K) → 
Rerank (if enabled) → Build Context → 
Generate Response → Save to History → Return Answer
```

Each step is logged and can be monitored for debugging and optimization.
