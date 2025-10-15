# RAG System Architecture

This document describes the advanced RAG (Retrieval-Augmented Generation) system implementation based on the provided architecture diagram.

## Architecture Components

### 1. **Document Processing Pipeline**
- **Documents** → **Chunks** → **Embedding (nomic-embed-text)**
  - Documents are split into chunks
  - Each chunk is embedded using the nomic-embed-text model
  - Embeddings and metadata are stored in FAISS database

### 2. **Query Processing Flow**

#### Step 1: User Query Input
- User submits a natural language question through the web UI
- Query parameters include:
  - `prompt`: The question text
  - `model`: LLM model to use (tinyllama, llama3.1:8b, mistral)
  - `use_reranker`: Enable/disable reranking
  - `top_k`: Number of documents to retrieve initially (default: 10)
  - `rerank_top_k`: Number of documents after reranking (default: 5)

#### Step 2: Historical Context (Cache)
- System loads conversation history from persistent cache
- History is limited to 512KB to prevent memory issues
- History includes previous Q&A pairs and context used

#### Step 3: Question Condensation
- **Input**: Original question + Historical context
- **Process**: LLM condenses the question into a standalone query
- **Output**: Improved query that captures all relevant context
- **Purpose**: Makes follow-up questions self-contained

#### Step 4: Semantic Search (Database Retrieval)
- Condensed question is embedded using the same model
- FAISS index performs vector similarity search
- Returns top-k most relevant document chunks
- Each result includes: document index, text, and distance score

#### Step 5: Reranking (Optional)
- **Input**: Retrieved top-k documents + original query
- **Process**: LLM scores each document on 0-10 relevance scale
- **Output**: Reranked top documents sorted by relevance
- **Purpose**: Improves precision by re-scoring with LLM understanding
- **Note**: Can be disabled for faster responses

#### Step 6: Context Building
- Combines reranked documents into a single context string
- Separates documents with clear delimiters

#### Step 7: Full Prompt Construction
- **Components**:
  1. **Context**: Retrieved and reranked document chunks
  2. **Conversation History**: Last 2000 characters of previous interactions
  3. **Question**: User's original prompt
- **Format**: Structured prompt with clear sections

#### Step 8: LLM Response Generation
- **Input**: Full prompt with context + history + question
- **Process**: Ollama API streams response from selected model
- **Output**: Generated answer based on provided context
- **Streaming**: Real-time response accumulation

#### Step 9: History Update
- Saves current Q&A interaction to cache
- Includes question, answer, and context used
- Automatically trims if exceeds 512KB limit

### 3. **Key Features**

#### Advanced Retrieval
- **Two-stage retrieval**: Initial retrieval + reranking
- **Vector similarity**: Fast approximate nearest neighbor search
- **Relevance scoring**: LLM-based document scoring

#### Context Management
- **Persistent cache**: Survives container restarts via Docker volume
- **Size limiting**: Auto-trims to maintain 512KB max
- **Session continuity**: Multi-turn conversations with memory

#### Query Enhancement
- **Question condensation**: Improves retrieval quality
- **History integration**: Understands conversation flow
- **Standalone queries**: Follow-ups work without context

#### User Controls
- **Reranker toggle**: Balance quality vs speed
- **Top-k tuning**: Adjust retrieval breadth
- **Model selection**: Choose appropriate LLM
- **History management**: Clear history when needed

## API Endpoints

### POST `/rag`
Main query endpoint with advanced RAG pipeline.

**Request Body**:
```json
{
  "prompt": "What is ZebOS?",
  "model": "tinyllama",
  "max_tokens": 256,
  "use_reranker": true,
  "top_k": 10,
  "rerank_top_k": 5
}
```

**Response**:
```json
{
  "answer": "ZebOS is a network operating system..."
}
```

### POST `/clear_history`
Clears conversation history cache.

**Response**:
```json
{
  "status": "History cleared"
}
```

### GET `/`
Web UI for interactive querying.

## Processing Steps Summary

```
User Query
    ↓
1. Load Historical Context (Cache)
    ↓
2. Condense Question (LLM) → Standalone Query
    ↓
3. Embed Query (nomic-embed-text)
    ↓
4. Semantic Search (FAISS) → Top-K Documents
    ↓
5. Rerank Documents (LLM Scoring) → Top Documents
    ↓
6. Build Context (Combine Documents)
    ↓
7. Construct Full Prompt (Context + History + Question)
    ↓
8. Generate Response (LLM via Ollama)
    ↓
9. Save to History Cache
    ↓
Return Answer to User
```

## Performance Considerations

- **Reranking**: Adds latency but improves quality significantly
- **Top-K values**: Higher values = more context but slower reranking
- **Model selection**: TinyLlama (fast) vs Llama 3.1 (accurate)
- **History size**: Trimmed automatically to prevent memory issues
- **Caching**: Persistent storage enables multi-session continuity

## Future Enhancements

- [ ] Hybrid search (keyword + semantic)
- [ ] Citation tracking (which docs were used)
- [ ] Confidence scoring
- [ ] Multi-modal support (images, tables)
- [ ] Advanced caching strategies (embeddings cache)
- [ ] Streaming responses in UI
- [ ] Query analytics and logging
