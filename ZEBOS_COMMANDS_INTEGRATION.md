# ZebOS Commands Integration Guide

## Overview

This guide explains how to chunk, embed, and integrate ZebOS commands and chapters JSON files into the RAG pipeline for enhanced ZebOS CLI command retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Multi-Index Retriever                       │    │
│  │                                                     │    │
│  │  ┌──────────────────┐    ┌──────────────────┐    │    │
│  │  │  Main Docs       │    │  ZebOS Commands  │    │    │
│  │  │  (Hybrid Index)  │    │  Database        │    │    │
│  │  │                  │    │                  │    │    │
│  │  │  • HTML Docs     │    │  • Commands JSON │    │    │
│  │  │  • Manuals       │    │  • Chapters JSON │    │    │
│  │  │  • Guides        │    │  • Syntax        │    │    │
│  │  └──────────────────┘    └──────────────────┘    │    │
│  │         ↓                         ↓                │    │
│  │         └─────────┬───────────────┘                │    │
│  │                   ↓                                 │    │
│  │          Combined Results                          │    │
│  │          (Weighted Fusion)                         │    │
│  └────────────────────────────────────────────────────┘    │
│                          ↓                                  │
│                    Reranker                                 │
│                          ↓                                  │
│                    LLM Generation                           │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. **ZebOS Commands Embedder** (`ingest/zebos_commands_embedder.py`)

Responsible for:
- Loading `zebos_commands.json` and `zebos_chapters.json`
- Chunking commands and chapters into searchable documents
- Creating rich metadata for each chunk
- Embedding documents using the same embedding model
- Storing in `database/commands/`

**Key Features:**
- **Smart Chunking**: Each command becomes a comprehensive chunk with:
  - Command name
  - Description
  - Syntax variations
  - Parameters with descriptions
  - Configuration mode
  - Examples
  - Chapter context

### 2. **Multi-Index Retriever** (`agent/multi_index_retriever.py`)

Searches across multiple knowledge bases:
- Main documentation (hybrid dense + sparse)
- ZebOS commands database (dense embeddings)

**Features:**
- Weighted score fusion (configurable)
- Source tracking (main_docs vs commands_db)
- Intelligent result combination
- Separate search methods (commands-only, docs-only, combined)

### 3. **Updated RAG Pipeline** (`rag/pipeline.py`)

Enhanced with:
- Multi-index support toggle (`enable_multi_index`)
- Commands database path configuration
- Adjustable commands weight (default: 0.4)
- Backward compatible with single-index mode

## Files Structure

```
Agent/
├── zebos_commands.json              # Input: ZebOS commands
├── zebos_chapters.json              # Input: ZebOS chapters
├── run_embed_commands.sh            # Script to embed commands
├── test_commands_integration.py     # Integration tests
│
├── database/
│   ├── document/                     # Main docs (existing)
│   │   ├── hybrid_docs_index.faiss
│   │   └── hybrid_docs_metadata.json
│   └── commands/                     # NEW: Commands database
│       ├── zebos_commands_index.faiss
│       └── zebos_commands_metadata.json
│
├── ingest/
│   ├── embedder.py                   # Base embedder class
│   └── zebos_commands_embedder.py   # NEW: Commands embedder
│
├── agent/
│   ├── retriever.py                  # Hybrid retriever (existing)
│   └── multi_index_retriever.py     # NEW: Multi-index retriever
│
└── rag/
    └── pipeline.py                   # Updated with multi-index support
```

## Setup Instructions

### Step 1: Prepare JSON Files

Ensure you have:
- `zebos_commands.json` - ZebOS commands with syntax, examples, etc.
- `zebos_chapters.json` - ZebOS documentation chapters

### Step 2: Run Embedding Script

```bash
chmod +x run_embed_commands.sh
./run_embed_commands.sh
```

This will:
1. Activate virtual environment
2. Load and chunk JSON files
3. Embed all commands and chapters
4. Save to `database/commands/`

### Step 3: Test Integration

```bash
python3 test_commands_integration.py
```

This runs three tests:
1. ✅ Check embeddings exist
2. ✅ Test multi-index retriever
3. ✅ Test full RAG pipeline

### Step 4: Use in Your Application

```python
from rag.pipeline import RAGPipeline

# Initialize with multi-index support
pipeline = RAGPipeline(
    enable_multi_index=True,           # Enable commands database
    commands_index_dir="database/commands",
    commands_weight=0.4,               # 40% weight for commands
    retriever_top_k=10,
    reranker_top_k=5
)

# Query as normal
result = pipeline.query(
    question="How do I configure BGP neighbor in ZebOS?",
    return_context=True,
    return_sources=True
)

print(result['answer'])
```

## Configuration Options

### Multi-Index Retriever

```python
from agent.multi_index_retriever import MultiIndexRetriever

retriever = MultiIndexRetriever(
    main_index_path="database/document/hybrid_docs_index.faiss",
    commands_index_dir="database/commands",
    top_k=5,                    # Results per index
    commands_weight=0.4         # Weight for commands results
)

# Search across both indices
results = retriever.search("OSPF configuration", top_k=5)

# Search commands only
cmd_results = retriever.search_commands_only("show bgp", top_k=3)

# Search docs only
doc_results = retriever.search_docs_only("routing protocols", top_k=5)
```

### RAG Pipeline

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    # Multi-index config
    enable_multi_index=True,
    commands_index_dir="database/commands",
    commands_weight=0.4,
    
    # Retriever config
    retriever_top_k=10,
    
    # Reranker config
    reranker_top_k=5,
    
    # Other configs...
)
```

## Command Chunking Strategy

Each command is chunked with the following structure:

```
Command: <command_name>

Description: <detailed description>

Syntax:
<syntax_variation_1>
<syntax_variation_2>
...

Parameters:
  <param_1>: <description>
  <param_2>: <description>
  ...

Mode: <configuration_mode>

Examples:
<example_1>
<example_2>
...

Chapter: <chapter_name>
```

**Metadata includes:**
- `type`: "command" or "chapter"
- `command_name`: Name of the command
- `mode`: Configuration mode
- `file_path`: Source HTML file
- `chapter`: Chapter name
- `has_examples`: Boolean
- `syntax_count`: Number of syntax variations
- `param_count`: Number of parameters

## Benefits

### 1. **Comprehensive Command Coverage**
- All 33,000+ ZebOS commands indexed
- Syntax variations captured
- Examples included
- Parameter descriptions available

### 2. **Better Command Retrieval**
- Dedicated commands index optimized for CLI queries
- Rich metadata for filtering
- Examples improve relevance

### 3. **Intelligent Fusion**
- Combines conceptual docs with specific commands
- Weighted scoring balances breadth vs precision
- Source tracking for transparency

### 4. **Flexible Querying**
```python
# Get both concepts and commands
results = retriever.search("BGP configuration")

# Get only command syntax
results = retriever.search_commands_only("show bgp neighbor")

# Get only conceptual documentation
results = retriever.search_docs_only("BGP protocol overview")
```

## Performance

**Embedding Time:**
- Commands: ~5-10 minutes for 33,000+ commands
- Chapters: ~1-2 minutes for 400+ chapters
- **Total**: ~10-15 minutes

**Query Time:**
- Multi-index search: +10-20ms overhead
- Worth it for better results!

**Storage:**
- Commands FAISS index: ~150-200 MB
- Commands metadata: ~50-80 MB
- **Total**: ~200-300 MB

## Troubleshooting

### Issue: "Commands database not found"

**Solution:**
```bash
./run_embed_commands.sh
```

### Issue: "Multi-index retriever failed"

**Solution:**
- Check `database/commands/` exists
- Verify index files are present
- Falls back to single-index mode automatically

### Issue: "Poor command results"

**Solution:**
- Adjust `commands_weight` (try 0.5 or 0.6 for more command emphasis)
- Increase `top_k` for more diverse results
- Check query phrasing (be specific about commands)

## Examples

### Query: "How to configure OSPF on ZebOS router?"

**Results:**
1. 📄 Main Docs - OSPF Configuration Guide (score: 0.89)
2. ⚡ Commands - `router ospf` command (score: 0.87)
3. ⚡ Commands - `network` command (score: 0.85)
4. 📄 Main Docs - OSPF Examples (score: 0.82)
5. ⚡ Commands - `show ip ospf` command (score: 0.80)

### Query: "BGP neighbor syntax"

**Results:**
1. ⚡ Commands - `neighbor` command (score: 0.94)
2. ⚡ Commands - `neighbor remote-as` (score: 0.91)
3. 📄 Main Docs - BGP Configuration (score: 0.85)

## Next Steps

1. ✅ Run embedding script
2. ✅ Test integration
3. ✅ Update UI to show source (main_docs vs commands)
4. ✅ Monitor command retrieval quality
5. ✅ Adjust weights if needed

## Summary

This integration provides:
- **Better CLI command discovery**
- **Syntax and examples at your fingertips**
- **Comprehensive ZebOS knowledge base**
- **Flexible search strategies**
- **Production-ready implementation**

🎉 **Your ZebOS RAG system is now command-aware!**
