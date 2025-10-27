# Two-Stage RAG Pipeline Documentation

## Overview

The RAG pipeline has been redesigned to use a **two-stage retrieval approach** that separates conceptual information retrieval from command syntax lookup. This provides more accurate and complete answers for ZebOS CLI configuration questions.

## Pipeline Architecture

### Previous Single-Stage Approach
```
Question → Retrieve (docs + commands mixed) → Rerank → Generate Answer
```

### New Two-Stage Approach
```
Question 
  ↓
  STAGE 1: Hybrid Search in Documentation
  ├─ Retrieve relevant conceptual information
  └─ Detect command keywords mentioned
  ↓
  Rerank Documentation Chunks
  ↓
  STAGE 2: Search Commands Database
  ├─ Query for detected commands
  ├─ Get exact syntax, parameters, examples
  └─ Retrieve additional related commands
  ↓
  Build Combined Context
  ├─ Network topology (if available)
  ├─ Detected commands summary
  ├─ Documentation context (conceptual)
  └─ Command syntax reference (exact)
  ↓
  Generate Answer with LLM
```

## Key Components

### 1. Stage 1: Documentation Retrieval

**Purpose:** Find conceptual information and detect what commands are needed

**Process:**
- Hybrid search (70% dense + 30% sparse) in main documentation
- Retrieve top-K documents (default: 10)
- Extract command keywords from documents and question

**File:** `rag/pipeline.py` - Lines 495-520

```python
# Always search main docs first (not multi-index)
if self.enable_multi_index and hasattr(self.retriever, 'main_retriever'):
    retrieved_docs = self.retriever.main_retriever.retrieve_with_scores(
        question, top_k=self.retriever_top_k
    )
```

### 2. Reranking

**Purpose:** Filter to most relevant documentation chunks

**Process:**
- LLM-based reranking of retrieved documents
- Reduce to top-K2 (default: 5)
- Uses qwen2.5-coder:3b for relevance scoring

**File:** `rag/pipeline.py` - Lines 533-541

### 3. Command Detection

**Purpose:** Extract ZebOS command mentions from reranked docs and question

**Patterns Detected:**
- Router protocols: `router ospf`, `router bgp`, `router rip`
- Interface commands: `interface ethernet`, `interface loopback`
- IP addressing: `ipv4 address`, `ipv6 address`
- BGP: `neighbor`, `bgp router-id`, `bgp network`
- OSPF: `ospf area`, `ospf network`, `passive-interface`
- Show commands: `show ip route`, `show bgp summary`
- General: `redistribute`, `network`, `no shutdown`

**File:** `rag/pipeline.py` - Method `_extract_command_mentions()` - Lines 244-301

### 4. Stage 2: Commands Database Search

**Purpose:** Get exact syntax, parameters, and examples for detected commands

**Process:**
- Search commands database for each detected command keyword
- Limit to top 10 detected commands
- Retrieve 2 results per command
- Also do general search with original question (top 3)
- Use hybrid search on commands FAISS index

**File:** `rag/pipeline.py` - Lines 548-572

```python
for cmd in detected_commands[:10]:
    cmd_query = f"{cmd} command syntax parameters examples"
    if hasattr(self.retriever, '_search_commands'):
        cmd_results = self.retriever._search_commands(cmd_query, top_k=2)
        commands_docs.extend(cmd_results)
```

### 5. Combined Context Building

**Purpose:** Merge documentation and command syntax into structured context

**Sections:**
1. **Network Topology Context** (if enabled)
   - Topology information from YAML file
   - Router/interface relationships

2. **Detected Commands Summary**
   - List of command keywords found
   - Helps LLM focus on relevant syntax

3. **Documentation Context**
   - Conceptual information from reranked docs
   - Protocol explanations, best practices
   - Source attribution

4. **Command Syntax Reference**
   - Exact ZebOS syntax
   - Parameter descriptions
   - Usage examples
   - Mode requirements

**File:** `rag/pipeline.py` - Method `_build_two_stage_context()` - Lines 303-380

### 6. Answer Generation

**Purpose:** Generate ZebOS CLI configuration with combined context

**Features:**
- Uses full combined context (docs + commands)
- Enforces ZebOS-only syntax (not Cisco IOS)
- Supports multiple output formats
- Optional Chain-of-Thought reasoning
- CLI formatting with code blocks

**File:** `rag/pipeline.py` - Lines 587-620

## Configuration

### Enable Two-Stage Retrieval

Set `enable_multi_index=True` when initializing the pipeline:

```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    # Multi-index config (REQUIRED for two-stage)
    enable_multi_index=True,
    commands_index_dir="database/commands",
    commands_weight=0.4,  # Not used in two-stage
    
    # Retriever config
    retriever_top_k=10,  # Stage 1 retrieval
    reranker_top_k=5,    # After reranking
    
    # LLM config
    llm_model="qwen2.5-coder:3b",
    llm_temperature=0.1,
    
    # Other features
    enable_cache=True,
    enable_topology=True,
    enable_cli_format=True,
    enable_cot=True
)
```

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retriever_top_k` | 10 | Documents to retrieve in Stage 1 |
| `reranker_top_k` | 5 | Documents after reranking |
| `enable_multi_index` | True | Enable two-stage retrieval |
| `commands_index_dir` | `database/commands` | Commands database path |
| `enable_cache` | True | Enable query caching |
| `enable_topology` | True | Include network topology |
| `enable_cot` | True | Use Chain-of-Thought |

## Usage Example

### Basic Query

```python
# Query the pipeline
result = pipeline.query(
    question="How do I configure OSPF area 0 on router R1?",
    return_context=True,
    return_sources=True
)

# Access results
print(result['answer'])
print(f"Time: {result['elapsed_time']:.2f}s")
print(f"From cache: {result['from_cache']}")
```

### Result Structure

```python
{
    'question': str,           # Original question
    'answer': str,             # Generated answer
    'from_cache': bool,        # Cache hit/miss
    'elapsed_time': float,     # Total time (seconds)
    'breakdown': {             # Time breakdown
        'retrieval': float,    # Stage 1 time
        'reranking': float,    # Reranking time
        'generation': float    # LLM generation time
    },
    'model': str,              # LLM model used
    'tokens': int,             # Total tokens
    'context': str,            # Combined context (if return_context=True)
    'sources': list            # Source documents (if return_sources=True)
}
```

## Testing

### Run Two-Stage Pipeline Test

```bash
python test_two_stage_pipeline.py
```

This test suite validates:
- Two-stage retrieval workflow
- Command detection accuracy
- Context structure (docs + commands)
- ZebOS syntax enforcement
- Pipeline timing and statistics

### Test Cases Included

1. **OSPF Configuration**
   - Tests: router ospf, network, area commands
   - Validates: OSPF protocol understanding

2. **BGP Configuration**
   - Tests: router bgp, neighbor, bgp router-id
   - Validates: BGP session setup

3. **Interface Configuration**
   - Tests: interface ethernet, ipv4 address, no shutdown
   - Validates: Basic interface setup

## Advantages of Two-Stage Approach

### 1. Separation of Concerns
- **Stage 1:** Conceptual understanding (what to configure)
- **Stage 2:** Exact syntax (how to configure)

### 2. Better Command Coverage
- Detects all mentioned commands
- Retrieves specific syntax for each
- Includes related commands

### 3. Improved Context Quality
- Clear separation of docs vs commands
- Structured format helps LLM
- Reduces hallucination

### 4. Flexible Weighting
- Commands don't compete with docs in same search
- Each stage optimized independently
- Better relevance signals

### 5. Easier Debugging
- Can see what commands were detected
- Inspect each retrieval stage separately
- Clear logging at each step

## Logging Output

The pipeline provides detailed logging at each stage:

```
======================================================================
📝 QUERY: How do I configure OSPF area 0 on router R1?
   Output format: multi_code_block
   Pipeline: Two-Stage Retrieval
======================================================================

[STEP 1/6] 💾 Checking cache...
❌ Cache MISS - Processing through two-stage pipeline...

[STEP 2/6] 🔍 STAGE 1: Searching main documentation...
   Goal: Find relevant info & detect needed commands
✅ Retrieved 10 documents in 0.45s

[STEP 3/6] 🤖 Reranking documentation to top-5...
✅ Reranked to 5 documents in 1.23s

[STEP 4/6] 🔎 Detecting commands mentioned in documentation...
✅ Detected 5 command(s): router ospf, network, area, interface, ipv4 address

[STEP 4/6] ⚡ STAGE 2: Searching commands database...
   Goal: Get exact syntax, parameters & examples
   Found 2 results for 'router ospf'
   Found 2 results for 'network'
   Found 2 results for 'area'
   Found 2 results for 'interface'
   Found 2 results for 'ipv4 address'
✅ Retrieved 13 command documents in 0.38s

[STEP 5/6] 📝 Building combined context...
   Documentation chunks: 5
   Command chunks: 13
✅ Combined context built: 8542 characters

[STEP 6/6] 💬 Generating answer with qwen2.5-coder:3b...
   Format: multi_code_block
✅ Answer generated in 3.45s

⏱️  Total pipeline time: 5.51s
   ├─ Retrieval: 0.45s (8.2%)
   ├─ Reranking: 1.23s (22.3%)
   └─ Generation: 3.45s (62.6%)
```

## Database Requirements

### Commands Database Structure

The commands database must exist at `database/commands/` with:

```
database/commands/
├── zebos_commands_index.faiss       # FAISS hybrid index
├── zebos_commands_metadata.json     # Document metadata
├── tfidf_vectorizer.pkl             # TF-IDF vectorizer
└── svd_transformer.pkl              # SVD transformer
```

### Generate Commands Database

If not already created:

```bash
./run_embed_commands.sh
```

This processes:
- `zebos_commands.json` (1013 commands)
- `zebos_chapters.json` (337 chapters)

Total: 1350 documents with hybrid embeddings

## Integration with Other Features

### Network Topology
Two-stage pipeline fully integrates with topology context:
- Topology info added to context (Section 1)
- Used by LLM for device-specific configs
- Router/interface awareness

### Chain-of-Thought
CoT reasoning works with two-stage context:
- Analyzes combined context (docs + commands)
- Reasons about both conceptual and syntax info
- Produces structured thinking trace

### CLI Output Formatting
Output formatting applies to generated answer:
- `default`: Natural language with inline code
- `single_code_block`: One unified code block
- `multi_code_block`: Separate blocks per device

### Caching
Cache works transparently with two-stage:
- Caches final answer (not intermediate stages)
- Cache key based on question
- TTL configurable (default: 24 hours)

## Performance Characteristics

### Typical Timing (on standard hardware)

| Stage | Time | Percentage |
|-------|------|------------|
| Cache check | <0.01s | <1% |
| Stage 1: Retrieval | 0.3-0.6s | 5-10% |
| Reranking | 1.0-1.5s | 20-25% |
| Stage 2: Commands | 0.3-0.5s | 5-10% |
| Context building | <0.1s | <2% |
| LLM generation | 3.0-4.0s | 60-70% |
| **Total** | **5-7s** | **100%** |

### Bottlenecks
- LLM generation: 60-70% of time
- Reranking: 20-25% of time
- Retrieval stages: 10-20% of time

### Optimization Tips
1. Enable caching for repeated queries
2. Reduce `reranker_top_k` if too slow
3. Use faster LLM model if available
4. Disable CoT for faster responses
5. Adjust `retriever_top_k` based on accuracy needs

## Troubleshooting

### No Commands Detected
**Symptom:** Stage 2 skipped, no command syntax in context

**Solutions:**
- Check if question mentions specific commands
- Verify regex patterns in `_extract_command_mentions()`
- Look at retrieved docs - do they mention commands?

### Commands Database Not Found
**Symptom:** `enable_multi_index=True` but Stage 2 fails

**Solutions:**
- Run `./run_embed_commands.sh` to create database
- Check `database/commands/` directory exists
- Verify all 4 files present (index, metadata, tfidf, svd)

### Wrong Syntax (Cisco vs ZebOS)
**Symptom:** Answer contains Cisco IOS commands

**Solutions:**
- Check `rag/cli_output_config.py` has ZebOS enforcement
- Verify `rag/llm_client.py` system prompts
- Ensure commands database has ZebOS syntax, not Cisco

### Slow Performance
**Symptom:** Query takes >10 seconds

**Solutions:**
- Check LLM model responsiveness
- Reduce `retriever_top_k` and `reranker_top_k`
- Disable CoT if not needed
- Enable caching
- Use GPU acceleration if available

## Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `rag/pipeline.py` | Main pipeline | Rewritten `query()` method for two-stage |
| `rag/pipeline.py` | Command detection | Added `_extract_command_mentions()` |
| `rag/pipeline.py` | Context building | Added `_build_two_stage_context()` |
| `test_two_stage_pipeline.py` | Testing | New test suite for two-stage workflow |
| `TWO_STAGE_PIPELINE.md` | Documentation | This file |

## Related Documentation

- [ZEBOS_COMMANDS_INTEGRATION.md](ZEBOS_COMMANDS_INTEGRATION.md) - Commands database setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [COT_SETUP_COMPLETE.md](COT_SETUP_COMPLETE.md) - Chain-of-Thought integration
- [RING_TOPOLOGY_PIPELINE_INTEGRATION.md](RING_TOPOLOGY_PIPELINE_INTEGRATION.md) - Topology integration

## Future Enhancements

### Potential Improvements
1. **Adaptive Command Detection**
   - Use LLM to detect commands instead of regex
   - Learn command patterns from user queries

2. **Command Relationship Graph**
   - Build dependency graph of commands
   - Suggest prerequisite commands automatically

3. **Confidence Scoring**
   - Score each stage's relevance
   - Return confidence metrics with answer

4. **Stage 2 Optimization**
   - Cache detected commands per query type
   - Pre-compute common command combinations

5. **Parallel Retrieval**
   - Run Stage 1 and Stage 2 in parallel
   - Use predicted commands from question analysis

---

**Last Updated:** 2024
**Version:** 1.0
**Status:** ✅ Production Ready
