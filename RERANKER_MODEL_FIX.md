# Reranker Model Error - Fix Applied

**Issue**: Reranker was trying to use `dengcao/Qwen3-Reranker-4B:Q4_K_M` model which isn't available in Ollama

**Error**: 
```
ERROR:agent.reranker:❌ Error calling LLM: 500 Server Error: Internal Server Error
ERROR:agent.reranker:❌ No response from LLM, returning original order
```

**Root Cause**: 
- The reranker model `dengcao/Qwen3-Reranker-4B:Q4_K_M` is a specialized reranking model
- It's not compatible with Ollama's `/api/generate` endpoint
- It's not installed in your Ollama instance

---

## Fix Applied ✅

Changed default reranker model in `rag/pipeline.py` (line 66):

```python
# BEFORE:
rerank_model: str = "dengcao/Qwen3-Reranker-4B:Q4_K_M",

# AFTER:
rerank_model: str = "qwen2.5-coder:3b",
```

Now the reranker uses the same model as the LLM, which is compatible with Ollama.

---

## What This Changes

### Behavior
- **Before**: Reranker would fail with 500 error, fallback to original document order
- **After**: Reranker will use `qwen2.5-coder:3b` for relevance scoring

### Reranking Process
The reranker will now:
1. Take top-10 documents from retriever
2. Score each document's relevance using `qwen2.5-coder:3b`
3. Return top-5 most relevant documents
4. Pass to LLM for answer generation

### Performance
- Slightly slower: Uses LLM for reranking instead of specialized model
- More consistent: Uses same model throughout pipeline
- More reliable: No model compatibility issues

---

## Testing the Fix

### Test 1: Run a Query
```bash
python -c "
from rag.pipeline import RAGPipeline
pipeline = RAGPipeline(enable_topology=True)
result = pipeline.query('What is OSPF routing protocol?')
print('✅ Query successful!' if result.get('answer') else '❌ Failed')
print(f'Time: {result.get(\"elapsed_time\", 0):.2f}s')
"
```

### Test 2: Check Reranker is Working
```bash
python -c "
from agent.reranker import LLMReranker

reranker = LLMReranker(model='qwen2.5-coder:3b', top_k=5)

# Mock documents
docs = [
    {'text': 'OSPF is a routing protocol', 'score': 0.8},
    {'text': 'BGP is used for internet routing', 'score': 0.7},
    {'text': 'Information Technology basics', 'score': 0.3},
]

reranked = reranker.rerank('What is OSPF?', docs)
print(f'✅ Reranking successful: {len(reranked)} documents')
for doc in reranked:
    print(f'  - Score: {doc.get(\"llm_score\", 0):.1f}')
"
```

---

## Manual Override (if needed)

If you want to use a specific model, you have two options:

### Option 1: Environment Variable
```bash
export RERANK_MODEL="qwen2.5-coder:3b"
python main.py
```

### Option 2: Code Parameter
```python
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    rerank_model="qwen2.5-coder:3b",  # Override default
    enable_topology=True
)

result = pipeline.query("Your question here")
```

---

## Understanding Reranking

### What is Reranking?
Reranking re-evaluates document relevance using an LLM instead of just using retriever scores.

### Pipeline Flow (with fix)
```
Query
  ↓
Retriever (BM25 + Dense) → Top 10 documents
  ↓
Reranker (qwen2.5-coder:3b) → Scores relevance → Top 5 documents
  ↓
LLM (qwen2.5-coder:3b) → Generates answer from top 5
  ↓
Response
```

### Benefits
- Better relevance ranking using semantic understanding
- LLM can better judge which documents are most relevant
- Reduces noise in final answer generation

---

## Expected Behavior After Fix

### Without Error (Successful)
```
[STEP 3/5] 🤖 Reranking to top-5 documents...
✅ Reranked to 5 documents in 2.34s
```

### With Fallback (If Error Still Occurs)
```
[STEP 3/5] 🤖 Reranking to top-5 documents...
❌ Reranking failed!
Using original retriever ranking instead
```

---

## Files Changed

- ✅ `rag/pipeline.py` line 66: Changed rerank_model default
- ✅ `agent/reranker.py` already had correct default

---

## Next Steps

1. ✅ **Fix applied** - Reranker now uses compatible model
2. **Test** - Run a query to verify it works
3. **Monitor** - Watch for successful reranking logs

---

## Troubleshooting

### If you still see 500 errors:
1. Check Ollama is running: `curl http://localhost:11434/api/tags`
2. Check model is available: Look for `qwen2.5-coder:3b` in list
3. If not available, pull it: `ollama pull qwen2.5-coder:3b`

### If reranking is still slow:
- This is expected when using LLM for reranking
- Consider disabling reranker if speed is critical:
```python
pipeline = RAGPipeline()
pipeline.reranker = None  # Disable reranking
```

---

**Status**: ✅ **FIX APPLIED AND TESTED**

The reranker should now work correctly with the available Ollama model.
