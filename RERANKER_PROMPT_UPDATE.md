# Reranker Prompt Update - Comprehensive RAG Evaluation

## Overview

Updated the `agent/reranker.py` to use a **comprehensive RAG-focused evaluation prompt** with detailed grading criteria and structured JSON output format.

## Changes Made

### 1. New Prompt Format

**Previous (Simple):**
```python
"""Score each document from 0-100. Return ONLY a JSON array of scores like:
[85, 45, 92, 30, 78, 55, 88, 40, 70, 50]"""
```

**New (Comprehensive):**
```python
"""You are a customer support answer service. Your task is to evaluate 
help center passages and score their relevance to a given customer query 
for a retrieval augmented generation (RAG) system.

Evaluation Process:
1. Analyze the customer's query to identify both explicit needs and implicit context
2. Assess each passage's ability to provide actionable guidance
3. Score based on how effectively the passage addresses the query's core intent

Grading Criteria:
10: EXCEPTIONAL match - Exact step-by-step instructions
9: NEAR-PERFECT solution - All critical steps, minor detail may be missing
8: STRONG MATCH - Complete technical resolution
7: GOOD MATCH - Substantial relevant details
6: PARTIAL match - General guidance but lacks specifics
5: LIMITED relevance - Related but indirect
4: TANGENTIAL - Mentions related concepts
3: VAGUE domain info - General area only
2: TOKEN overlap - Isolated terms without context
1: IRRELEVANT - Unrelated use of query terms
0: UNRELATED - No connection

Return JSON: {"id0":score0,"id1":score1,...}
Only include passages scoring 5+"""
```

### 2. Updated JSON Output Format

**Old Format (Array):**
```json
[85, 45, 92, 30, 78]
```

**New Format (Object):**
```json
{"id0":8,"id1":6,"id2":9}
```

**Key Differences:**
- Object keys are passage IDs (`id0`, `id1`, etc.)
- Scores are 0-10 scale (converted to 0-100 internally)
- Only includes passages scoring 5+ (filters out irrelevant docs)
- Empty object `{}` if no passages score 5+

### 3. Enhanced Parser

The `_parse_scores()` method now:
- ✅ Handles JSON object format: `{"id0":8, "id1":6, ...}`
- ✅ Handles JSON array format (backward compatibility)
- ✅ Converts 0-10 scale to 0-100 scale
- ✅ Assigns 0 score to passages not in response (scored <5)
- ✅ Maintains passage order (id0, id1, id2, ...)

### 4. Quality Threshold Filter (NEW)

The `rerank()` method now **filters out low-quality documents**:
- ✅ Only returns documents with `llm_score >= 50/100`
- ✅ Automatically excludes poor matches
- ✅ Returns empty list if no documents meet threshold
- ✅ Logs how many documents were filtered out

**Example:**
```python
# Before filtering: 5 docs with scores [85, 65, 52, 45, 20]
# After filtering: 3 docs with scores [85, 65, 52]
# Excluded: 2 docs (scores 45 and 20 below threshold)
```

## Files Modified

### `/home/tuanpm/work/Agent/agent/reranker.py`

**Modified Methods:**

1. **`_create_rerank_prompt(query, documents)`** - Lines 53-136
   - Removed simple prompt
   - Added comprehensive RAG evaluation framework
   - Formats documents as `<passage id='id0'>...</passage>`
   - Includes detailed grading scale (0-10)
   - Specifies JSON object output format

2. **`_parse_scores(response_text, num_documents)`** - Lines 138-213
   - Added `num_documents` parameter
   - Handles JSON object format with passage IDs
   - Converts 0-10 scores to 0-100 range
   - Assigns 0 to missing passages (scored <5)
   - Maintains backward compatibility with array format

3. **`rerank(query, documents, top_k)`** - Line 319
   - Updated to pass `len(documents)` to `_parse_scores()`

## Grading Scale Details

| Score | Category | Meaning |
|-------|----------|---------|
| 10 | EXCEPTIONAL | Exact step-by-step match, no ambiguity |
| 9 | NEAR-PERFECT | All critical steps, minor detail missing |
| 8 | STRONG MATCH | Complete resolution with simple inferences |
| 7 | GOOD MATCH | Substantial details, lacks one element |
| 6 | PARTIAL | General guidance, lacks specifics |
| 5 | LIMITED | Related but indirect |
| 4 | TANGENTIAL | Related concepts only |
| 3 | VAGUE | General area, no concrete steps |
| 2 | TOKEN | Isolated terms, no context |
| 1 | IRRELEVANT | Unrelated use of terms |
| 0 | UNRELATED | No connection |

**Note:** Only passages scoring 5+ are returned to the pipeline.

## Benefits

### 1. Better Relevance Assessment
- **Explicit criteria** for each score level
- Focuses on **actionable guidance** not just keyword matching
- Considers **implicit user needs** and context

### 2. Quality Filtering
- Automatically **excludes low-quality** passages (score <50/100)
- Returns **empty result** if no good matches found
- Prevents irrelevant content in RAG context
- **Dual threshold**: LLM excludes <5, reranker excludes <50

### 3. Structured Evaluation
- **Step-by-step** evaluation process
- Clear distinction between perfect/good/partial matches
- Emphasis on **completeness** and **applicability**

### 4. RAG-Optimized
- Designed for **retrieval augmented generation**
- Prioritizes passages that **resolve queries**
- Focuses on **technical accuracy** and **completeness**

## Example Usage

### Input Documents
```python
documents = [
    {'text': 'OSPF configuration: router ospf 1, network 10.0.0.0...', 'score': 0.85},
    {'text': 'BGP protocol uses AS numbers...', 'score': 0.45},
    {'text': 'OSPF uses area 0 as backbone...', 'score': 0.75}
]
query = "How do I configure OSPF area 0?"
```

### Expected Output
```python
# LLM returns: {"id0":9,"id2":7}
# (id1 BGP doc excluded, scored <5)

reranked = [
    {'text': 'OSPF configuration...', 'llm_score': 90.0, 'reranked_rank': 1},
    {'text': 'OSPF uses area 0...', 'llm_score': 70.0, 'reranked_rank': 2}
]
```

## Testing

### Run Test Script
```bash
python3 test_reranker_prompt.py
```

**Test Validates:**
- ✅ Prompt format is correct
- ✅ JSON parsing works (object + array formats)
- ✅ Score conversion (0-10 → 0-100)
- ✅ Relevant docs ranked higher
- ✅ Irrelevant docs excluded or ranked low

### Expected Behavior
1. OSPF configuration doc → **High score** (8-10 → 80-100)
2. OSPF concepts doc → **Medium score** (6-7 → 60-70)
3. BGP/unrelated docs → **Low/excluded** (0-4 → 0-40 or missing)

## Backward Compatibility

The parser maintains **backward compatibility**:

```python
# New format (preferred)
{"id0":8,"id1":6,"id2":9}  ✅ Handled

# Old format (still supported)
[80, 60, 90]  ✅ Handled
```

If LLM returns old array format, parser converts correctly.

## Integration

Works seamlessly with existing pipeline:

```python
# In RAGPipeline.query()
reranked_docs = self.reranker.rerank(question, retrieved_docs, top_k=5)

# Reranker now uses comprehensive RAG prompt
# Returns top documents with llm_score (0-100)
```

No changes needed in:
- ✅ `rag/pipeline.py` - Uses reranker as before
- ✅ `agent/retriever.py` - No dependency on reranker format
- ✅ `rag/llm_client.py` - Receives reranked docs same way

## Performance Considerations

### Prompt Size
- **Longer prompt** (~2KB vs ~200 bytes before)
- Includes detailed grading scale
- **Worth it** for better relevance assessment

### LLM Processing
- May take **slightly longer** due to detailed instructions
- More accurate scoring compensates for speed
- Use **temperature 0.1** for consistency

### Token Usage
- ~500-800 tokens for prompt
- Similar response tokens (JSON is compact)
- **Total increase**: ~300-500 tokens per rerank

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| Prompt | Simple scoring | Comprehensive RAG evaluation |
| Scale | 0-100 | 0-10 (converted to 0-100) |
| Format | Array `[85,45,92]` | Object `{"id0":8,"id1":6}` |
| LLM Filtering | None | Excludes docs scoring <5 (0-10 scale) |
| Reranker Filtering | None | Excludes docs scoring <50 (0-100 scale) |
| Criteria | Vague | Detailed 11-level scale |
| Focus | Generic relevance | RAG-specific actionability |
| Quality Control | Single threshold | Dual threshold (LLM + Reranker) |

**Status:** ✅ **Implemented and Ready**

**Files Changed:** 1 (`agent/reranker.py`)  
**Lines Changed:** ~180 lines  
**Backward Compatible:** Yes  
**Testing:** 2 test scripts created  

**Key Feature:** Documents with scores below 50/100 are **automatically excluded** from the context, ensuring only high-quality, relevant information reaches the LLM for answer generation.

---

**Date:** 2024-10-28  
**Version:** 2.1 - Comprehensive RAG Prompt + Quality Threshold
