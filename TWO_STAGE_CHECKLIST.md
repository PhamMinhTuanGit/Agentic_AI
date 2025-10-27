# Two-Stage Pipeline Implementation Checklist

## ✅ Implementation Complete

### Core Code Changes
- [x] Modified `rag/pipeline.py` - `query()` method rewritten
- [x] Added `_extract_command_mentions()` method
- [x] Added `_build_two_stage_context()` method
- [x] Updated logging to show two-stage workflow
- [x] Kept `_build_context()` for backward compatibility
- [x] No syntax errors detected

### Documentation Created
- [x] `TWO_STAGE_PIPELINE.md` - Complete documentation (15KB)
- [x] `TWO_STAGE_IMPLEMENTATION_SUMMARY.md` - Summary for quick reference
- [x] `TWO_STAGE_WORKFLOW_DIAGRAM.txt` - Visual workflow diagram
- [x] Code comments and docstrings

### Test Suite
- [x] `test_two_stage_pipeline.py` - Comprehensive test script
- [x] Test cases for OSPF, BGP, Interface configuration
- [x] Validation of command detection
- [x] Context structure verification
- [x] ZebOS syntax checking

### Features Implemented
- [x] Stage 1: Documentation retrieval with hybrid search
- [x] Reranking with LLM
- [x] Command detection using regex patterns
- [x] Stage 2: Commands database search
- [x] Combined context building (4 sections)
- [x] Integration with existing features (topology, CoT, cache)

## ⏳ Prerequisites to Test

### Required Files
Before running tests, ensure these exist:

```bash
# Commands database (CRITICAL)
database/commands/zebos_commands_index.faiss
database/commands/zebos_commands_metadata.json
database/commands/tfidf_vectorizer.pkl
database/commands/svd_transformer.pkl

# Main documentation database
database/document/hybrid_docs_index.faiss
database/document/hybrid_docs_metadata.json
database/document/tfidf_vectorizer.pkl
database/document/svd_transformer.pkl

# Network topology
network_stat/ring_topology.yaml
```

### Create Commands Database (if missing)

```bash
# Run embedding script
./run_embed_commands.sh

# Expected output:
# Loading zebos_commands.json...
# Loaded 1013 commands
# Loading zebos_chapters.json...
# Loaded 337 chapters
# Total documents: 1350
# Creating dense embeddings...
# Creating sparse embeddings...
# Saving to database/commands/...
# ✅ Done!
```

## 🧪 Testing Steps

### Step 1: Verify Database Files Exist

```bash
# Check main docs
ls -lh database/document/

# Check commands database
ls -lh database/commands/

# Both should have 4 files each:
# - *.faiss (FAISS index)
# - *_metadata.json (metadata)
# - tfidf_vectorizer.pkl (TF-IDF)
# - svd_transformer.pkl (SVD)
```

### Step 2: Run Test Suite

```bash
# Run comprehensive test
python test_two_stage_pipeline.py

# Expected output:
# - Pipeline initialization
# - Test case 1: OSPF Configuration
# - Test case 2: BGP Configuration
# - Test case 3: Interface Configuration
# - Statistics summary
```

### Step 3: Verify Two-Stage Workflow

Check logs for these indicators:

```
✅ Expected Log Output:
---
[STEP 2/6] 🔍 STAGE 1: Searching main documentation...
   Goal: Find relevant info & detect needed commands
✅ Retrieved 10 documents in 0.45s

[STEP 4/6] 🔎 Detecting commands mentioned in documentation...
✅ Detected 5 command(s): router ospf, network, area, interface, ipv4 address

[STEP 4/6] ⚡ STAGE 2: Searching commands database...
   Goal: Get exact syntax, parameters & examples
   Found 2 results for 'router ospf'
   Found 2 results for 'network'
✅ Retrieved 13 command documents in 0.38s

[STEP 5/6] 📝 Building combined context...
   Documentation chunks: 5
   Command chunks: 13
✅ Combined context built: 8542 characters
```

### Step 4: Validate Context Structure

Check that context includes all 4 sections:

```
✅ Expected Context Structure:
---
==================================================
NETWORK TOPOLOGY CONTEXT
==================================================

==================================================
DETECTED COMMANDS
==================================================

==================================================
DOCUMENTATION CONTEXT
==================================================

==================================================
COMMAND SYNTAX REFERENCE
==================================================
```

### Step 5: Verify ZebOS Syntax

Ensure answers use ZebOS (NOT Cisco IOS):

```
✅ ZebOS Syntax (CORRECT):
- configure
- router ospf {process-id}
- ipv4 address {ip} {mask}
- no shutdown

❌ Cisco IOS Syntax (WRONG):
- configure terminal
- router ospf {process-id}
- ip address {ip} {mask}
- no shut
```

## 🔧 Troubleshooting

### Issue: Commands Database Not Found

**Symptom:**
```
[STEP 4/6] ℹ️  STAGE 2: Skipped (no multi-index or no commands detected)
```

**Solution:**
```bash
# Create commands database
./run_embed_commands.sh

# Verify files created
ls database/commands/
```

### Issue: No Commands Detected

**Symptom:**
```
✅ Detected 0 command(s): 
```

**Solution:**
- Check if question mentions specific commands
- Review regex patterns in `_extract_command_mentions()`
- Try questions like:
  - "How to configure router ospf?"
  - "Configure BGP neighbor"
  - "Set IP address on interface"

### Issue: Wrong Syntax (Cisco vs ZebOS)

**Symptom:**
Answer contains Cisco IOS syntax like "configure terminal"

**Solution:**
- Verify `rag/cli_output_config.py` has ZebOS enforcement
- Check `rag/llm_client.py` system prompts
- Ensure commands database has ZebOS syntax

### Issue: Slow Performance (>10s)

**Symptom:**
Query takes very long time

**Solution:**
```python
# Reduce retrieval sizes
pipeline = RAGPipeline(
    retriever_top_k=5,    # Reduce from 10
    reranker_top_k=3,     # Reduce from 5
    enable_cot=False      # Disable CoT
)
```

## 📊 Success Criteria

### ✅ Pipeline Working Correctly If:

1. **Two-stage workflow executes**
   - Logs show "STAGE 1" and "STAGE 2"
   - Both stages retrieve documents

2. **Commands are detected**
   - Log shows detected commands list
   - Count > 0 for relevant questions

3. **Context structure correct**
   - Has all 4 sections
   - Documentation chunks present
   - Command syntax present

4. **ZebOS syntax used**
   - No "configure terminal"
   - Uses "ipv4 address" not "ip address"
   - Has ZebOS-specific commands

5. **Performance acceptable**
   - Total time: 5-7 seconds
   - No errors or timeouts

6. **Results accurate**
   - Answer addresses question
   - Includes relevant commands
   - Configuration is complete

## 🚀 Production Deployment

### Step 1: Update Main Application

```python
# In main.py or wherever pipeline is initialized
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    # Enable two-stage
    enable_multi_index=True,
    commands_index_dir="database/commands",
    
    # Standard config
    retriever_top_k=10,
    reranker_top_k=5,
    enable_cache=True,
    enable_topology=True,
    enable_cli_format=True,
    enable_cot=True
)
```

### Step 2: Monitor Performance

```python
# Check statistics after queries
stats = pipeline.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Avg time: {stats['total_time'] / stats['total_queries']:.2f}s")
print(f"Cache hit rate: {stats['cache_hits'] / stats['total_queries'] * 100:.1f}%")
```

### Step 3: Optimize as Needed

Based on usage patterns:
- Adjust `retriever_top_k` for speed vs accuracy
- Tune `commands_weight` (currently unused in two-stage)
- Enable/disable CoT based on accuracy needs
- Monitor cache hit rate

## 📝 Quick Reference

### Files Modified
```
rag/pipeline.py               # Main pipeline logic
```

### Files Created
```
test_two_stage_pipeline.py              # Test suite
TWO_STAGE_PIPELINE.md                   # Full documentation
TWO_STAGE_IMPLEMENTATION_SUMMARY.md     # Summary
TWO_STAGE_WORKFLOW_DIAGRAM.txt          # Visual diagram
TWO_STAGE_CHECKLIST.md                  # This file
```

### Key Methods
```python
# In RAGPipeline class
query()                          # Main entry point (rewritten)
_extract_command_mentions()      # Command detection (new)
_build_two_stage_context()       # Context building (new)
_build_context()                 # Legacy (kept for compatibility)
```

### Configuration
```python
RAGPipeline(
    enable_multi_index=True,           # REQUIRED for two-stage
    commands_index_dir="database/commands",
    retriever_top_k=10,                # Stage 1 retrieval
    reranker_top_k=5,                  # After reranking
    # ... other params
)
```

## ✅ Final Checklist

- [x] Code implementation complete
- [x] No syntax errors
- [x] Test suite created
- [x] Documentation written
- [x] Workflow diagram created
- [x] Backward compatibility maintained
- [ ] Commands database exists (CHECK BEFORE TESTING)
- [ ] Tests pass successfully (RUN test_two_stage_pipeline.py)
- [ ] Performance acceptable (<10s per query)
- [ ] ZebOS syntax enforced (no Cisco IOS)
- [ ] Ready for production use

## 🎯 Next Action

**RECOMMENDED:**

1. Check if commands database exists:
   ```bash
   ls database/commands/
   ```

2. If missing, create it:
   ```bash
   ./run_embed_commands.sh
   ```

3. Run test suite:
   ```bash
   python test_two_stage_pipeline.py
   ```

4. Review results and validate workflow

5. Deploy to production if tests pass

---

**Status:** ✅ Implementation Complete - Ready for Testing  
**Last Updated:** 2024  
**Version:** 1.0
