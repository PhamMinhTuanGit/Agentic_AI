"""
Settings page for RAG Pipeline UI
Allows configuration of pipeline parameters and cache management
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Settings - RAG Pipeline",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Settings & Configuration")

# Check if pipeline is initialized
if "pipeline" not in st.session_state or st.session_state.pipeline is None:
    st.warning("⚠️ Pipeline not initialized. Please go to main page first.")
    st.stop()

# ==================== Pipeline Configuration ====================
st.markdown("## 🔧 Pipeline Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Retrieval Settings")
    st.info("""
    - **Top-K**: Number of documents to retrieve
    - Higher = more candidates, slower
    - Default: 10
    """)

with col2:
    st.markdown("### Reranking Settings")
    st.info("""
    - **Top-K**: Documents to keep after reranking
    - Higher = more thorough, slower
    - Default: 5
    """)

st.divider()

# ==================== Model Configuration ====================
st.markdown("## 🤖 Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Rerank Model")
    st.markdown("""
    **Current**: qwen2.5-coder:3b
    
    Options:
    - qwen2.5-coder:3b (faster, 3B params)
    - qwen2.5-coder:7b (better quality, 7B params)
    
    Model is used to:
    - Score document relevance
    - Generate answers
    """)

with col2:
    st.markdown("### Generation Model")
    st.markdown("""
    **Current**: qwen2.5-coder:3b
    
    Used for:
    - Answer generation from context
    
    Temperature: 0.1-0.3 (factual)
    Default: 0.7
    """)

st.divider()

# ==================== Cache Configuration ====================
st.markdown("## 💾 Cache Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Cache Settings")
    st.markdown("""
    **Backend**: SQLite
    **Location**: cache/rag_cache.db
    
    **TTL**: 24 hours (default)
    - How long to keep cached answers
    - Shorter = fresher answers
    - Longer = better hit rate
    
    **Storage**:
    - Query (SHA-256 hash)
    - Answer text
    - Sources list
    - Metadata
    """)

with col2:
    st.markdown("### Cache Statistics")
    if st.session_state.pipeline:
        try:
            stats = st.session_state.pipeline.get_stats()
            st.metric("Total Queries", stats.get('total_queries', 0))
            st.metric("Cache Hits", stats.get('cache_hits', 0))
            st.metric("Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}")
        except:
            st.warning("Could not load cache statistics")

st.divider()

# ==================== Performance Tuning ====================
st.markdown("## 🚀 Performance Tuning Guide")

with st.expander("Speed Optimization", expanded=False):
    st.markdown("""
    ### ⚡ Make it Faster
    
    1. **Reduce Reranker Top-K** (5 → 3)
       - Fewer documents to rerank
       - 30-40% faster
    
    2. **Use 3B Model instead of 7B**
       - Faster inference
       - Less memory
    
    3. **Increase Temperature to 0.9**
       - LLM generates faster
       - Less careful
    
    4. **Enable Cache**
       - Repeat queries are 600x faster
       - Best optimization
    
    **Typical**: Full pipeline ~11s → With cache optimization ~3s
    """)

with st.expander("Quality Optimization", expanded=False):
    st.markdown("""
    ### 🎯 Improve Answer Quality
    
    1. **Increase Retriever Top-K** (10 → 20)
       - Better candidates
       - More computation
    
    2. **Increase Reranker Top-K** (5 → 8)
       - More thorough evaluation
       - 20% slower
    
    3. **Use 7B Model**
       - Better understanding
       - Slower (2x)
    
    4. **Lower Temperature to 0.3**
       - More factual
       - Less creative
    
    5. **Update Embeddings**
       - Re-run: python -m ingest.embedder
       - Better semantic matching
    """)

with st.expander("Memory Optimization", expanded=False):
    st.markdown("""
    ### 💾 Reduce Memory Usage
    
    1. **Monitor FAISS Index Size**
       - ls -lh database/document/hybrid_docs_index.faiss
       - Larger = slower search but better recall
    
    2. **Clear Old Caches**
       - Remove: cache/rag_cache.db
       - Pipeline recreates it
    
    3. **Reduce Embedding Dimensions** (advanced)
       - Currently: 768 dims
       - Could reduce to 256
       - Would need to rebuild index
    
    4. **Use Streaming Generation**
       - See LLMClient in rag/llm_client.py
       - Generates response token-by-token
    """)

st.divider()

# ==================== Troubleshooting ====================
st.markdown("## 🐛 Troubleshooting")

with st.expander("Common Issues"):
    st.markdown("""
    ### Issue: Slow Responses
    - ✅ Enable cache (should be default)
    - ✅ Check cache hit rate in Analytics
    - ✅ Reduce reranker_top_k
    - ✅ Check system resources (CPU/RAM)
    
    ### Issue: Poor Answer Quality
    - ✅ Check retrieved documents
    - ✅ Increase retriever_top_k
    - ✅ Lower temperature (0.1-0.3)
    - ✅ Regenerate FAISS index
    
    ### Issue: "Connection refused"
    - ✅ Start Ollama: ollama serve
    - ✅ Check models: ollama list
    - ✅ Pull if missing: ollama pull qwen2.5-coder:3b
    
    ### Issue: Out of Memory
    - ✅ Clear cache
    - ✅ Restart Ollama
    - ✅ Reduce batch size
    - ✅ Monitor: top or Activity Monitor
    
    ### Issue: Cache not working
    - ✅ Check cache/rag_cache.db exists
    - ✅ Verify enable_cache=True
    - ✅ Check TTL (should be >60s)
    """)

st.divider()

# ==================== Advanced Settings ====================
st.markdown("## 🔬 Advanced Settings")

with st.expander("Developer Settings"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Embedding Settings**")
        st.info("""
        - Dense Model: nomic-embed-text
        - Dense Dims: 768
        - Sparse: TF-IDF (5000 features)
        - Sparse Dims (SVD): 768
        - Alpha: 0.7 (70% dense, 30% sparse)
        """)
    
    with col2:
        st.write("**Hybrid Retrieval**")
        st.info("""
        - Index Type: FAISS IndexFlatL2
        - Total Chunks: 2213
        - Chunk Size: 800 tokens
        - Min Chunk: 200 tokens
        - Similarity Threshold: 0.5
        """)

st.divider()

# ==================== About ====================
st.markdown("## ℹ️ About")

st.markdown("""
**RAG Pipeline UI v1.0**

Built with:
- 🐍 Python + Streamlit
- 🤖 Ollama (LLM inference)
- 🔍 FAISS (vector search)
- 💾 SQLite (caching)

**Architecture**:
- Query → Cache Check → Retrieve (top-10) → Rerank (top-5) → Generate → Cache Save

**Performance Target**:
- Cache HIT: 0.01s
- Cache MISS: 11s (full pipeline)
- Hit Rate: >50%

---

For detailed documentation, see: [README.md](../README.md)
""")
