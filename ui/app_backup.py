"""
Streamlit UI for RAG Pipeline
Provides interactive interface for document question-answering system
"""

import streamlit as st
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline
from rag.cache import CacheManager

# ==================== Configure Logging ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Streamlit Page Config ====================
st.set_page_config(
    page_title="🚀 RAG Pipeline UI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .doc-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
    }
    .cache-hit {
        background-color: #c8e6c9;
        padding: 10px;
        border-radius: 5px;
        color: #2e7d32;
    }
    .cache-miss {
        background-color: #ffccbc;
        padding: 10px;
        border-radius: 5px;
        color: #d84315;
    }
    .debug-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State Initialization ====================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.cache_manager = None

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "loading" not in st.session_state:
    st.session_state.loading = False

if "error_message" not in st.session_state:
    st.session_state.error_message = None

if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# ==================== Cached Pipeline Initialization ====================
@st.cache_resource
def initialize_pipeline(
    retriever_top_k: int,
    reranker_top_k: int,
    rerank_model: str,
    llm_model: str,
    llm_temperature: float,
    enable_cache: bool,
    cache_ttl: int
) -> tuple:
    """Initialize RAG Pipeline and Cache Manager with caching"""
    try:
        st.info("🚀 Initializing RAG Pipeline... (this may take a moment)")
        
        # Convert seconds to hours for RAGPipeline
        cache_ttl_hours = cache_ttl // 3600
        
        pipeline = RAGPipeline(
            retriever_top_k=retriever_top_k,
            reranker_top_k=reranker_top_k,
            rerank_model=rerank_model,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            enable_cache=enable_cache,
            cache_ttl_hours=cache_ttl_hours
        )
        
        # CacheManager uses ttl_hours parameter
        cache_manager = CacheManager(ttl_hours=cache_ttl_hours)
        
        st.success("✅ Pipeline initialized successfully!")
        logger.info("RAG Pipeline initialized")
        
        return pipeline, cache_manager
    except Exception as e:
        st.error(f"❌ Error initializing pipeline: {str(e)}")
        logger.error(f"Pipeline initialization error: {str(e)}", exc_info=True)
        return None, None

# ==================== Sidebar Configuration ====================
def render_sidebar():
    """Render configuration sidebar"""
    st.sidebar.title("⚙️ Configuration")
    
    with st.sidebar.expander("🔧 Pipeline Settings", expanded=True):
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            retriever_top_k = st.number_input(
                "Retriever Top-K",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of documents to retrieve"
            )
            reranker_top_k = st.number_input(
                "Reranker Top-K",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of documents after reranking"
            )
        
        with col2:
            llm_temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more factual"
            )
            cache_ttl = st.number_input(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=86400,
                value=86400,
                step=3600,
                help="Time to live for cached queries"
            )
    
    with st.sidebar.expander("🤖 Model Selection"):
        rerank_model = st.selectbox(
            "Rerank Model",
            ["qwen2.5-coder:3b", "qwen2.5-coder:7b"],
            index=0,
            help="LLM model for reranking"
        )
        llm_model = st.selectbox(
            "Generation Model",
            ["qwen2.5-coder:3b", "qwen2.5-coder:7b"],
            index=0,
            help="LLM model for answer generation"
        )
    
    with st.sidebar.expander("💾 Cache Settings"):
        enable_cache = st.checkbox(
            "Enable Caching",
            value=True,
            help="Cache query-answer pairs"
        )
    
    st.sidebar.divider()
    
    # Statistics section
    st.sidebar.title("📊 Statistics")
    with st.sidebar.expander("Statistics", expanded=True):
        if st.session_state.pipeline:
            try:
                stats = st.session_state.pipeline.get_stats()
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric(
                        "Total Queries",
                        stats.get('total_queries', 0)
                    )
                    st.metric(
                        "Cache Hits",
                        f"{stats.get('cache_hits', 0)}"
                    )
                
                with col2:
                    cache_hit_rate = stats.get('cache_hit_rate', 0)
                    st.metric(
                        "Hit Rate",
                        f"{cache_hit_rate:.1%}"
                    )
                    st.metric(
                        "Avg Latency",
                        f"{stats.get('avg_total_time', 0):.2f}s"
                    )
                
                # Detailed stats
                with st.sidebar.expander("Detailed Breakdown"):
                    st.metric("Avg Retrieval", f"{stats.get('avg_retrieval_time', 0):.2f}s")
                    st.metric("Avg Rerank", f"{stats.get('avg_rerank_time', 0):.2f}s")
                    st.metric("Avg Generation", f"{stats.get('avg_generation_time', 0):.2f}s")
                    st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
            except Exception as e:
                st.sidebar.warning(f"Could not load stats: {str(e)}")
        else:
            st.sidebar.info("Initialize pipeline to see statistics")
    
    # Clear cache button
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear All Cache", use_container_width=True):
        if st.session_state.cache_manager:
            try:
                st.session_state.cache_manager.cleanup_expired(force=True)
                st.sidebar.success("✅ Cache cleared")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error clearing cache: {str(e)}")
    
    # Clear history button
    if st.sidebar.button("📋 Clear History", use_container_width=True):
        st.session_state.query_history = []
        st.session_state.current_result = None
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.caption("🔍 RAG Pipeline UI v1.0")
    st.sidebar.caption("Powered by Streamlit + Ollama")
    
    return retriever_top_k, reranker_top_k, rerank_model, llm_model, llm_temperature, enable_cache, cache_ttl

# ==================== Document Display ====================
def display_document(doc: Dict[str, Any], index: int):
    """Display a single retrieved document"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**Document {index + 1}: {doc.get('title', 'Untitled')}**")
    
    with col2:
        score = doc.get('score', 0)
        st.write(f"Score: `{score:.3f}`")
    
    with col3:
        source = doc.get('source', 'Unknown')
        st.write(f"Source: `{source}`")
    
    # Display snippet
    snippet = doc.get('snippet', '')
    if snippet:
        st.markdown(f"```\n{snippet[:300]}...\n```")
    
    st.divider()

# ==================== Query Processing ====================
def process_query(query: str) -> Dict[str, Any]:
    """Process query through RAG pipeline"""
    st.session_state.loading = True
    st.session_state.error_message = None
    
    placeholder = st.empty()
    placeholder.info("⏳ Processing query... (retrieving documents)")
    
    try:
        # Process through pipeline
        start_time = time.time()
        result = st.session_state.pipeline.query(query)
        total_time = time.time() - start_time
        
        # Add metadata
        result['processed_at'] = datetime.now().isoformat()
        result['total_time'] = total_time
        
        # Add to history
        history_entry = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': result.get('cache_hit', False),
            'sources_count': len(result.get('sources', []))
        }
        st.session_state.query_history.append(history_entry)
        
        # Store result
        st.session_state.current_result = result
        
        placeholder.success("✅ Query processed successfully!")
        logger.info(f"Query processed: {query[:50]}... in {total_time:.2f}s")
        
        return result
    
    except Exception as e:
        st.session_state.error_message = str(e)
        placeholder.error(f"❌ Error processing query: {str(e)}")
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        
        return None
    
    finally:
        st.session_state.loading = False

# ==================== Result Display ====================
def display_result(result: Dict[str, Any]):
    """Display query result"""
    
    # Answer section
    st.markdown("## 💬 Answer")
    answer_col1, answer_col2 = st.columns([3, 1])
    
    with answer_col1:
        answer = result.get('answer', 'No answer generated')
        st.markdown(f"""
        <div class="answer-box">
        {answer}
        </div>
        """, unsafe_allow_html=True)
    
    with answer_col2:
        # Cache status
        cache_hit = result.get('cache_hit', False)
        if cache_hit:
            st.markdown("""
            <div class="cache-hit">
            ✅ <b>Cache HIT</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="cache-miss">
            ❌ <b>Cache MISS</b>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Performance metrics
    st.markdown("## ⏱️ Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    timing = result.get('timing', {})
    
    with col1:
        st.metric(
            "Total Time",
            f"{result.get('total_time', 0):.2f}s"
        )
    
    with col2:
        st.metric(
            "Retrieval",
            f"{timing.get('retrieval_time', 0):.2f}s"
        )
    
    with col3:
        st.metric(
            "Reranking",
            f"{timing.get('rerank_time', 0):.2f}s"
        )
    
    with col4:
        st.metric(
            "Generation",
            f"{timing.get('generation_time', 0):.2f}s"
        )
    
    st.divider()
    
    # Retrieved documents
    st.markdown("## 📄 Retrieved & Reranked Documents")
    
    tab1, tab2 = st.tabs(["Reranked Docs (Top-5)", "All Retrieved Docs"])
    
    with tab1:
        sources = result.get('sources', [])
        if sources:
            for i, doc in enumerate(sources[:5]):
                display_document(doc, i)
        else:
            st.info("No documents retrieved")
    
    with tab2:
        all_docs = result.get('all_retrieved', [])
        if all_docs:
            for i, doc in enumerate(all_docs):
                display_document(doc, i)
        else:
            st.info("No documents in retrieval")
    
    st.divider()
    
    # Metadata
    st.markdown("## 📋 Query Metadata")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Token Count", result.get('token_count', 0))
    
    with col2:
        st.metric("Model Used", result.get('model', 'Unknown'))
    
    with col3:
        st.metric("Temperature", result.get('temperature', 0.7))

# ==================== Debug Section ====================
def render_debug_section():
    """Render debug information"""
    st.markdown("## 🔧 Debug Information")
    
    with st.expander("📝 Raw Result JSON"):
        if st.session_state.current_result:
            st.json(st.session_state.current_result)
        else:
            st.info("No result to display")
    
    with st.expander("📊 Query History"):
        if st.session_state.query_history:
            history_df_data = []
            for i, entry in enumerate(st.session_state.query_history):
                history_df_data.append({
                    '#': i + 1,
                    'Query': entry['query'][:50],
                    'Timestamp': entry['timestamp'],
                    'Cache Hit': '✅' if entry['cache_hit'] else '❌',
                    'Docs': entry['sources_count']
                })
            
            st.dataframe(history_df_data, use_container_width=True)
        else:
            st.info("No query history yet")
    
    with st.expander("🔍 Logs"):
        if st.session_state.debug_logs:
            st.code("\n".join(st.session_state.debug_logs), language="text")
        else:
            st.info("No logs yet")
    
    with st.expander("🛠️ System Info"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Pipeline Status:**")
            if st.session_state.pipeline:
                st.success("✅ Initialized")
            else:
                st.warning("⚠️ Not initialized")
        
        with col2:
            st.write("**Cache Status:**")
            if st.session_state.cache_manager:
                st.success("✅ Initialized")
            else:
                st.warning("⚠️ Not initialized")

# ==================== History Sidebar ====================
def render_history_sidebar():
    """Render query history in sidebar"""
    st.sidebar.divider()
    st.sidebar.title("📜 Query History")
    
    if st.session_state.query_history:
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            query_short = entry['query'][:40]
            cache_indicator = "✅" if entry['cache_hit'] else "❌"
            
            if st.sidebar.button(
                f"{cache_indicator} {query_short}...",
                key=f"hist_{i}",
                use_container_width=True
            ):
                st.session_state.selected_query = entry['query']
    else:
        st.sidebar.info("No query history yet")

# ==================== Main App ====================
def main():
    """Main Streamlit app"""
    
    # Title
    st.title("🚀 RAG Pipeline Interactive UI")
    st.markdown("**Ask questions about your documents using advanced RAG with hybrid retrieval and reranking**")
    
    # Render sidebar
    retriever_top_k, reranker_top_k, rerank_model, llm_model, llm_temperature, enable_cache, cache_ttl = render_sidebar()
    render_history_sidebar()
    
    # Initialize pipeline
    pipeline, cache_manager = initialize_pipeline(
        retriever_top_k=retriever_top_k,
        reranker_top_k=reranker_top_k,
        rerank_model=rerank_model,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
        enable_cache=enable_cache,
        cache_ttl=cache_ttl
    )
    
    if pipeline is None or cache_manager is None:
        st.error("❌ Failed to initialize RAG pipeline. Please check your configuration.")
        return
    
    st.session_state.pipeline = pipeline
    st.session_state.cache_manager = cache_manager
    
    # Main content
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is BGP protocol?",
            help="Ask any question about your documents"
        )
    
    with col2:
        submit_button = st.button(
            "🔍 Submit",
            use_container_width=True,
            type="primary"
        )
    
    st.divider()
    
    # Process query
    if submit_button and query:
        if not st.session_state.loading:
            result = process_query(query)
            
            if result:
                display_result(result)
            else:
                st.error(f"❌ {st.session_state.error_message}")
    
    elif submit_button and not query:
        st.warning("⚠️ Please enter a question first")
    
    # Show current result if available
    elif st.session_state.current_result and not submit_button:
        display_result(st.session_state.current_result)
    
    else:
        st.info("💡 Enter a question in the input field above and click 'Submit' to get started!")
    
    st.divider()
    
    # Debug section
    render_debug_section()

# ==================== Entry Point ====================
if __name__ == "__main__":
    main()
