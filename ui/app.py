"""
Streamlit UI for RAG Pipeline - Simplified Chat Interface
Provides clean chat window with function buttons
"""

import streamlit as st
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
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
    page_title="🚀 RAG Chat Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2196f3;
    }
    .assistant-message {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4caf50;
    }
    .cache-hit {
        background-color: #c8e6c9;
        padding: 8px 12px;
        border-radius: 5px;
        color: #2e7d32;
        font-size: 0.9em;
        display: inline-block;
    }
    .cache-miss {
        background-color: #ffccbc;
        padding: 8px 12px;
        border-radius: 5px;
        color: #d84315;
        font-size: 0.9em;
        display: inline-block;
    }
    .timestamp {
        font-size: 0.85em;
        color: #999;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Session State Initialization ====================
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.cache_manager = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "loading" not in st.session_state:
    st.session_state.loading = False

# ==================== Cached Pipeline Initialization ====================
@st.cache_resource
def initialize_pipeline():
    """Initialize RAG pipeline (cached for performance)"""
    try:
        logger.info("🚀 Initializing RAG Pipeline...")
        
        # Fixed parameters (default values)
        pipeline = RAGPipeline(
            retriever_top_k=10,
            reranker_top_k=5,
            rerank_model="qwen2.5-coder:3b",
            llm_model="qwen2.5-coder:3b",
            llm_temperature=0.7,
            llm_max_tokens=2048,
            enable_cache=True,
            cache_dir="cache",
            cache_ttl_hours=24
        )
        
        # CacheManager uses ttl_hours parameter
        cache_manager = CacheManager(ttl_hours=24)
        
        logger.info("✅ Pipeline initialized successfully!")
        return pipeline, cache_manager
        
    except Exception as e:
        st.error(f"❌ Error initializing pipeline: {str(e)}")
        logger.error(f"Pipeline initialization error: {str(e)}", exc_info=True)
        return None, None

# ==================== Process Query ====================
def process_query(query: str) -> Dict[str, Any]:
    """Process user query through RAG pipeline"""
    
    try:
        start_time = time.time()
        result = st.session_state.pipeline.query(query)
        total_time = time.time() - start_time
        
        result['processed_at'] = datetime.now().isoformat()
        result['total_time'] = total_time
        
        logger.info(f"Query processed in {total_time:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        return None

# ==================== Sidebar ====================
def render_sidebar():
    """Render sidebar with function buttons"""
    st.sidebar.title("🎯 Functions")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.sidebar.button("📊 Stats", use_container_width=True):
            st.session_state.show_stats = not st.session_state.get('show_stats', False)
            st.rerun()
    
    with col2:
        if st.sidebar.button("📜 History", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
            st.rerun()
    
    with col3:
        if st.sidebar.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.sidebar.divider()
    
    # Show stats if requested
    if st.session_state.get('show_stats', False):
        st.sidebar.title("📈 Statistics")
        if st.session_state.pipeline:
            try:
                stats = st.session_state.pipeline.get_stats()
                st.sidebar.metric("Total Queries", stats.get('total_queries', 0))
                st.sidebar.metric("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.1%}")
                st.sidebar.metric("Avg Latency", f"{stats.get('avg_total_time', 0):.2f}s")
                st.sidebar.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
            except Exception as e:
                st.sidebar.warning(f"Could not load stats: {str(e)}")
    
    # Show history if requested
    if st.session_state.get('show_history', False):
        st.sidebar.title("📜 Query History")
        if st.session_state.messages:
            for i, msg in enumerate(reversed(st.session_state.messages)):
                if msg['role'] == 'user':
                    query_short = msg['content'][:40]
                    timestamp = msg.get('timestamp', '')
                    st.sidebar.caption(f"👤 {query_short}...")
                    if timestamp:
                        st.sidebar.caption(f"⏰ {timestamp}")
        else:
            st.sidebar.info("No chat history yet")

# ==================== Main App ====================
def main():
    """Main chat interface"""
    
    # Initialize pipeline
    pipeline, cache_manager = initialize_pipeline()
    
    if pipeline is None or cache_manager is None:
        st.error("❌ Failed to initialize RAG pipeline. Please check your configuration.")
        return
    
    st.session_state.pipeline = pipeline
    st.session_state.cache_manager = cache_manager
    
    # Render sidebar
    render_sidebar()
    
    # Header
    st.title("💬 RAG Chat Assistant")
    st.markdown("*Ask questions about your documents using AI-powered retrieval and generation*")
    st.divider()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                <b>👤 You:</b><br>
                {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                cache_indicator = "💾" if message.get('from_cache', False) else "🔄"
                time_taken = message.get('time_taken', 0)
                
                st.markdown(f"""
                <div class="assistant-message">
                <b>🤖 Assistant:</b><br>
                {message['content']}
                <div class="timestamp">
                {cache_indicator} {time_taken:.2f}s
                </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    
    # Input area with multiline support
    user_input = st.text_area(
        "Ask a question...",
        placeholder="e.g., What is BGP protocol? How to configure OSPF?\n\nYou can use multiple lines and press Ctrl+Enter to submit",
        label_visibility="collapsed",
        height=100,
        key="user_input"
    )
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        pass  # Empty space
    
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.user_input = ""
            st.rerun()
    
    with col3:
        submit_button = st.button("🔍 Send", use_container_width=True, type="primary")
    
    # Process input
    if submit_button and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input.strip(),
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        # Clear input field
        st.session_state.user_input = ""
        
        # Show processing status
        with st.spinner("🤔 Processing your question..."):
            result = process_query(user_input.strip())
            
            if result:
                answer = result.get('answer', 'No answer generated')
                from_cache = result.get('from_cache', False)
                time_taken = result.get('total_time', 0)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'from_cache': from_cache,
                    'time_taken': time_taken,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Rerun to display new messages and clear input
                st.rerun()
            else:
                st.error("❌ Failed to process your question. Please try again.")
    
    elif submit_button and not user_input.strip():
        st.warning("⚠️ Please enter a question first")

# ==================== Entry Point ====================
if __name__ == "__main__":
    main()
