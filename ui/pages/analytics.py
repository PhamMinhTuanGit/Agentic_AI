"""
Analytics page for RAG Pipeline UI
Shows statistics, performance charts, and cache analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Analytics - RAG Pipeline",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analytics & Performance Monitoring")

# Check if pipeline is initialized
if "pipeline" not in st.session_state or st.session_state.pipeline is None:
    st.warning("⚠️ Pipeline not initialized. Please go to main page first.")
    st.stop()

pipeline = st.session_state.pipeline

# Get statistics
stats = pipeline.get_stats()

# ==================== Key Metrics ====================
st.markdown("## 📈 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Queries",
        stats.get('total_queries', 0)
    )

with col2:
    st.metric(
        "Cache Hits",
        stats.get('cache_hits', 0)
    )

with col3:
    cache_hit_rate = stats.get('cache_hit_rate', 0)
    st.metric(
        "Hit Rate",
        f"{cache_hit_rate:.1%}"
    )

with col4:
    avg_time = stats.get('avg_total_time', 0)
    st.metric(
        "Avg Time",
        f"{avg_time:.2f}s"
    )

with col5:
    tokens = stats.get('total_tokens', 0)
    st.metric(
        "Total Tokens",
        f"{tokens:,}"
    )

st.divider()

# ==================== Performance Breakdown ====================
st.markdown("## ⏱️ Time Breakdown")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Avg Retrieval Time",
        f"{stats.get('avg_retrieval_time', 0):.3f}s",
        help="Average time to retrieve top-10 documents"
    )

with col2:
    st.metric(
        "Avg Rerank Time",
        f"{stats.get('avg_rerank_time', 0):.3f}s",
        help="Average time to rerank documents"
    )

with col3:
    st.metric(
        "Avg Generation Time",
        f"{stats.get('avg_generation_time', 0):.3f}s",
        help="Average time to generate answer"
    )

# ==================== Performance Charts ====================
st.markdown("## 📊 Performance Charts")

# Sample data for charts
if stats.get('total_queries', 0) > 0:
    # Create performance data
    performance_data = {
        'Stage': ['Retrieval', 'Reranking', 'Generation', 'Cache Lookup'],
        'Avg Time (s)': [
            stats.get('avg_retrieval_time', 0),
            stats.get('avg_rerank_time', 0),
            stats.get('avg_generation_time', 0),
            0.01  # Typical cache lookup time
        ]
    }
    
    df_perf = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            df_perf,
            x='Stage',
            y='Avg Time (s)',
            title="Average Time per Stage",
            color='Avg Time (s)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Cache hit/miss pie chart
        cache_data = {
            'Status': ['Cache Hits', 'Cache Misses'],
            'Count': [
                stats.get('cache_hits', 0),
                stats.get('total_queries', 0) - stats.get('cache_hits', 0)
            ]
        }
        df_cache = pd.DataFrame(cache_data)
        
        fig_pie = px.pie(
            df_cache,
            names='Status',
            values='Count',
            title="Cache Hit vs Miss",
            color_discrete_map={'Cache Hits': '#90EE90', 'Cache Misses': '#FFB6C6'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("💡 Run some queries to see performance charts")

st.divider()

# ==================== Detailed Statistics ====================
st.markdown("## 📋 Detailed Statistics")

with st.expander("Full Statistics JSON", expanded=False):
    st.json(stats)

# ==================== Performance Recommendations ====================
st.markdown("## 💡 Performance Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Hit Rate Optimization")
    
    hit_rate = stats.get('cache_hit_rate', 0)
    
    if hit_rate < 0.3:
        st.warning("⚠️ Low cache hit rate (<30%)")
        st.markdown("""
        - Run more similar queries
        - Increase cache TTL
        - Normalize queries better
        """)
    elif hit_rate > 0.7:
        st.success("✅ Great cache hit rate (>70%)")
    else:
        st.info("✓ Moderate cache hit rate (30-70%)")

with col2:
    st.markdown("### ⚡ Speed Optimization")
    
    avg_time = stats.get('avg_total_time', 0)
    
    if avg_time > 15:
        st.warning("⚠️ High average latency (>15s)")
        st.markdown("""
        - Reduce reranker_top_k (5→3)
        - Use faster model
        - Check system resources
        """)
    elif avg_time < 5:
        st.success("✅ Fast average latency (<5s)")
    else:
        st.info("✓ Acceptable latency (5-15s)")

st.divider()

# ==================== Query History Table ====================
st.markdown("## 📜 Query History")

if "query_history" in st.session_state and st.session_state.query_history:
    history_data = []
    for i, entry in enumerate(st.session_state.query_history):
        history_data.append({
            '#': i + 1,
            'Query': entry['query'][:60],
            'Timestamp': entry['timestamp'],
            'Cache Hit': '✅' if entry['cache_hit'] else '❌',
            'Docs Count': entry['sources_count']
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("No query history yet")
